"""Base agent — shared agentic tool-use loop using OpenAI-compatible API (Ollama / vLLM)."""

import json
import re
import sys
import os
import threading
from types import SimpleNamespace
from openai import OpenAI

# Global stop event — set this to interrupt any running agent loop
stop_event = threading.Event()

# Sentinel raised by _stream_llm when stop_event fires mid-stream
class _Stopped(Exception):
    pass

# Add project root to path so config is importable regardless of working dir
_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, MAX_TOKENS_TOOL, MAX_TOOL_CALLS

# Keep legacy names so any external code that imported these still works
VLLM_BASE_URL = OLLAMA_BASE_URL
MODEL_NAME    = OLLAMA_MODEL

MAX_TOOL_ITERATIONS = MAX_TOOL_CALLS  # prevent infinite loops if model misfires

# ---------------------------------------------------------------------------
# Multi-format tool call parser
# ---------------------------------------------------------------------------
# Maps known model hallucinations / alternative names → canonical tool names.
# Add entries here when a new model variant uses a different name.
_TOOL_NAME_ALIASES = {
    # Gemma 4 hallucinations
    "threshold_analysis":   "threshold_tuning",
    "threshold_sweep":      "threshold_tuning",
    "fp_fn_analysis":       "threshold_tuning",
    "fp_fn_tuning":         "threshold_tuning",
    "analyze_threshold":    "threshold_tuning",
    "alert_analysis":       "threshold_tuning",
    # Segmentation tool aliases — V26 model trained on ss_cluster_analysis, bridge to ds_cluster_analysis
    "ss_cluster_analysis":  "ds_cluster_analysis",
    "cluster_analysis":     "ds_cluster_analysis",
    "segment_analysis":     "ds_cluster_analysis",
    "segmentation_analysis":"ds_cluster_analysis",
    "segment_customers":    "ds_cluster_analysis",
    "segmentation_kmeans":  "ds_cluster_analysis",
    "sanctions_screening":  "ofac_screening",
    "ofac_check":           "ofac_screening",
    "sdn_screening":        "ofac_screening",
    "sar_analysis":         "sar_backtest",
    "sar_detection":        "sar_backtest",
    "backtest":             "sar_backtest",
}

# Maps alternative argument names → canonical argument names, per tool.
_ARG_ALIASES = {
    "threshold_tuning": {
        "customer_type":       "segment",
        "customer_segment":    "segment",
        "segment_type":        "segment",
        "transaction_amount":  "threshold_column",
        "amount_type":         "threshold_column",
        "column":              "threshold_column",
        "metric":              "threshold_column",
    },
    "sar_backtest": {
        "customer_type":       "segment",
        "customer_segment":    "segment",
        "column":              "threshold_column",
        "transaction_amount":  "threshold_column",
    },
    "rule_sar_backtest": {
        "rule":                "risk_factor",
        "rule_name":           "risk_factor",
        "risk_factor_name":    "risk_factor",
        "parameter":           "sweep_param",
        "param":               "sweep_param",
        "sweep_parameter":     "sweep_param",
    },
    "rule_2d_sweep": {
        "rule":                "risk_factor",
        "rule_name":           "risk_factor",
        "param_1":             "sweep_param_1",
        "param_2":             "sweep_param_2",
        "parameter_1":         "sweep_param_1",
        "parameter_2":         "sweep_param_2",
    },
}


def _normalize_tool_name(name: str) -> str:
    return _TOOL_NAME_ALIASES.get(name, name)


def _normalize_args(tool_name: str, args: dict) -> dict:
    aliases = _ARG_ALIASES.get(tool_name, {})
    return {aliases.get(k, k): v for k, v in args.items()}


def _parse_tool_call_from_content(content: str) -> tuple | None:
    """
    Fallback parser for when Ollama fails to extract a tool call from model output.
    Handles three model output formats:

    Gemma 4 native:    call:tool_name\\n{...}
    OpenAI-style JSON: {"name": "tool_name", "arguments": {...}}
    Qwen style:        <tool_call>\\n{"name": "...", "arguments": {...}}\\n</tool_call>

    Returns (tool_name, args_dict) or None.
    """
    if not content:
        return None

    # Format 1: Gemma 4 native — "call:tool_name" followed by a JSON block
    m = re.search(
        r'call:(\w+)\s*\n\s*(\{(?:[^{}]|\{[^{}]*\})*\})',
        content, re.DOTALL
    )
    if m:
        raw_name = m.group(1).strip()
        name = _normalize_tool_name(raw_name)
        try:
            args = json.loads(m.group(2))
            print(f"[base_agent] fallback parse (Gemma4 format): {raw_name} → {name}")
            return name, _normalize_args(name, args)
        except json.JSONDecodeError:
            pass

    # Format 2: Qwen style — <tool_call>{"name": ..., "arguments": ...}</tool_call>
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if "name" in obj and "arguments" in obj:
                raw_name = obj["name"]
                name = _normalize_tool_name(raw_name)
                args = obj["arguments"] if isinstance(obj["arguments"], dict) else {}
                print(f"[base_agent] fallback parse (Qwen format): {raw_name} → {name}")
                return name, _normalize_args(name, args)
        except json.JSONDecodeError:
            pass

    # Format 3: OpenAI-style JSON — {"name": "...", "arguments": {...}}
    m = re.search(
        r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})\s*\}',
        content, re.DOTALL
    )
    if m:
        raw_name = m.group(1).strip()
        name = _normalize_tool_name(raw_name)
        try:
            args = json.loads(m.group(2))
            print(f"[base_agent] fallback parse (OpenAI-JSON format): {raw_name} → {name}")
            return name, _normalize_args(name, args)
        except json.JSONDecodeError:
            pass

    # Format 7: Gemma 4 native tool_code — [<eos>]tool_code print(func(kwargs))
    m = re.search(r'(?:<eos>)?tool_code\s+print\((\w+)\((.*?)\)\)', content, re.DOTALL)
    if m:
        raw_name = m.group(1)
        name = _normalize_tool_name(raw_name)
        raw_kwargs = m.group(2)
        args = {}
        for km in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([\d.]+)|(True|False))', raw_kwargs):
            key = km.group(1)
            if km.group(2) is not None:
                val = km.group(2)
            elif km.group(3) is not None:
                raw_v = km.group(3)
                try:
                    val = float(raw_v) if '.' in raw_v else int(raw_v)
                except ValueError:
                    val = raw_v
            else:
                val = km.group(4) == 'True'
            args[key] = val
        print(f"[base_agent] fallback parse (Gemma4 tool_code format): {raw_name} → {name}, args={args}")
        return name, _normalize_args(name, args)

    # Format 8: backtick-wrapped function call — `func_name(kwargs)`
    _known_tools = {"threshold_tuning", "rule_sar_backtest", "rule_2d_sweep",
                    "list_rules", "search_policy_kb", "ds_cluster_analysis"}
    m = re.search(r'`(\w+)\(([^`]{0,400})\)`', content, re.DOTALL)
    if m and m.group(1) in _known_tools:
        raw_name = m.group(1)
        name = _normalize_tool_name(raw_name)
        raw_kwargs = m.group(2)
        args = {}
        for km in re.finditer(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\d+(?:\.\d+)?))", raw_kwargs):
            key = km.group(1)
            val = km.group(2) or km.group(3) or km.group(4)
            if km.group(4):
                try:
                    val = float(val) if '.' in val else int(val)
                except ValueError:
                    pass
            args[key] = val
        print(f"[base_agent] fallback parse (backtick format): {raw_name} → {name}, args={args}")
        return name, _normalize_args(name, args)

    # Format 5: natural language — "call [the] `tool_name`"
    _skip = {"the", "a", "an", "this", "that", "it", "my", "our"}
    for m in re.finditer(
        r'(?:call|use|invoke|using)\s+(?:(?:the|a|an)\s+)?[`"]?(\w+)[`"]?',
        content, re.IGNORECASE
    ):
        raw_name = m.group(1)
        if raw_name.lower() in _skip:
            continue
        name = _normalize_tool_name(raw_name)
        args = {}
        for km in re.finditer(r'[`"]?(\w+)[`"]?\s*[=:]\s*[`\'"]([^`\'"]+)[`\'"]', content):
            k, v = km.group(1), km.group(2)
            if k.lower() not in {"call", "name"} | _skip:
                args[k] = v
        print(f"[base_agent] fallback parse (NL format): {raw_name} → {name}, args={args}")
        return name, _normalize_args(name, args)

    return None


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: list, max_tool_calls: int = 1):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.model = OLLAMA_MODEL
        self.max_tool_calls = max_tool_calls

    def _stream_llm(self, **kwargs):
        """Call the LLM with stream=True and check stop_event on every token.

        Returns a SimpleNamespace(content, tool_calls) that mimics
        response.choices[0].message so the rest of run() is unchanged.
        Raises _Stopped if stop_event fires during streaming.
        """
        kwargs = {**kwargs, "stream": True}
        stream = self.client.chat.completions.create(**kwargs)
        content_parts = []
        tc_acc = {}  # index → {id, name, arguments}
        try:
            for chunk in stream:
                if stop_event.is_set():
                    stream.close()
                    raise _Stopped()
                choices = chunk.choices
                if not choices:
                    continue
                delta = choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_acc:
                            tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tc_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc_acc[idx]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc_acc[idx]["arguments"] += tc_delta.function.arguments
        finally:
            stream.close()

        content = "".join(content_parts) or None
        tool_calls = []
        for i in sorted(tc_acc.keys()):
            tc = tc_acc[i]
            fn = SimpleNamespace(name=tc["name"], arguments=tc["arguments"])
            tool_calls.append(SimpleNamespace(id=tc["id"] or f"tc_{i}", function=fn))
        return SimpleNamespace(content=content, tool_calls=tool_calls if tool_calls else None)

    def run(self, query: str, tool_executor, policy_context: str = "") -> tuple:
        """
        Agentic loop. Calls tools until the model returns a final text response.
        Returns: (response_text, [(tool_name, tool_input, fig), ...])
        """
        user_content = f"{policy_context}\n\n{query}".strip() if policy_context else query

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        chart_results = []
        tool_call_count = 0

        create_kwargs = dict(
            model=self.model,
            max_tokens=MAX_TOKENS_TOOL,
            messages=messages,
        )
        if self.tools:
            create_kwargs["tools"] = self.tools
            create_kwargs["tool_choice"] = "auto"

        for iteration in range(MAX_TOOL_ITERATIONS):
            if stop_event.is_set():
                stop_event.clear()
                return "Cancelled.", []
            try:
                msg = self._stream_llm(**create_kwargs)
            except _Stopped:
                stop_event.clear()
                return "Cancelled.", []

            # ── Determine tool calls to execute ──────────────────────────────
            # Primary path: Ollama parsed tool_calls correctly (well-trained model)
            # Fallback path: parse raw content for Gemma4 / Qwen / OpenAI-JSON formats
            structured_calls = []  # list of (name, args, tc_id)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    name = _normalize_tool_name(tc.function.name)
                    args = _normalize_args(name, args)
                    structured_calls.append((name, args, tc.id))

                # Record assistant turn with original tool_calls for context
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

            elif msg.content and tool_call_count < self.max_tool_calls:
                # Fallback: try to extract a tool call from raw content
                parsed = _parse_tool_call_from_content(msg.content)
                if parsed:
                    name, args = parsed
                    fake_id = f"fallback_{iteration}"
                    structured_calls.append((name, args, fake_id))
                    # Use a simplified message format compatible with all models
                    messages.append({"role": "assistant", "content": msg.content})

            # ── Execute tool calls ────────────────────────────────────────────
            if structured_calls:
                # Deduplicate: same tool + same args called twice in one response
                seen = set()
                deduped = []
                for item in structured_calls:
                    key = (item[0], str(item[1]))
                    if key not in seen:
                        seen.add(key)
                        deduped.append(item)
                if len(deduped) < len(structured_calls):
                    print(f"[{self.name}] deduplicated {len(structured_calls) - len(deduped)} duplicate tool call(s)")
                structured_calls = deduped
                for name, args, tc_id in structured_calls:
                    print(f"[{self.name}] tool call: {name}({args})")
                    result_text, fig = tool_executor(name, args)
                    if fig is not None:
                        chart_results.append((name, args, fig))

                    # Use tool role — matches Gemma 4 training format <|turn>tool
                    # Prefix matches training data: "Tool result for {name}:\n{content}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": f"Tool result for {name}:\n{result_text}",
                    })

                tool_call_count += 1
                create_kwargs["messages"] = messages
                if tool_call_count >= self.max_tool_calls:
                    # Remove tools entirely so Ollama's native parser can't fire again
                    create_kwargs.pop("tools", None)
                    create_kwargs.pop("tool_choice", None)

            else:
                # No tool call found — final text response
                return msg.content or "", chart_results

        # Exceeded max iterations — return whatever text we have
        print(f"[{self.name}] WARNING: hit MAX_TOOL_ITERATIONS ({MAX_TOOL_ITERATIONS})")
        return msg.content or "[No response after max tool iterations]", chart_results
