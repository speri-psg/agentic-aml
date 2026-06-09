"""Base agent — shared agentic tool-use loop using OpenAI-compatible API (Ollama / vLLM)."""

import json
import re
import sys
import os
import threading
import time
from types import SimpleNamespace
from openai import OpenAI

# Global stop event — set this to interrupt any running agent loop
stop_event = threading.Event()


def _strip_thinking(text: str) -> str:
    """Strip Gemma 4 thinking preamble and special tokens from response text."""
    if not text:
        return text
    # vLLM format: <think>...</think> (safety net — --reasoning-parser gemma4 may miss these)
    if "<think>" in text:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if cleaned:
            text = cleaned
        else:
            # </think> missing (truncated response) — keep only text before the tag
            idx = text.find("<think>")
            text = text[:idx].strip() if idx > 0 else text
    # Ollama format: "Thinking...\n...\n...done thinking.\n\n<answer>"
    if text.startswith("Thinking..."):
        marker = "...done thinking."
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):].strip()
        else:
            text = "\n".join(text.splitlines()[1:]).strip()
    # Legacy format: "Thinking Process:\n1. ...\n<answer>"
    elif text.startswith("Thinking Process:"):
        lines = text.splitlines()
        last_num_idx = -1
        for i, line in enumerate(lines):
            if re.match(r"^\d+\.", line.strip()):
                last_num_idx = i
        if last_num_idx == -1:
            text = "\n".join(lines[1:]).strip()
        else:
            answer = "\n".join(lines[last_num_idx + 1:]).strip()
            text = answer if answer else text
    return text

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

    # Format 1b: vLLM Gemma4 compact — call:name{key:val,key:val} (no newline, no JSON quotes)
    m = re.search(r'\bcall:(\w+)\{([^}]*)\}', content)
    if m:
        raw_name = m.group(1)
        name = _normalize_tool_name(raw_name)
        args = {}
        for pair in m.group(2).split(','):
            pair = pair.strip()
            if ':' in pair:
                k, v = pair.split(':', 1)
                # Strip Gemma 4 special quote tokens that appear when thinking is active
                k = re.sub(r'<\|[^|]*\|>', '', k).strip('" \'')
                v = re.sub(r'<\|[^|]*\|>', '', v).strip('" \'')
                if not k:
                    continue
                try:
                    args[k] = int(v)
                except ValueError:
                    try:
                        args[k] = float(v)
                    except ValueError:
                        args[k] = v
        print(f"[base_agent] fallback parse (compact call format): {raw_name} → {name}, args={args}")
        return name, _normalize_args(name, args)

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
    # Only match if the captured word is a known tool name or alias — prevents common English
    # words like "has", "algorithm", "function" from being mistaken for tool calls.
    #
    # Refusal/clarification gate (added 2026-06-08 PM): skip NL parsing entirely
    # when the content reads like a refusal or a clarification request. When the
    # model says "I do not have a rule named X, please use the list_rules tool"
    # or "I need a specific rule name, you can use list_rules to see them",
    # the tool name appears in a sentence addressed to the user — NOT a directive
    # to make a tool call. Extracting from refusals caused production misroutes:
    # SAR backtest queries for shorthand rule names ('Crypto', 'ACH', 'Mule')
    # landed on list_rules instead of rule_sar_backtest, producing ranking
    # garbage instead of an honest "I don't recognize that name" surfaced to the
    # user. (Diagnosed via _smoke_tool_choice_none.py — model emits refusal text,
    # old NL parser obediently extracted whatever tool name was mentioned.)
    # Surfacing the refusal directly is the model-not-tool honest behavior.
    _refusal_patterns = [
        r"\bi (?:do not|don't) have\b",
        r"\bi (?:cannot|can't) find\b",
        r"\bi (?:do not|don't) know\b",
        r"\bi(?:'m| am) not sure\b",
        r"\bi(?:'m| am) sorry\b",
        r"\bi need (?:a |an |the |to know |you to |some |more |you )",
        r"\bplease (?:provide|specify|use|tell|clarify)\b",
        r"\byou (?:can|could|may) use\b",
        r"\bif you (?:would|want|'d) like\b",
    ]
    if any(re.search(p, content, re.IGNORECASE) for p in _refusal_patterns):
        print(f"[base_agent] fallback parse (NL format): SKIPPED — refusal/clarification detected")
        return None

    _nl_valid_tools = _known_tools | set(_TOOL_NAME_ALIASES.keys()) | {
        "segment_stats", "cluster_rule_summary", "ofac_screening", "ofac_name_lookup",
        "sar_backtest",
    }
    _skip = {"the", "a", "an", "this", "that", "it", "my", "our"}
    for m in re.finditer(
        r'(?:call|use|invoke|using)\s+(?:(?:the|a|an)\s+)?[`"]?(\w+)[`"]?',
        content, re.IGNORECASE
    ):
        raw_name = m.group(1)
        if raw_name.lower() in _skip:
            continue
        name = _normalize_tool_name(raw_name)
        if raw_name.lower() not in _nl_valid_tools and name not in _nl_valid_tools:
            continue
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
        # Timing instrumentation — measure prefill (time-to-first-chunk) and total wall time
        _msgs = kwargs.get("messages", [])
        _msg_chars = sum(len(m.get("content") or "") for m in _msgs)
        _tools = kwargs.get("tools") or []
        _tools_chars = len(json.dumps(_tools)) if _tools else 0
        _t_start = time.perf_counter()
        _t_first_chunk = None
        stream = self.client.chat.completions.create(**kwargs)
        content_parts = []
        tc_acc = {}  # index → {id, name, arguments}
        try:
            for chunk in stream:
                if _t_first_chunk is None:
                    _t_first_chunk = time.perf_counter()
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
        # Timing summary — prefill ≈ ttfc, decode = total - ttfc
        _t_total = time.perf_counter() - _t_start
        _ttfc = (_t_first_chunk - _t_start) if _t_first_chunk else _t_total
        _decode = _t_total - _ttfc
        _out_chars = len(content or "") + sum(len(t["arguments"]) for t in tc_acc.values())
        print(
            f"[timing] {self.name} stream: total={_t_total:.2f}s "
            f"ttfc={_ttfc:.2f}s decode={_decode:.2f}s | "
            f"prompt_msgs={len(_msgs)}/{_msg_chars}ch tools={len(_tools)}/{_tools_chars}ch | "
            f"out={_out_chars}ch ({len(tool_calls)} tc)",
            flush=True,
        )
        return SimpleNamespace(content=content, tool_calls=tool_calls if tool_calls else None)

    def run(self, query: str, tool_executor, policy_context: str = "", history: list = None) -> tuple:
        """
        Agentic loop. Calls tools until the model returns a final text response.
        Returns: (response_text, [(tool_name, tool_input, fig), ...])
        """
        user_content = f"{policy_context}\n\n{query}".strip() if policy_context else query

        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        print(f"[base_agent] msgs to model ({len(messages)} total):", flush=True)
        for i, m in enumerate(messages):
            role = m.get("role", "?")
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls")
            if role == "system":
                print(f"  [{i}] system: ({len(content)} chars)", flush=True)
            elif tool_calls:
                tc = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
                name = tc.get("function", {}).get("name", "?") if isinstance(tc, dict) else getattr(getattr(tc, "function", None), "name", "?")
                print(f"  [{i}] assistant tool_call: {name}", flush=True)
            else:
                preview = content[:120].replace("\n", " ")
                print(f"  [{i}] {role}: {repr(preview)}", flush=True)

        chart_results = []
        tool_call_count = 0

        create_kwargs = dict(
            model=self.model,
            max_tokens=MAX_TOKENS_TOOL,
            temperature=0,
            messages=messages,
            extra_body={"think": False, "chat_template_kwargs": {"enable_thinking": False}},
        )
        if self.tools:
            create_kwargs["tools"] = self.tools
            create_kwargs["tool_choice"] = "none"

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

                # Record assistant turn with original tool_calls for context.
                # content=None — Ollama with think=True leaks thinking into delta.content
                # even when structured tool_calls are returned; storing it pollutes history
                # and causes the second LLM call to return 0 chars.
                messages.append({
                    "role": "assistant",
                    "content": None,
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
                    # Structure with tool_calls so the subsequent tool message is valid OpenAI
                    # format for both Ollama and vLLM (orphaned tool messages skip in vLLM template)
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": fake_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }],
                    })

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
                # When max_tool_calls=1, enforce single tool per response to prevent
                # spurious parallel calls contaminating chart results with stale data
                if self.max_tool_calls == 1 and len(structured_calls) > 1:
                    print(f"[{self.name}] capping to 1 tool call (was {len(structured_calls)}): keeping {structured_calls[0][0]}")
                    structured_calls = structured_calls[:1]
                for name, args, tc_id in structured_calls:
                    print(f"[{self.name}] tool call: {name}({args})")
                    result_text, fig = tool_executor(name, args)
                    if fig is not None:
                        chart_results.append((name, args, fig))

                    # Use tool role — matches Gemma 4 training format <|turn>tool
                    # Prefix matches training data: "Tool result for {name}:\n{content}"
                    # Trailing synthesis nudge — see feedback_instruction_placement: data-turn
                    # cues beat system-prompt rules. Forces analyst-style synthesis instead of
                    # verbatim echo, AND demands specific references (rule/segment/cluster names)
                    # — "be specific" moved here from the system prompt for stronger anchoring.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": (
                            f"Tool result for {name}:\n{result_text}"
                            f"\n\nSynthesize the observations above into 2-3 insightful sentences "
                            f"for the analyst. Be specific — name the rule, segment, cluster, "
                            f"parameter, and values explicitly rather than saying \"the rule\" or "
                            f"\"the data\". "
                            f"Plain text only — no LaTeX, no dollar-sign math wrappers. "
                            f"Write '>= 90%' or '90% or more', not '$\\ge 90\\%$'."
                        ),
                    })

                tool_call_count += 1
                create_kwargs["messages"] = messages
                if tool_call_count >= self.max_tool_calls:
                    # Remove tools entirely so Ollama's native parser can't fire again
                    create_kwargs.pop("tools", None)
                    create_kwargs.pop("tool_choice", None)

            else:
                # No tool call found — final text response.
                final_text = _strip_thinking(msg.content or "")

                # If model produced no synthesis text BUT a tool just ran, surface
                # the tool's content directly. Common case: rule-guard refusal
                # message went into the tool slot and the model returned 0 chars
                # in the synthesis pass — producing a silent 'No response' in the
                # UI. Better to show the (already user-friendly) refusal than
                # nothing. Strip the trailing synthesis nudge before surfacing.
                if not final_text.strip():
                    last_tool_msg = next(
                        (m for m in reversed(messages) if m.get("role") == "tool"),
                        None,
                    )
                    if last_tool_msg:
                        tool_content = (last_tool_msg.get("content") or "").strip()
                        # Strip the leading "Tool result for X:\n" prefix
                        tool_content = re.sub(
                            r"^Tool result for \w+:\s*", "", tool_content
                        )
                        # Strip the trailing synthesis-nudge block (added in the
                        # tool message — starts at the first 'Synthesize the
                        # observations above' sentinel).
                        tool_content = re.split(
                            r"\n\s*Synthesize the observations above",
                            tool_content, maxsplit=1,
                        )[0].strip()
                        if tool_content:
                            print(f"[{self.name}] empty model output — surfacing last tool content directly ({len(tool_content)} chars)")
                            return tool_content, chart_results
                    # No tool ran either — fall through to a generic hint
                    final_text = (
                        "I could not produce a response for that query. "
                        "Please rephrase — for rule-specific queries, try adding "
                        "the word 'rule' after the name (e.g., 'Show SAR backtest "
                        "for Velocity Single rule')."
                    )
                return final_text, chart_results

        # Exceeded max iterations — return whatever text we have
        print(f"[{self.name}] WARNING: hit MAX_TOOL_ITERATIONS ({MAX_TOOL_ITERATIONS})")
        return _strip_thinking(msg.content or "[No response after max tool iterations]"), chart_results
