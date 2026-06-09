"""Threshold Tuning Agent — FP/FN trade-off analysis across segments and columns."""

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from lambda_rule_analysis import RULE_CATALOGUE as _RC  # noqa: E402
from .base_agent import BaseAgent


def _build_rule_inventory() -> str:
    n = len(_RC)
    lines = [
        f"RULE INVENTORY — exactly {n} AML detection rules in this system. "
        "Use this directly to answer all count, name-list, categorization, and sweep-parameter questions. "
        "Do NOT call list_rules for these — only call list_rules when the user needs live SAR/FP/precision metrics from the dataset."
    ]
    for i, (_, entry) in enumerate(_RC.items(), 1):
        sweep = ", ".join(entry["sweep_params"].keys())
        lines.append(
            f"{i:2d}. {entry['name']:<45} current: {entry['current']} | sweep_params: {sweep}"
        )
    return "\n".join(lines)


_N_RULES = len(_RC)
_RULE_INVENTORY = _build_rule_inventory()

# OpenAI function-calling format (matches the fine-tuning training data)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "threshold_tuning",
            "description": (
                "Analyze false positive / false negative trade-offs as a threshold column is swept "
                "for a given customer segment. FP decreases and FN increases as the threshold rises. "
                "Returns a sweep table of FP and FN counts at each threshold step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "enum": ["Business", "Individual"],
                        "description": "Customer segment to analyze.",
                    },
                    "threshold_column": {
                        "type": "string",
                        "enum": ["AVG_TRXNS_WEEK", "AVG_TRXN_AMT", "TRXN_AMT_MONTHLY"],
                        "description": (
                            "Column to sweep as the alert threshold. "
                            "AVG_TRXNS_WEEK = average NUMBER of transactions per week (a count, not a dollar amount). "
                            "AVG_TRXN_AMT = average DOLLAR AMOUNT per transaction. "
                            "TRXN_AMT_MONTHLY = average total monthly transaction DOLLAR VOLUME. "
                            "Use AVG_TRXN_AMT when the user says 'transaction amount', 'average amount', or 'dollar amount'. "
                            "Use AVG_TRXNS_WEEK when the user says 'transaction count', 'number of transactions', or 'frequency'. "
                            "Use TRXN_AMT_MONTHLY when the user says 'monthly amount' or 'monthly volume'."
                        ),
                    },
                },
                "required": ["segment", "threshold_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "segment_stats",
            "description": (
                "Return summary statistics (total accounts, alerts, false positives, false negatives) "
                "broken down by Business and Individual segments."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sar_backtest",
            "description": (
                "SEGMENT-LEVEL SAR backtest. Sweep a customer-attribute threshold "
                "(monthly transaction amount, weekly transaction count, etc.) across "
                "an ENTIRE customer segment (Business or Individual) — NOT for a "
                "specific AML rule. Use ONLY when the user names a customer segment "
                "and a threshold column WITHOUT naming a specific rule. Examples: "
                "'SAR backtest for Business customers on monthly transaction amount', "
                "'Show SAR catch rate for Individual customers', "
                "'Backtest threshold for Business segment'. "
                "If the user names a specific AML rule (Elder Abuse, Velocity Single, "
                "Mule, Crypto, etc.), use rule_sar_backtest instead, NOT this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "enum": ["Business", "Individual"],
                        "description": "Customer segment to analyze.",
                    },
                    "threshold_column": {
                        "type": "string",
                        "enum": ["AVG_TRXNS_WEEK", "AVG_TRXN_AMT", "TRXN_AMT_MONTHLY"],
                        "description": (
                            "Column to sweep as the alert threshold. "
                            "AVG_TRXNS_WEEK = average NUMBER of transactions per week (a count, not a dollar amount). "
                            "AVG_TRXN_AMT = average DOLLAR AMOUNT per transaction. "
                            "TRXN_AMT_MONTHLY = average total monthly transaction DOLLAR VOLUME. "
                            "Use AVG_TRXN_AMT when the user says 'transaction amount', 'average amount', or 'dollar amount'. "
                            "Use AVG_TRXNS_WEEK when the user says 'transaction count', 'number of transactions', or 'frequency'. "
                            "Use TRXN_AMT_MONTHLY when the user says 'monthly amount' or 'monthly volume'."
                        ),
                    },
                },
                "required": ["segment", "threshold_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rule_2d_sweep",
            "description": (
                "2D grid sweep: vary two condition parameters simultaneously for an AML rule "
                "and produce a heatmap showing SAR catch rate and FP count at each combination. "
                "Use this when the user asks how two parameters interact, wants a grid or heatmap, "
                "or wants to optimize two thresholds at once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_factor": {
                        "type": "string",
                        "description": "Rule name (e.g. 'Activity Deviation (ACH)', 'Activity Deviation (Check)', 'Elder Abuse', 'Velocity Single', 'Detect Excessive').",
                    },
                    "sweep_param_1": {
                        "type": "string",
                        "description": (
                            "First parameter to sweep. "
                            "Activity Deviation (ACH): floor_amount or z_threshold. "
                            "Activity Deviation (Check): floor_amount or z_threshold. "
                            "Elder Abuse: floor_amount, z_threshold, or age_threshold. "
                            "Velocity Single: pair_total or ratio_tolerance. "
                            "Detect Excessive: floor_amount or time_window. "
                            "Omit to use rule default."
                        ),
                    },
                    "sweep_param_2": {
                        "type": "string",
                        "description": "Second parameter to sweep (must differ from sweep_param_1). Omit to use rule default.",
                    },
                    "cluster": {
                        "type": "integer",
                        "description": (
                            "Optional behavioral cluster number (1–4) from dynamic segmentation. "
                            "When specified, the sweep runs only on customers in that cluster. "
                            "Cluster resolves against the segment of the most recent clustering "
                            "(Business or Individual); if no clustering ran this session, falls "
                            "back to all-customer clustering and the tool result will include a "
                            "note stating the segment used — include that note in your response. "
                            "Use this when the user asks about a specific segment cluster "
                            "(e.g. 'show Elder Abuse sweep for Cluster 4')."
                        ),
                    },
                },
                "required": ["risk_factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_rules",
            "description": (
                "List all available AML detection rules with their SAR count, "
                "false positive count, and precision. Use this when the user asks which rules "
                "exist, which rules generate the most FPs, a rule performance overview, "
                "or when no specific rule name is given. "
                "Also call this whenever the user references a rule by number (e.g. 'Rule 7', "
                "'metrics for Rule 12'); the result includes 'Rule N:' prefixes so you can "
                "resolve the number to the rule name and answer from that data."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rule_sar_backtest",
            "description": (
                "RULE-LEVEL SAR backtest. Sweep a rule condition parameter for ONE "
                "specific AML rule. Use ONLY when the user names a specific rule "
                "(Elder Abuse, Velocity Single, Activity Deviation ACH, CTR Client, "
                "Detect Excessive, Mule, Crypto, Funnel, etc.). Examples: "
                "'SAR backtest for Elder Abuse', 'rule SAR catch rate for Crypto', "
                "'how does Velocity Single perform on SARs'. "
                "Do NOT use this for SEGMENT-LEVEL queries (Business or Individual "
                "customers without a named rule) — use sar_backtest for those. "
                "If the user references a rule by number (e.g. 'Rule 7'), call "
                "list_rules first to resolve the rule name, then call this tool "
                "with the resolved name as risk_factor."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_factor": {
                        "type": "string",
                        "description": (
                            "Risk factor / rule name to analyze "
                            "(e.g. 'Activity Deviation', 'Elder Abuse', 'Velocity Single', "
                            "'Detect Excessive'). Use list_rules to see all available rules."
                        ),
                    },
                    "sweep_param": {
                        "type": "string",
                        "description": (
                            "OPTIONAL. Omit by default — the tool picks the rule's "
                            "primary sweep parameter (typically floor_amount). Only set "
                            "this when the user explicitly names a parameter to sweep "
                            "(e.g. 'sweep z_threshold', 'try age_threshold')."
                        ),
                    },
                    "cluster": {
                        "type": "integer",
                        "description": (
                            "Optional behavioral cluster number (1–4) from dynamic segmentation. "
                            "When specified, the SAR backtest runs only on customers in that cluster. "
                            "Cluster resolves against the segment of the most recent clustering "
                            "(Business or Individual); if no clustering ran this session, falls "
                            "back to all-customer clustering and the tool result will include a "
                            "note stating the segment used — include that note in your response. "
                            "Use this when the user asks about a specific segment cluster "
                            "(e.g. 'show Elder Abuse SAR backtest for Cluster 2')."
                        ),
                    },
                },
                "required": ["risk_factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cluster_rule_summary",
            "description": (
                "Return SAR/FP/precision for ALL AML rules filtered to customers in a specific "
                "behavioral cluster. Use this when the user asks about rule performance across "
                "all rules for a specific cluster (e.g. 'show all rule results for Cluster 4', "
                "'which rules perform best in Cluster 2', 'SAR performance across all rules for "
                "that segment'). Do NOT use this for a single named rule — use rule_sar_backtest instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster": {
                        "type": "integer",
                        "description": (
                            "Behavioral cluster number (1–4) to filter all rules to. "
                            "Resolves against the segment of the most recent clustering "
                            "(Business or Individual); if no clustering ran this session, "
                            "falls back to all-customer clustering and the tool result will "
                            "include a note stating the segment used — include that note in "
                            "your response."
                        ),
                    },
                },
                "required": ["cluster"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cluster_threshold_analysis",
            "description": (
                "Run K-Means behavioral segmentation on a customer segment, then compute per-cluster "
                "adaptive thresholds that reduce false positives while maintaining SAR catch rate. "
                "Returns a comparison of uniform vs. cluster-adaptive alert thresholds and a bar chart "
                "showing false positive counts per cluster under each approach. "
                "Use this when the user asks about adaptive thresholds, per-cluster thresholds, "
                "how behavioral segmentation improves alert sensitivity, cluster-specific threshold "
                "recommendations, or reducing FPs by segment cluster."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "enum": ["Business", "Individual"],
                        "description": "Customer segment to analyze.",
                    },
                    "threshold_column": {
                        "type": "string",
                        "enum": ["AVG_TRXNS_WEEK", "AVG_TRXN_AMT", "TRXN_AMT_MONTHLY"],
                        "description": (
                            "Column to use as the alert threshold dimension. "
                            "If not specified, defaults to AVG_TRXNS_WEEK. "
                            "AVG_TRXN_AMT = average dollar amount per transaction. "
                            "TRXN_AMT_MONTHLY = average total monthly transaction volume. "
                            "AVG_TRXNS_WEEK = average number of transactions per week."
                        ),
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of behavioral clusters (2–6). Default 4.",
                    },
                    "target_sar_rate": {
                        "type": "number",
                        "description": (
                            "Minimum SAR catch rate to maintain at each cluster threshold (0–1). "
                            "Default 0.90 (90%). Lower values allow more aggressive FP reduction."
                        ),
                    },
                },
                "required": ["segment"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "Look at the given data and respond strictly to the data and no more. "
    "When the user references a rule by shorthand or partial name (e.g. "
    "'Crypto', 'ACH', 'Mule', 'Funnel') and the exact name is not in your "
    "rule index, resolve it to the closest matching canonical rule name "
    "and call the requested tool with that name as the risk_factor argument. "
    "Do not stop after identifying the match — actually run the tool the "
    "user asked for."
)


_INVALID_PARAMS = {
    "threshold_min", "threshold_max", "threshold_step",
    "step", "min_threshold",
}

# Queries containing these terms are ranking/sorting/counting questions answerable directly
# from an injected rule list — route to _run_with_rule_list (no tool call).
# Everything else (SAR backtest, 2D sweep, threshold tuning, etc.) must call a live tool,
# so the rule list context is dropped and the regular agentic loop runs instead.
#
# Additions 2026-06-08 (challenge / elliptical-follow-up recognition):
#   what about / does it / doesn't it / shouldn't / actually
# Catches user pushback patterns like 'what about Elder Abuse does it not have
# high SARs' where the user is challenging a prior ranking. Lets the model
# re-read the injected rule list and revise its answer instead of falling
# through to a tool call that doesn't address the challenge.
_RANKING_QUERY_TERMS = frozenset({
    "top", "bottom", "most", "least", "best", "worst",
    "highest", "lowest", "fewest", "how many", "how about",
    "which has", "which rules", "which is", "list all",
    "the same", "same by",
    "what about", "does it", "doesn't it", "doesnt it",
    "shouldn't", "shouldnt", "actually",
})

_REJECTION_MSG = (
    "threshold_min, threshold_max, threshold_step, step, and min_threshold are NOT valid "
    "parameters. The only valid parameters are segment (Business or Individual) and "
    "threshold_column (AVG_TRXNS_WEEK, AVG_TRXN_AMT, or TRXN_AMT_MONTHLY). "
    "Please specify one of those instead."
)

class ThresholdAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="threshold",
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
        )

    def _stream_llm(self, **kwargs):
        from config import BACKEND_THINK
        if BACKEND_THINK:
            kwargs.setdefault("extra_body", {})["think"] = True
            if kwargs.get("max_tokens", 0) <= 2048:
                kwargs["max_tokens"] = 4096
        return super()._stream_llm(**kwargs)

    def _run_with_rule_list(self, query: str, rule_list: str, history: list) -> tuple:
        """Answer a ranking/sorting query using rule list data embedded in the user message.

        Sends: history → user("Based on the following data: [rules]\n\n{query}")
        No system prompt — Rule 1 ("ALWAYS call a tool") suppresses context reading.
        """
        from .base_agent import _strip_thinking, MAX_TOKENS_TOOL
        from config import BACKEND_THINK
        user_content = (
            f"Based on the following AML rule data, answer the question below. "
            f"Do not call any tools — use only the data provided.\n\n"
            f"{rule_list}\n\n"
            f"Question: {query}"
        )
        messages = list(history) + [{"role": "user", "content": user_content}]
        # Call super() directly to bypass the tool-loop max_tokens bump (4096).
        # Ranking queries are short; 1500 tokens is sufficient and keeps latency low.
        extra = {"think": True} if BACKEND_THINK else {"chat_template_kwargs": {"enable_thinking": False}}
        msg = super()._stream_llm(
            model=self.model,
            max_tokens=MAX_TOKENS_TOOL,
            temperature=0,
            messages=messages,
            extra_body=extra,
        )
        return _strip_thinking(msg.content or ""), []

    def run(self, query: str, tool_executor, policy_context: str = "", history: list = None) -> tuple:
        query_lower = query.lower().replace("-", "_").replace(" ", "_")
        if any(p in query_lower for p in _INVALID_PARAMS):
            return _REJECTION_MSG, []
        if policy_context and "=== RULE LIST ===" in policy_context:
            # Only use _run_with_rule_list for ranking/sorting/counting queries.
            # Tool-calling queries (SAR backtest, 2D sweep, threshold tuning, etc.) must go
            # through the agentic loop WITHOUT the rule list context — otherwise the model
            # reads rule metrics from the injected data and skips the tool call entirely.
            if any(kw in query.lower() for kw in _RANKING_QUERY_TERMS):
                return self._run_with_rule_list(query, policy_context, history or [])
            return super().run(query, tool_executor, "", history)
        return super().run(query, tool_executor, policy_context, history)
