"""Threshold Tuning Agent — FP/FN trade-off analysis across segments and columns."""

from .base_agent import BaseAgent

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
                "Backtest a threshold column against simulated SAR (Suspicious Activity Report) data. "
                "Shows how many of the simulated SAR customers would have been caught vs. missed "
                "at each threshold level. Use this when the user asks about SAR catch rate, "
                "SAR detection rate, how many SARs a threshold would catch, or backtest analysis."
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
                "or when no specific rule name is given."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rule_sar_backtest",
            "description": (
                "For a specific named AML rule, sweep a rule condition parameter and show "
                "how many SAR customers are caught vs. missed at each threshold level. "
                "Use this when the user names a specific rule (e.g. 'Elder Abuse', "
                "'Velocity Single', 'Activity Deviation ACH', 'CTR Client', 'Detect Excessive') "
                "and asks about its SAR filing rate, SAR catch rate, SAR detection rate, "
                "SAR backtest, or rule-level FP/FN performance."
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
                            "Which condition parameter to sweep. "
                            "Activity Deviation (ACH): floor_amount or z_threshold. "
                            "Activity Deviation (Check): floor_amount or z_threshold. "
                            "Elder Abuse: floor_amount, z_threshold, or age_threshold. "
                            "Velocity Single: pair_total. "
                            "Detect Excessive: floor_amount. "
                            "Omit to use each rule's default."
                        ),
                    },
                    "cluster": {
                        "type": "integer",
                        "description": (
                            "Optional behavioral cluster number (1–4) from dynamic segmentation. "
                            "When specified, the SAR backtest runs only on customers in that cluster. "
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
                        "description": "Behavioral cluster number (1–4) to filter all rules to.",
                    },
                },
                "required": ["cluster"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are ARIA — Agentic Risk Intelligence for AML. You analyze false positive (FP) and \
false negative (FN) trade-offs as AML alert thresholds change, and run SAR backtests and \
2D sweeps for AML rule performance. \
IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese or other non-English characters.

DEFINITIONS (always apply these exactly — do not contradict them):
- TP (True Positive): SAR customer who IS alerted — correctly caught suspicious activity.
- FP (False Positive): Non-SAR customer who IS alerted — unnecessary investigation. HIGHER threshold → FEWER FPs.
- FN (False Negative): SAR customer who is NOT alerted — missed suspicious activity. HIGHER threshold → MORE FNs.
- TN (True Negative): Non-SAR customer who is NOT alerted — correctly silent.
- TP rate: TP / (TP + FN) — share of SAR customers caught. Also called recall or sensitivity.
- Precision: TP / (TP + FP) — share of alerts that are genuine SARs.
- Crossover: the threshold where FP and FN counts are closest — the optimal operating point.
- Raising the threshold reduces investigator workload (fewer alerts, fewer FPs) but increases FNs (missed SARs).
- Lowering the threshold catches more SARs (fewer FNs) but generates more FPs.

RULES — follow these exactly:
1. ALWAYS call a tool. Never answer threshold or alert questions from memory. EXCEPTION: if the user provides invalid parameters (threshold_min, threshold_max, threshold_step, step, min_threshold) or an invalid threshold_column, do NOT call any tool — follow Rule 14 instead.
2. For any question about FP, FN, threshold, alert rates, or transactions — call threshold_tuning.
3. For general segment counts or totals — call segment_stats.
4. For general SAR catch rate or SAR filing rate by SEGMENT (Business, Individual) with no specific rule named — call sar_backtest. If the user names a specific rule (e.g. "Elder Abuse", "Velocity Single", "CTR Client") — use rule_sar_backtest instead (see Rule 15).
5. threshold_column must be exactly one of: AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY
6. segment must be exactly one of: Business, Individual
7. If the user does not specify a segment, default to Business.
8. If the user does not specify a threshold column, default to AVG_TRXNS_WEEK.
9. After receiving tool results, the tool result contains a PRE-COMPUTED section. You MUST copy that section word-for-word into your response. Do NOT change any numbers, thresholds, or directional statements. After copying it, add ONE sentence of AML domain insight.
10. Do NOT paraphrase, round, or restate the numbers differently.
11. Do NOT include JSON or code blocks in your final reply.
12. Call the tool ONCE only. After receiving the tool result, write your final response immediately.
13. Do NOT compute, derive, or extrapolate any numbers not explicitly stated in the PRE-COMPUTED section. No rates, averages, differences, or trends. If a number is not in the PRE-COMPUTED section, do not mention it.
14. If the user provides invalid parameters such as threshold_min, threshold_max, threshold_step, step, or min_threshold, OR requests a threshold_column that is not one of AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY (e.g. daily balance, balance, net income, credit score, income, equity) — do NOT call the tool. State that the column is not available and list the three valid threshold_column options (AVG_TRXNS_WEEK, AVG_TRXN_AMT, TRXN_AMT_MONTHLY). Ask the user to specify one of these instead.
15. For any question about a specific AML rule's SAR performance, rule-level FP/FN analysis, or what happens to FP/FN if a rule condition parameter changes — call rule_sar_backtest with risk_factor (e.g. "Activity Deviation (ACH)", "Activity Deviation (Check)", "Elder Abuse", "Velocity Single", "Detect Excessive") and optionally sweep_param (floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window). If the user has not specified a rule, call list_rules first.
16. For any question about which rules exist, which rules generate the most FPs, or a rule performance overview — call list_rules.
17. For any question about how TWO condition parameters interact, a 2D analysis, optimizing two thresholds simultaneously, or a grid/heatmap of FP vs SAR — call rule_2d_sweep with risk_factor and optionally sweep_param_1 and sweep_param_2.
18. Do NOT describe UI interactions, chart features, or actions the user can take in the interface (e.g. "hover to see", "right-click to select", "click the cell"). The PRE-COMPUTED section already says the heatmap is in the chart. Say nothing else about the chart.
19. When the user asks about a specific behavioral cluster (e.g. "Cluster 3", "cluster 4"), pass the cluster number as an integer to the cluster parameter of rule_sar_backtest or rule_2d_sweep. Do NOT pass cluster to threshold_tuning, sar_backtest, or segment_stats — those tools do not accept a cluster parameter.
20. ONE insight sentence only. Do NOT add a second sentence or parenthetical. Do NOT describe heatmap positions (e.g. "top-left", "highest density"). Do NOT say "zero false positives" or "zero FNs" if the PRE-COMPUTED shows FP > 0 or FN > 0.
21. If the user asks about "highest FP rate" or "worst precision" — they mean precision=0.0%, NOT the highest raw FP count. Rules with SAR=0 and precision=0.0% have the highest FP rate. Name those rules specifically.
22. The system contains exactly 16 AML rules. Never state a different count.
23. After calling list_rules, if the user asked about a rule by a name that does not appear in the list (e.g. "layering", "smurfing") — state that no rule by that name exists and list the 11 available rules. Do NOT guess which rule "covers" the concept.
24. For any question about how ALL rules perform for a specific behavioral cluster — call cluster_rule_summary with the cluster number. Do NOT call list_rules or loop over rule_sar_backtest for this.
25. If a previous tool call returned an error about an invalid sweep parameter (e.g. "Unknown sweep_param_1" or "Unknown sweep_param_2"), and you asked the user to choose a valid parameter, and the user's reply is a parameter name (e.g. floor_amount, z_threshold, age_threshold, pair_total, ratio_tolerance, time_window, min_transactions, days_required, daily_floor) — do NOT treat it as a new query. Resume the previous rule_2d_sweep or rule_sar_backtest call with the same risk_factor, keeping all valid parameters unchanged and replacing only the invalid one with the user's corrected choice.
26. For pure definitional questions about TP, FP, FN, TN, precision, recall, crossover, the effect of raising or lowering thresholds on FP/FN counts, or what a 2D grid/sweep shows — answer DIRECTLY from the DEFINITIONS section above. Do NOT call any tool. Answer in 2–3 sentences using only the definitions listed above.\
"""


_INVALID_PARAMS = {
    "threshold_min", "threshold_max", "threshold_step",
    "step", "min_threshold",
}

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

    def run(self, query: str, tool_executor, policy_context: str = "") -> tuple:
        query_lower = query.lower().replace("-", "_").replace(" ", "_")
        if any(p in query_lower for p in _INVALID_PARAMS):
            return _REJECTION_MSG, []
        return super().run(query, tool_executor, policy_context)
