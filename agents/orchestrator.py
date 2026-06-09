"""
Orchestrator Agent — routes user queries to specialist agents and runs them in parallel.

Routing is done via LLM classification (single fast API call):
  threshold    → ThresholdAgent    (FP/FN tuning, alert stats)
  segmentation → SegmentationAgent (clustering, dynamic segmentation, alerts distribution)
  policy       → PolicyAgent       (AML policy, regulatory questions)
  greeting     → friendly greeting response (no agent run)
  policy       → default for anything unclassified (base model handles gracefully)
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
print("[orchestrator] MODULE LOADED", flush=True)

from .base_agent import OLLAMA_BASE_URL, OLLAMA_MODEL
from .threshold_agent import ThresholdAgent
from .segmentation_agent import SegmentationAgent
from .policy_agent import PolicyAgent

_CONFIRMATION_PHRASES = {
    "are you sure", "are you certain", "are you confident",
    "really", "really?", "is that right", "is that correct",
    "double check", "double-check", "can you verify", "can you confirm",
    "are you sure about that", "are you sure about this",
    "you sure", "you sure?",
}

# Cluster-filter directive — appended to segmentation user query when the user is asking
# the app to display only specific cluster(s). Moved out of the segmentation system prompt
# (Rule 10) per feedback_instruction_placement — data-turn cues beat system rules.
DISPLAY_CLUSTERS_DIRECTIVE = (
    "\n\nOn the very last line of your response, write exactly:\n"
    "DISPLAY_CLUSTERS: N\n"
    "where N is a comma-separated list of cluster numbers (e.g. DISPLAY_CLUSTERS: 4 "
    "or DISPLAY_CLUSTERS: 1,3). Do NOT include any other text on that line."
)

_CLUSTER_FILTER_PATTERNS = re.compile(
    r"\b(show only|highest[- ]risk|lowest[- ]risk|low activity|high activity|"
    r"top \d+|bottom \d+|filter (?:to|for) cluster)\b",
    re.IGNORECASE,
)

def _is_elliptical(query: str) -> bool:
    """True if query is a short/elliptical continuation that lacks standalone routing signal."""
    q = query.lower().strip()
    words = q.split()
    if q.startswith(("and ", "what about ", "how about ", "which ", "top ", "bottom ")):
        return True
    if len(words) <= 7 and any(kw in q for kw in [
        "highest", "lowest", "most", "least", "best", "worst",
        "youngest", "oldest", "largest", "smallest", "fewest",
    ]):
        return True
    # Confirmation/skepticism phrases always inherit context from prior turn
    if q.rstrip("?") in {p.rstrip("?") for p in _CONFIRMATION_PHRASES}:
        return True
    return False


_CLASSIFY_SYSTEM = """\
You are a routing classifier for ARIA. Given a user query, respond with one or more of these labels (comma-separated, no other text):
  threshold    — user wants to RUN analysis on OUR LOCAL DATA: FP/FN trade-offs, SAR catch rates, rules, rule performance, transaction stats, or rule-level sweeps
  segmentation — user wants to RUN K-Means clustering or alert distribution
  ofac         — user wants to RUN OFAC sanctions screening
  greeting     — query is a greeting or social pleasantry
  out_of_scope — query is not related to any of the above
  policy       — user is asking a GENERAL KNOWLEDGE question about ARIA, AML, regulations, definitions, or concepts
Output ONLY the label(s), comma-separated. No explanation.\
"""

# Legacy few-shot prompt kept for diagnostic / rollback comparison only.
_CLASSIFY_SYSTEM_LEGACY = """\
You are a routing classifier for ARIA. Given a user query, respond with \
one or more of these labels (comma-separated, no other text):

  threshold    — user wants to RUN threshold tuning analysis on OUR LOCAL DATA (FP/FN trade-off charts, sweep analysis)
  segmentation — user wants to RUN clustering/segmentation on OUR LOCAL DATA (K-Means, treemap, behavioral groups)
  ofac         — user wants to RUN OFAC sanctions screening on OUR LOCAL CUSTOMER DATA (SDN list hits, sanctioned country exposure)
  policy       — user is asking a GENERAL KNOWLEDGE question about AML, compliance, regulations, industry practices, or best practices — does NOT require running local data analysis
  greeting     — query is a greeting or social pleasantry (hello, hi, how are you, etc.)
  out_of_scope — query is not related to any of the above AML topics

Key distinction:
- "Show FP/FN tuning for Business customers" → threshold  (run local analysis)
- "Show FP/FN threshold tuning for Individual customers" → threshold
- "Run SAR backtest for Individual customers" → threshold  (SAR backtest is a threshold tool)
- "What threshold catches 90% of SARs?" → threshold
- "SAR catch rate for Business monthly transaction amount" → threshold
- "Run SAR backtest" → threshold
- "Show me a 2D grid for Activity Deviation ACH" → threshold  (2D sweep is a threshold tool)
- "How do floor amount and sigma interact for Activity Deviation?" → threshold
- "Show me the ACH deviation rule performance" → threshold
- "What is the SAR catch rate for Activity Deviation Check?" → threshold
- "Show the heatmap for Elder Abuse" → threshold
- "2D analysis for Velocity Single" → threshold
- "How does time window interact with floor amount for Detect Excessive?" → threshold
- "Show me the AML rule performance overview" → threshold  (list_rules is a threshold tool)
- "Which rules generate the most false positives?" → threshold
- "What is the SAR catch rate for the Activity Deviation rule?" → threshold
- "Show rule-level FP analysis" → threshold
- "What happens to FP if I raise the age threshold for Elder Abuse?" → threshold
- "How do banks manage alert volumes?" → policy  (general knowledge question)
- "What is AML?" → policy  (general knowledge question)
- "What is threshold tuning?" → policy  (conceptual overview of the practice — NOT a request to run analysis on local data)
- "Explain threshold tuning" → policy  (explanation request — NOT running local data)
- "How does threshold tuning work?" → policy
- "Can you explain what threshold tuning means?" → policy
- "What is dynamic segmentation?" → policy  (conceptual question — NOT a request to run clustering)
- "Explain dynamic segmentation" → policy
- "How does behavioral segmentation work?" → policy
- "What is K-Means clustering?" → policy
- "What is customer segmentation?" → policy
- "Cluster all customers" → segmentation  (run local analysis)
- "What does AML policy say about structuring?" → policy  (general knowledge + knowledge base)
- "Show alerts and false positive distribution across segments" → segmentation  (distribution chart, NOT threshold tuning)
- "Show alert distribution" → segmentation
- "How are alerts spread across segments?" → segmentation
- "Which segment has the most alerts?" → segmentation
- "What is the average transaction amount for Business customers?" → threshold  (segment_stats tool)
- "How many alerts does the Individual segment have?" → threshold  (segment_stats tool)
- "What are the transaction stats for Business customers?" → threshold
- "Show me Business customer stats" → threshold
- "Show me all AML rules" → threshold  (list_rules is a threshold tool — NOT policy)
- "What rules are in the system?" → threshold
- "List all the AML rules" → threshold
- "What transactions are flagged by the layering rule?" → threshold  (list_rules — 'layering' is not a KB topic)
- "Which rule covers layering?" → threshold  (list_rules)
- "Show rule sweep for xyz_column" → threshold  (rule sweep request, even with unknown param — NOT policy)
- "Show rule sweep for an invalid parameter" → threshold
- "What is the SAR filing rate for Individual?" → threshold  (sar_backtest is a threshold tool)
- "SAR filing rate for Business" → threshold
- "Which rule has the highest FP rate?" → threshold  (list_rules)
- "Which rules generate only false positives?" → threshold
- "Run a SAR backtest for the structuring rule" → threshold  (rule_sar_backtest — NOT policy)
- "SAR backtest for Elder Abuse" → threshold
- "Show Elder Abuse sweep for Cluster 4" → threshold  (cluster-filtered rule sweep)
- "Run SAR backtest for Activity Deviation ACH in Cluster 2" → threshold
- "Show 2D heatmap for Elder Abuse for Cluster 3" → threshold
- "Which cluster has the most false positives for Velocity Single?" → threshold
- "Which cluster of Business customers has the highest transaction volume?" → segmentation
- "Which Business cluster has the most activity?" → segmentation
- "Which cluster has the most transaction activity?" → segmentation
- "Show Business customer clusters by transaction behavior" → segmentation
- "Run OFAC screening" → ofac
- "Show OFAC sanctions exposure" → ofac
- "Which customers are on the sanctions list?" → ofac
- "How many customers are from sanctioned countries?" → ofac
- "Show me OFAC hits" → ofac
- "Screen customers against SDN list" → ofac
- "What is our Iran/North Korea customer exposure?" → ofac
- "Show comprehensive sanctions hits" → ofac
- "Show me a 2D grid for Elder Abuse" → threshold  (2D grid = 2D sweep, same tool)
- "Show 2D analysis for Detect Excessive Transaction Activity" → threshold  (2D analysis = 2D sweep)
- "Run a 2D grid analysis for Velocity Single" → threshold
- "Show grid analysis for Activity Deviation ACH" → threshold
- "What are Canada's suspicious transaction reporting requirements?" → policy
- "What are Canada's AML rules?" → policy
- "What does FINTRAC require?" → policy
- "What is AML structuring?" → policy  (prefix 'AML' does not change the topic — still a policy question)
- "What is tructuring?" → policy  (typo for 'structuring' — still an AML definition question)
- "What is smurfing?" → policy  (synonym for structuring — AML definition question)
- "What is AML layering?" → policy
- "What is AML typology?" → policy
- "cluster into 3 groups" → segmentation  (user specifying cluster count is still a segmentation request)
- "I only want 2 business clusters" → segmentation
- "show me 4 clusters for Individual customers" → segmentation
- "I want k-means with 3 clusters" → segmentation
- "What are the EU requirements for beneficial ownership registers?" → policy  (EU regulatory question)
- "What does the 4th AMLD require for customer due diligence?" → policy
- "What does the 5th AMLD say about virtual assets?" → policy
- "What are FATF recommendations for banks?" → policy
- "What does UN Security Council Resolution 1373 require of banks?" → policy
- "What are EBA guidelines on ML/TF risk factors?" → policy
- "What are the beneficial ownership disclosure requirements?" → policy
- "What does the EU AML Regulation require?" → policy
- "What is the AMLA?" → policy
- "Does UNODC have guidance on AML?" → policy
- "What are PEP requirements under AML regulations?" → policy
- "Thanks, that was helpful!" → greeting
- "Thanks, that's great" → greeting
- "Got it, thanks" → greeting
- "Thank you" → greeting
- "That was useful, thanks" → greeting
- "Can you send this to my compliance team?" → out_of_scope  (action request, not an AML analysis task)
- "Can you email this to someone?" → out_of_scope
- "Can you export this as a PDF?" → out_of_scope
- "What is a false positive?" → policy  (definitional question — base model gives better educational answer)
- "What is a false negative?" → policy
- "What is the difference between FP and FN?" → policy
- "Explain false positives in AML monitoring" → policy
- "What does FP mean?" → policy
- "Can you explain false positives and false negatives?" → policy
- "What is a 2D grid?" → policy  (definitional/conceptual question)
- "What is a 2D sweep?" → policy
- "How does a 2D grid work?" → policy
- "Are you ARIA?" → greeting  (identity question — not an AML topic)
- "What is your name?" → greeting
- "Who are you?" → greeting
- "Ahoy!" → greeting
- "Ahoy matey!" → greeting
- "What are true positives in AML monitoring?" → policy  (definitional question — base model gives better educational answer)
- "What are true negatives?" → policy
- "What is the difference between TP and TN?" → policy
- "What is OFAC?" → policy  (definition question — NOT a screening request)
- "What does OFAC stand for?" → policy
- "My dog OFAC met a cat the other day" → out_of_scope  (OFAC here is a name, not AML topic)
- "OFAC said hello" → out_of_scope
- "Is OFAC the same as sanctions screening?" → policy  (terminology question — NOT a screening request)
- "What does OFAC stand for?" → policy  (terminology question)
- "What is OFAC?" → policy
- "What are the rules that have z_threshold as a parameter?" → threshold  (list_rules — filter by parameter)
- "Which rule shows the highest SAR count?" → threshold  (list_rules tool)

Rules:
- Output ONLY the label(s), comma-separated. No explanation, no punctuation other than commas.
- A query can map to multiple labels (e.g. threshold,segmentation).
- When in doubt between out_of_scope and an AML label, prefer the AML label.\
"""


class OrchestratorAgent:

    _GREETING = (
        "Hello! I'm ARIA — Agentic Risk Intelligence for AML. I can help you with:\n"
        "- **Threshold tuning** — optimize your alert investigation budget by analyzing FP/FN trade-offs, SAR catch rates, and rule sweep performance across threshold parameters\n"
        "- **Customer segmentation** — identify behavioral risk clusters using K-Means across transaction velocity, volume, and account characteristics\n"
        "- **AML policy Q&A** — answer questions on BSA/AML regulations, FFIEC examination guidance, FinCEN advisories, and Wolfsberg Group best practices\n\n"
        "Try one of the suggested prompts on the left, or ask me a question."
    )

    _CAPABILITY = (
        "I'm ARIA — Agentic Risk Intelligence for AML. Here's what I do:\n\n"
        "**1. Threshold Tuning**\n"
        "I analyze FP/FN trade-offs as alert thresholds are swept across Business and Individual customer segments. "
        "For each threshold column — average transaction amount, monthly transaction volume, or weekly transaction count — "
        "I show you exactly how many SARs you catch and how many false positives you generate at every threshold level. "
        "This lets your compliance team find the optimal cut-point: the threshold that maximizes SAR detection while "
        "minimizing the investigator workload from low-value alerts.\n\n"
        "**2. Customer Behavioral Segmentation**\n"
        "I apply K-Means clustering to your customer base to identify natural behavioral risk groups based on "
        "transaction velocity, volume, average amounts, account age, and account type. Each cluster gets a risk "
        "profile so your team can apply different monitoring intensities to different customer groups instead of "
        "treating all customers identically.\n\n"
        "**3. AML Rule Analysis**\n"
        "I run SAR backtests and 2D parameter sweeps across your active monitoring rules. A SAR backtest shows "
        "how the rule's SAR catch rate changes as its threshold is adjusted. A 2D sweep maps two parameters "
        "simultaneously so you can see the full trade-off surface and identify the setting that best balances "
        "detection against false positives.\n\n"
        "**4. AML Policy Q&A**\n"
        "I answer regulatory and compliance questions on BSA/AML, FinCEN guidance, OFAC/sanctions concepts, "
        "Wolfsberg Principles, FATF recommendations, and general AML typologies — drawn from my training knowledge. "
        "You can also upload your own org-specific documents and ask questions about them.\n\n"
        "Ask me a question or try one of the suggested prompts on the left."
    )

    _OUT_OF_SCOPE = (
        "I can only help with AML-specific topics:\n"
        "- **Threshold tuning** — FP/FN trade-off analysis, SAR catch rates, rule sweep optimization\n"
        "- **Customer segmentation** — K-Means behavioral clustering, alert distribution by segment\n"
        "- **AML policy Q&A** — BSA/AML regulations, FFIEC guidance, FinCEN advisories\n\n"
        "Please rephrase your question around one of these areas."
    )

    def __init__(self):
        self.threshold_agent    = ThresholdAgent()
        self.segmentation_agent = SegmentationAgent()
        self.policy_agent       = PolicyAgent()
        self._agent_map = {
            "threshold":    self.threshold_agent,
            "segmentation": self.segmentation_agent,
            "policy":       self.policy_agent,
        }
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self._last_agent: str = ""  # tracks last threshold/segmentation turn for sticky routing

    def _route(self, query: str, last_assistant: str = "") -> list:
        """LLM-based routing — classify query into agent labels."""
        _exception = False
        try:
            # Multi-turn bootstrap: the classifier prompt lives as a prior user
            # turn that the model "acknowledges" with a hardcoded assistant
            # response. This mirrors the Ollama-shell pattern that empirically
            # classifies correctly across a much wider set of queries than
            # either system-role placement or single-turn user concatenation.
            # See feedback_instruction_placement.
            classify_messages = [
                {"role": "user",      "content": _CLASSIFY_SYSTEM},
                # Empty assistant ack outperformed a hardcoded sentence on 2026-06-05
                # smokes: same 1-API-call cost, +1 correct (resolved "list of all the
                # AML rules in the system" → threshold instead of policy). The empty
                # turn supplies structural scaffolding (a multi-turn shape the model
                # was trained against) without injecting topic-content that nudged
                # the model into conversational mode.
                {"role": "assistant", "content": ""},
            ]
            # NOTE: last_assistant is INTENTIONALLY NOT appended here. The classifier
            # must be stateless — it classifies only the current query against the
            # examples in _CLASSIFY_SYSTEM. Injecting a previous chat response (e.g.
            # a multi-paragraph cluster summary) shifts the model's output
            # distribution at temperature=0 and was producing empty classifier
            # output on long-context turns. Elliptical follow-ups are already
            # handled at the run() level via sticky routing on self._last_agent.
            classify_messages.append({"role": "user", "content": query})
            _t_start = time.perf_counter()
            _prompt_chars = sum(len(m.get("content") or "") for m in classify_messages)
            # ── INPUT DIAGNOSTICS ─────────────────────────────────────────
            # Print per-message char counts and a snippet of last_assistant
            # to track context-size sensitivity in classifier output.
            print(f"[debug.classify.input] query={repr(query[:200])}", flush=True)
            for _i, _m in enumerate(classify_messages):
                _content = _m.get("content") or ""
                _role = _m.get("role")
                if _i == 0:
                    print(f"[debug.classify.input] msg[{_i}] role={_role} len={len(_content)} (system prompt)", flush=True)
                elif _role == "assistant" and len(_content) > 100:
                    # last_assistant — show head + tail to spot patterns triggering drift
                    print(f"[debug.classify.input] msg[{_i}] role={_role} len={len(_content)}", flush=True)
                    print(f"[debug.classify.input]   head_500={repr(_content[:500])}", flush=True)
                    print(f"[debug.classify.input]   tail_500={repr(_content[-500:])}", flush=True)
                else:
                    print(f"[debug.classify.input] msg[{_i}] role={_role} len={len(_content)} content={repr(_content[:200])}", flush=True)
            resp = self._client.chat.completions.create(
                model=OLLAMA_MODEL,
                max_tokens=800,
                temperature=0,
                messages=classify_messages,
                extra_body={"think": False, "chat_template_kwargs": {"enable_thinking": False}},
            )
            _t_total = time.perf_counter() - _t_start
            print(
                f"[timing] orchestrator.classify: total={_t_total:.2f}s | "
                f"prompt_msgs={len(classify_messages)}/{_prompt_chars}ch tools=0",
                flush=True,
            )
            # ── OUTPUT DIAGNOSTICS ────────────────────────────────────────
            # Capture raw BEFORE think-tag stripping + finish_reason + token
            # usage so we can tell whether the model produced only thinking
            # content, hit max_tokens mid-thought, or actually returned empty.
            _choice = resp.choices[0]
            _msg = _choice.message
            _raw_full = (_msg.content or "")
            _finish = getattr(_choice, "finish_reason", "?")
            _usage = getattr(resp, "usage", None)
            _prompt_tokens     = getattr(_usage, "prompt_tokens",     None) if _usage else None
            _completion_tokens = getattr(_usage, "completion_tokens", None) if _usage else None
            _total_tokens      = getattr(_usage, "total_tokens",      None) if _usage else None
            print(
                f"[debug.classify.output] finish_reason={_finish} "
                f"prompt_toks={_prompt_tokens} completion_toks={_completion_tokens} "
                f"total_toks={_total_tokens}",
                flush=True,
            )
            print(f"[debug.classify.output] raw_pre_strip_len={len(_raw_full)}", flush=True)
            if _raw_full:
                # Show full pre-strip text — usually short for classifier, this is
                # the smoking gun if the model emitted only <think> content.
                print(f"[debug.classify.output] raw_pre_strip={repr(_raw_full[:2000])}", flush=True)
                if "<think>" in _raw_full:
                    print(f"[debug.classify.output] WARNING: response contains <think> tag — model is thinking despite enable_thinking=False", flush=True)
                if "<think>" in _raw_full and "</think>" not in _raw_full:
                    print(f"[debug.classify.output] WARNING: unclosed <think> — model likely hit max_tokens mid-thought", flush=True)
            raw = _raw_full
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            print(f"[orchestrator] classifier raw: {repr(raw)}", flush=True)
            valid = {"threshold", "segmentation", "ofac", "policy", "greeting", "out_of_scope"}
            labels = [l.strip().lower() for l in raw.split(",") if l.strip().lower() in valid]
            print(f"[orchestrator] classifier labels: {labels}", flush=True)
        except Exception as e:
            print(f"[orchestrator] classification error: {e} — defaulting to policy", flush=True)
            labels = ["policy"]
            _exception = True

        q_lower = query.lower()

        # Rescue: only fires when the LLM call threw an exception and defaulted to policy.
        # Does NOT override a legitimate policy classification from the model.
        _is_threshold_kw = any(w in q_lower for w in [
            "sweep", "fp", "fn", "sar", "heatmap", "backtest", "tuning", "threshold",
            "2d grid", "2d analysis", "grid analysis",
            "avg_trxns_week", "avg_trxn_amt", "trxn_amt_monthly",
            # list_rules-style queries — classifier prompt routes these to threshold
            # but a misclassification (policy/out_of_scope) used to slip through
            # without rescue, hitting the OOS handler instead.
            "rule", "rules", "aml rule", "list rules", "list_rules",
            "precision", "fp rate", "sar rate",
        ])
        _is_segmentation_kw = any(w in q_lower for w in ["cluster", "k-means", "kmeans", "treemap"])
        if _exception and labels == ["policy"] and _is_threshold_kw and not _is_segmentation_kw:
            labels = ["threshold"]
            print("[orchestrator] exception rescue → threshold", flush=True)

        # OOS rescue: classifier sometimes returns out_of_scope for data queries
        # that contain unmistakable threshold/segmentation keywords (list-rules
        # style "show me the AML rules" was bleeding through after the
        # production_switchover minimal-prompt rollout).
        if labels == ["out_of_scope"]:
            if _is_threshold_kw and not _is_segmentation_kw:
                labels = ["threshold"]
                print("[orchestrator] oos rescue → threshold", flush=True)
            elif _is_segmentation_kw and not _is_threshold_kw:
                labels = ["segmentation"]
                print("[orchestrator] oos rescue → segmentation", flush=True)

        # Segmentation rescue: empty-ack classifier (commit 5247fcc) is more
        # decisive and sometimes commits to single-label `segmentation` on
        # cluster-filtered threshold-tool queries — e.g. "run adaptive
        # thresholds on Individual customers" or "Show Elder Abuse SAR
        # backtest for Cluster 4". With single-label segmentation the Phase 5
        # multi-label suppression can't fire (it requires both labels), so the
        # segmentation agent runs alone and lacks the threshold tool. Same
        # keyword set as Phase 5 — promote to threshold when the cluster word
        # is a parameter, not the primary intent.
        if labels == ["segmentation"]:
            _q_lower_seg = query.lower()
            _is_threshold_tool_query = any(kw in _q_lower_seg for kw in [
                "backtest", "adaptive threshold", "threshold tuning",
                "sweep", "2d", "heatmap", "grid",
                "sar catch", "fp rate", "fn rate", "precision",
                # rule names that imply a threshold tool
                "activity deviation", "elder abuse", "velocity single", "velocity multiple",
                "funnel account", "structuring", "ctr client", "burst in",
                "risky international", "round-trip", "human trafficking", "detect excessive",
            ])
            if _is_threshold_tool_query:
                labels = ["threshold"]
                print("[orchestrator] segmentation rescue -> threshold (analytical tool query)", flush=True)

        # Empty classification → out_of_scope. Threshold/segmentation keyword rescue
        # still fires first because we'd rather route ambiguous data-y queries to the
        # right agent than refuse them. But if no signal at all, refuse explicitly
        # rather than risk a hallucinated policy answer.
        if not labels:
            if _is_threshold_kw:
                labels = ["threshold"]
            elif _is_segmentation_kw:
                labels = ["segmentation"]
            else:
                labels = ["out_of_scope"]
            print(f"[orchestrator] keyword fallback labels: {labels}", flush=True)

        print(f"[orchestrator] routing to: {labels}", flush=True)
        return labels

    def run(self, query: str, tool_executor, last_assistant: str = "", history: list = None, last_cluster_result: str = "", last_rule_list: str = "", last_threshold_params: dict = None) -> tuple:
        """
        Route query via LLM, run required agents (in parallel if >1), merge results.
        Returns: (combined_text, all_chart_results)
        """
        labels = self._route(query, last_assistant)

        # Multi-label suppression: classifier multi-labels [threshold, segmentation] on
        # queries like "adaptive thresholds on business customers" or "Elder Abuse
        # backtest for Cluster 4" because of cluster/segment words. But the operation
        # is a threshold-tool operation filtered by cluster — only the threshold agent
        # has the actual tool. Segmentation agent has no tool for this and fabricates
        # plausible-looking threshold numbers (Tests 11 and 13 in 2026-06-05 app run).
        # See cluster_segment_plan.md Phase 5.
        if "threshold" in labels and "segmentation" in labels:
            _q_lower_ml = query.lower()
            _is_threshold_tool_query = any(kw in _q_lower_ml for kw in [
                "backtest", "adaptive threshold", "threshold tuning",
                "sweep", "2d", "heatmap", "grid",
                "sar catch", "fp rate", "fn rate", "precision",
                # rule names that imply a threshold tool
                "activity deviation", "elder abuse", "velocity single", "velocity multiple",
                "funnel account", "structuring", "ctr client", "burst in",
                "risky international", "round-trip", "human trafficking", "detect excessive",
            ])
            if _is_threshold_tool_query:
                labels = ["threshold"]
                print("[orchestrator] multi-label suppression -> threshold (analytical tool query)", flush=True)

        # Sticky routing: elliptical follow-ups routed to policy/weak labels → re-use prior agent.
        # Covers "and the youngest", "which one has lowest", "top 3 by precision", "what about days_required?"
        _STICKY_SOURCES = {"threshold", "segmentation"}
        _WEAK_LABELS = {"policy", "out_of_scope", "greeting"}
        if (self._last_agent in _STICKY_SOURCES
                and set(labels) <= _WEAK_LABELS
                and _is_elliptical(query)):
            print(f"[orchestrator] sticky routing -> {self._last_agent} (elliptical follow-up)", flush=True)
            labels = [self._last_agent]

        # Extended sticky: cluster attribute follow-ups misrouted to threshold.
        # Queries like "which one has the highest monthly trxn amount" contain terms
        # ("monthly", "trxn", "amount") that look threshold-y to the LLM, but they
        # are cluster attribute lookups when a segmentation session is active.
        # Guard: no specific rule name or threshold tool keyword present.
        _q_lower = query.lower()
        _has_rule_or_tool = any(kw in _q_lower for kw in [
            "activity deviation", "elder abuse", "velocity single", "velocity multiple",
            "funnel account", "structuring", "ctr client", "burst in", "risky international",
            "round-trip", "human trafficking", "detect excessive",
            "backtest", "2d sweep", "2d grid", "floor_amount", "z_threshold",
            "pair_total", "days_required", "daily_floor", "threshold tuning",
            "list rules", "rule sweep", "sar backtest",
        ])
        if (self._last_agent == "segmentation"
                and last_cluster_result
                and _is_elliptical(query)
                and labels == ["threshold"]
                and not _has_rule_or_tool):
            labels = ["segmentation"]
            print("[orchestrator] sticky routing → segmentation (cluster attribute rescued from threshold)", flush=True)

        # Build prior session context once — passed to every agent (including policy)
        # so elliptical follow-ups ("and the youngest", "what about days_required?")
        # can be answered correctly even when misrouted.
        # Cluster context is safe to pass to any agent including policy —
        # reading ages/counts from a list is unambiguous.
        # Rule lists are NOT injected to policy — policy lacks sorting logic
        # and hallucates fake rule IDs when it tries to rank by precision.
        _prior_context = ""
        if last_cluster_result and "Cluster" in last_cluster_result:
            _cc = last_cluster_result
            if len(_cc) > 1500:
                lines = [l for l in _cc.splitlines() if l.strip().startswith("Cluster")]
                _cc = "\n".join(lines[:20]) if lines else _cc[:1500]
            _prior_context = f"[PREVIOUS CLUSTERING RESULT]\n{_cc}\n[END PREVIOUS RESULT]"

        if "greeting" in labels:
            return self._GREETING, []

        if "out_of_scope" in labels:
            return self._OUT_OF_SCOPE, []

        # OFAC screening is handled directly via tool_executor (no specialist agent)
        if "ofac" in labels:
            # Detect explicit name lookup: name must appear directly after a lookup verb.
            # Case-sensitive title-case detection prevents matching generic query phrases.
            import re as _re
            _has_lookup_verb = _re.search(
                r'\b(?:lookup|look up|check|find|search for|screen)\b', query, _re.IGNORECASE
            )
            _name_match = None
            if _has_lookup_verb:
                # Verb present — look for title-case name directly following it
                _name_match = _re.search(
                    r'\b(?:lookup|look up|check|find|search for|screen)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
                    query
                )
            # Also handle "is [Name] on the list?" form
            if not _name_match:
                _name_match = _re.search(
                    r'\bis\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+on\b', query
                )
            if _name_match:
                name = _name_match.group(1)
                text, fig = tool_executor("ofac_name_lookup", {"name": name})
            else:
                text, fig = tool_executor("ofac_screening", {})
                chart_results = [("ofac_screening", {}, fig)] if fig is not None else []
                return text, chart_results
            return text, []

        agent_labels = [l for l in labels if l in self._agent_map]

        if not agent_labels:
            return self.policy_agent.run(query, tool_executor, _prior_context, history)

        to_run = [(name, self._agent_map[name]) for name in agent_labels]

        if len(to_run) == 1:
            name, agent = to_run[0]
            if name in _STICKY_SOURCES:
                self._last_agent = name
            context = ""
            if last_rule_list and name == "threshold":
                _rule_ctx = last_rule_list[:4000] if len(last_rule_list) > 4000 else last_rule_list
                _is_rule_followup = (
                    _is_elliptical(query)
                    or any(kw in query.lower() for kw in {"the same", "same by", "sort by", "rank by"})
                )
                if last_assistant and _is_rule_followup:
                    _prev_header = last_assistant.split('\n')[0].strip()[:200]
                    context = f"[PRIOR RESPONSE]\n{_prev_header}\n[END PRIOR RESPONSE]\n\n{_rule_ctx}"
                else:
                    context = _rule_ctx
                print(f"[orchestrator] injecting rule list for threshold query ({len(_rule_ctx)} chars)")
            if name == "segmentation":
                # Only use actual cluster stats — never fall back to last_assistant because
                # non-clustering responses (e.g. Elder Abuse text) can contain "Cluster N"
                # and would be falsely injected as [PREVIOUS CLUSTERING RESULT].
                _cluster_ctx = last_cluster_result
                if _cluster_ctx and "Cluster" in _cluster_ctx:
                    # Trim to ~2500 chars to avoid context overflow in long conversations
                    if len(_cluster_ctx) > 2500:
                        lines = [l for l in _cluster_ctx.splitlines() if l.strip().startswith("Cluster")]
                        _cluster_ctx = "\n".join(lines[:20]) if lines else _cluster_ctx[:1500]
                    context = f"[PREVIOUS CLUSTERING RESULT]\n{_cluster_ctx}\n[END PREVIOUS RESULT]"
                    print(f"[orchestrator] injecting previous cluster context ({len(_cluster_ctx)} chars)", flush=True)
            # Threshold agent: strip raw history — rule list is already injected via
            # context above, and prior backtest/AT prose in history causes the model
            # to answer from memory instead of calling the tool (J-series failures).
            if name == "threshold" and last_threshold_params:
                seg = last_threshold_params.get("segment", "")
                col = last_threshold_params.get("threshold_column", "")
                if seg and col:
                    _tp_hint = f"[SESSION CONTEXT — last threshold call used segment={seg}, column={col}. Use only if the user has not specified a different segment or column.]"
                    context = f"{_tp_hint}\n\n{context}".strip() if context else _tp_hint
                    print(f"[orchestrator] injecting last threshold params: {seg}/{col}")
            # Segmentation: when the user is filtering clusters for the chart, append
            # the DISPLAY_CLUSTERS protocol directive to the query (moved from system
            # prompt Rule 10 per feedback_instruction_placement).
            _seg_query = query
            if name == "segmentation" and _CLUSTER_FILTER_PATTERNS.search(query):
                _seg_query = query + DISPLAY_CLUSTERS_DIRECTIVE
                print("[orchestrator] appended DISPLAY_CLUSTERS directive to segmentation query")
            _th_history = [] if name == "threshold" else history
            return agent.run(_seg_query, tool_executor, context, _th_history)

        results = {}
        with ThreadPoolExecutor(max_workers=len(to_run)) as executor:
            futures = {
                executor.submit(agent.run, query, tool_executor, "", history): name
                for name, agent in to_run
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    print(f"[orchestrator] agent '{name}' error: {e}")
                    results[name] = ("Something went wrong — please try again.", [])

        all_charts = []
        text_parts = []
        for name in agent_labels:
            if name in results:
                text, charts = results[name]
                text_parts.append(f"**{name.capitalize()} Analysis:**\n{text}")
                all_charts.extend(charts)

        return "\n\n".join(text_parts), all_charts
