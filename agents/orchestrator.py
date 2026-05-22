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
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
print("[orchestrator] MODULE LOADED — v2 with conceptual path", flush=True)

from .base_agent import OLLAMA_BASE_URL, OLLAMA_MODEL
from .threshold_agent import ThresholdAgent
from .segmentation_agent import SegmentationAgent
from .policy_agent import PolicyAgent

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
    return False


_CLASSIFY_SYSTEM = """\
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
- "What is a false positive?" → threshold  (FP/FN definition is a threshold concept, not policy)
- "What is a false negative?" → threshold
- "What is the difference between FP and FN?" → threshold
- "Explain false positives in AML monitoring" → threshold
- "What does FP mean?" → threshold
- "Can you explain false positives and false negatives?" → threshold
- "What is a 2D grid?" → threshold  (2D grid = rule_2d_sweep — a threshold tool concept)
- "What is a 2D sweep?" → threshold
- "How does a 2D grid work?" → threshold
- "Are you ARIA?" → greeting  (identity question — not an AML topic)
- "What is your name?" → greeting
- "Who are you?" → greeting
- "Ahoy!" → greeting
- "Ahoy matey!" → greeting
- "What are true positives in AML monitoring?" → threshold  (TP/TN definitions are threshold/confusion-matrix concepts)
- "What are true negatives?" → threshold
- "What is the difference between TP and TN?" → threshold
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
        try:
            classify_messages = [{"role": "system", "content": _CLASSIFY_SYSTEM}]
            if last_assistant:
                classify_messages.append({"role": "assistant", "content": last_assistant})
            classify_messages.append({"role": "user", "content": query})
            resp = self._client.chat.completions.create(
                model=OLLAMA_MODEL,
                max_tokens=20,
                temperature=0,
                messages=classify_messages,
            )
            raw = resp.choices[0].message.content or ""
            valid = {"threshold", "segmentation", "ofac", "policy", "greeting", "out_of_scope"}
            labels = [l.strip().lower() for l in raw.split(",") if l.strip().lower() in valid]
        except Exception as e:
            print(f"[orchestrator] classification error: {e} — defaulting to policy", flush=True)
            labels = ["policy"]

        # Keyword override — correct obvious misrouting regardless of LLM output
        import difflib as _dl
        q_lower = query.lower()
        _words = q_lower.split()

        # Conceptual/definitional questions about AML concepts → always policy
        # Must run before is_threshold keyword check so "threshold tuning" / "segmentation"
        # questions aren't hijacked by the operational keyword override.
        _conceptual_verbs = r'\b(what is|what are|explain|how does|how do|describe|define|tell me about|overview of|walk me through)\b'
        _conceptual_topics = [
            "threshold tuning",
            "dynamic segmentation", "behavioral segmentation", "customer segmentation",
            "segmentation approach", "segmentation work", "segmentation method",
            "k-means", "kmeans clustering",
            "sar backtesting", "sar backtest", "backtesting",
            "sanctions", "sanctions list",
            "aria help", "aria assist",   # "how does ARIA help with X"
        ]
        if (re.search(_conceptual_verbs, q_lower)
                and any(kw in q_lower for kw in _conceptual_topics)):
            print("[orchestrator] conceptual pre-check → conceptual", flush=True)
            return ["conceptual"]

        def _fuzzy(word_list, terms, cutoff=0.82):
            """True if any query word is a close match to any term (handles typos)."""
            for w in word_list:
                if len(w) < 4:
                    continue
                if _dl.get_close_matches(w, terms, n=1, cutoff=cutoff):
                    return True
            return False

        # Dataset summary / count queries → always threshold (segment_stats tool)
        _is_dataset_summary = any(p in q_lower for p in [
            "how many customers", "how many alerts", "how many accounts",
            "total customers", "total alerts", "total accounts",
            "customers and alerts", "alerts and customers",
            "in the dataset", "summary of the data", "data summary",
            "give me a summary", "overview of the data", "dataset overview",
            "how much data", "size of the dataset",
        ])
        if _is_dataset_summary:
            labels = ["threshold"]
            print("[orchestrator] keyword override → threshold (dataset summary / count query)")

        is_segmentation = (
            any(w in q_lower for w in ["cluster", "k-means", "kmeans", "treemap"])
            or _fuzzy(_words, ["cluster", "clustering", "segmentation", "kmeans"])
        )
        is_threshold = (
            any(w in q_lower for w in ["sweep", "fp", "fn", "sar", "heatmap", "backtest", "tuning", "threshold", "2d grid", "2d analysis", "grid analysis", "true positive", "true negative"])
            or _fuzzy(_words, ["threshold", "tuning", "backtest", "heatmap", "sweep"])
        )
        is_rule_query = (
            any(w in q_lower for w in ["rule", "rules", "false positive", "false negative", "precision", "layering", "structuring", "structr"])
            or _fuzzy(_words, ["precision"])
        )
        # "cluster N" in a sweep/backtest query = filter, not segmentation request
        cluster_as_filter = is_threshold and is_segmentation
        # "rule performance for Cluster X" / "which rules in Cluster 4" → cluster_rule_summary
        rule_cluster = is_rule_query and is_segmentation
        if rule_cluster:
            labels = ["threshold"]
            print("[orchestrator] keyword override → threshold (rule+cluster → cluster_rule_summary)")
        elif is_segmentation and not is_threshold:
            labels = ["segmentation"]
            print("[orchestrator] keyword override → segmentation")
        elif cluster_as_filter:
            labels = ["threshold"]
            print("[orchestrator] keyword override → threshold (cluster is a filter, not segmentation)")
        elif is_rule_query and "policy" in labels and "threshold" in labels:
            labels = ["threshold"]
            print("[orchestrator] keyword override → threshold (rule query, dropped policy)")
        # Rescue FP/FN/TP/TN/2D definitional questions classified as "policy" → threshold
        _fn_fp_kw = ["false positive", "false negative", "true positive", "true negative", "2d grid", "2d sweep"]
        if labels == ["policy"] and not is_segmentation and not is_threshold and any(kw in q_lower for kw in _fn_fp_kw):
            labels = ["threshold"]
            print("[orchestrator] keyword override → threshold (FP/FN/2D definition rescued from policy)")

        # Greetings and social acknowledgments
        _greeting_tokens = {"hello", "hi", "hey", "howdy", "greetings", "ahoy"}
        _social_phrases  = ["thanks", "thank you", "that was helpful", "that's helpful",
                            "got it", "great, thanks", "sounds good", "perfect, thanks",
                            "appreciate it", "cheers", "ahoy matey"]
        _identity_phrases = ["what is your name", "what's your name", "who are you",
                             "are you aria", "your name is", "tell me your name"]
        # Only match standalone capability questions — patterns that cannot be
        # a prefix of "how does ARIA help *with X*" or similar topic queries.
        _capability_phrases = [
            "what can aria do",
            "what does aria do",
            "what is aria",
            "aria's capabilities",
            "what are your capabilities",
            "what does aria offer",
            "aria features",
            "aria functionality",
        ]
        _is_capability = any(p in q_lower for p in _capability_phrases)
        _is_social = (q_lower.strip() in _greeting_tokens
                      or any(q_lower.strip().startswith(p) or q_lower.strip() == p
                             for p in _social_phrases))
        _is_identity = any(p in q_lower for p in _identity_phrases)
        if _is_capability:
            labels = ["capability"]
            print("[orchestrator] keyword override → capability (ARIA capability question)")
        elif _is_identity:
            labels = ["greeting"]
            print("[orchestrator] keyword override → greeting (identity question)")
        # Prevent data questions from being misclassified as greeting → policy instead
        is_data_question = any(w in q_lower for w in [
            "show me", "can you show", "credit", "score", "income", "balance",
            "distribution", "customers", "average", "what is the",
        ])
        if "greeting" in labels and is_data_question:
            labels = ["policy"]
            print("[orchestrator] keyword override → policy (data question misclassified as greeting)")

        # Social sentences where an AML term appears as a proper noun → out_of_scope
        # e.g. "My dog OFAC met a cat", "OFAC said hello", "My SAR is acting strange"
        _social_aml_noun = re.search(
            r'\b(my|our|his|her|their|the)\s+(dog|cat|friend|colleague|boss|wife|husband|kid|son|daughter|team|neighbor)\b',
            q_lower
        )
        if _social_aml_noun:
            labels = ["out_of_scope"]
            print("[orchestrator] keyword override → out_of_scope (social sentence containing AML term)")

        # OFAC guard: only route to ofac agent when there is explicit screening-action context
        if "ofac" in labels:
            _ofac_action = any(p in q_lower for p in [
                "screen", "sdn", "scan", "ofac hit", "ofac exposure", "check ofac",
                "run ofac", "ofac check", "ofac list", "sanctions", "sanctioned",
                "sanction list", "iran", "north korea", "dprk",
            ])
            if not _ofac_action:
                labels = ["policy"]
                print("[orchestrator] OFAC reclassified → policy (no screening action context)")

        # Threshold column names as bare replies (clarification follow-ups)
        _THRESHOLD_COLS = {"avg_trxns_week", "avg_trxn_amt", "trxn_amt_monthly"}
        if q_lower.strip() in _THRESHOLD_COLS:
            labels = ["threshold"]
            print("[orchestrator] keyword override → threshold (bare column name reply)")

        # Keyword fallback when fine-tuned model ignores classification prompt
        if not labels:
            q = query.lower()
            if any(w in q for w in ["threshold", "sweep", "fp", "fn", "sar", "heatmap", "rule", "alert", "tuning", "backtest",
                                      "avg_trxns_week", "avg_trxn_amt", "trxn_amt_monthly",
                                      "false positive", "false negative", "true positive", "true negative",
                                      "2d grid", "2d sweep", "2d analysis", "grid analysis"]):
                labels = ["threshold"]
            elif any(w in q for w in ["cluster", "segment", "k-means", "kmeans", "treemap"]):
                labels = ["segmentation"]
            elif any(re.fullmatch(w, tok) for w in ["hello", "hi", "hey", "howdy", "greetings"] for tok in q.split()):
                labels = ["greeting"]
            else:
                labels = ["policy"]
            print(f"[orchestrator] keyword fallback labels: {labels}", flush=True)

        # Final guard: data questions must never route to greeting
        if "greeting" in labels and is_data_question:
            labels = ["policy"]
            print("[orchestrator] post-fallback override → policy (data question)")

        print(f"[orchestrator] routing to: {labels}", flush=True)
        return labels

    def run(self, query: str, tool_executor, last_assistant: str = "", history: list = None, last_cluster_result: str = "", last_rule_list: str = "") -> tuple:
        """
        Route query via LLM, run required agents (in parallel if >1), merge results.
        Returns: (combined_text, all_chart_results)
        """
        labels = self._route(query, last_assistant)

        # Sticky routing: elliptical follow-ups routed to policy/weak labels → re-use prior agent.
        # Covers "and the youngest", "which one has lowest", "top 3 by precision", "what about days_required?"
        _STICKY_SOURCES = {"threshold", "segmentation"}
        _WEAK_LABELS = {"policy", "out_of_scope", "greeting", "capability"}
        if (self._last_agent in _STICKY_SOURCES
                and set(labels) <= _WEAK_LABELS
                and _is_elliptical(query)):
            print(f"[orchestrator] sticky routing -> {self._last_agent} (elliptical follow-up)", flush=True)
            labels = [self._last_agent]

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

        if "capability" in labels:
            return self._CAPABILITY, []

        if "out_of_scope" in labels:
            return self._OUT_OF_SCOPE, []

        if "conceptual" in labels:
            return self.policy_agent.run(query, tool_executor, _prior_context, history)

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
            if last_rule_list:
                _rule_ctx = last_rule_list[:4000] if len(last_rule_list) > 4000 else last_rule_list
                context = f"[PREVIOUS RULE LIST]\n{_rule_ctx}\n[END RULE LIST]"
                print(f"[orchestrator] injecting previous rule list for follow-up ({len(_rule_ctx)} chars):\n{_rule_ctx}\n[END]")
            if name == "segmentation":
                _cluster_ctx = last_cluster_result or last_assistant
                # If the stored stats are short (alert counts only, no cluster attributes),
                # combine with the assistant's previous response which may have more detail.
                if _cluster_ctx and last_assistant and len(_cluster_ctx) < 500 and len(last_assistant) > len(_cluster_ctx):
                    _cluster_ctx = _cluster_ctx + "\n\n" + last_assistant
                if _cluster_ctx and "Cluster" in _cluster_ctx:
                    # Trim to ~2500 chars to avoid context overflow in long conversations
                    if len(_cluster_ctx) > 2500:
                        lines = [l for l in _cluster_ctx.splitlines() if l.strip().startswith("Cluster")]
                        _cluster_ctx = "\n".join(lines[:20]) if lines else _cluster_ctx[:1500]
                    context = f"[PREVIOUS CLUSTERING RESULT]\n{_cluster_ctx}\n[END PREVIOUS RESULT]"
                    print(f"[orchestrator] injecting previous cluster context ({len(_cluster_ctx)} chars)")
            return agent.run(query, tool_executor, context, history)

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
