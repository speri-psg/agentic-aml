"""Tests for agents/orchestrator.py — keyword routing override logic (LLM mocked)."""
import pytest
from unittest.mock import MagicMock, patch


def _make_orchestrator(llm_label="out_of_scope"):
    """
    Build an OrchestratorAgent with all LLM calls returning a fixed label.
    """
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = llm_label

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("openai.OpenAI", return_value=mock_client):
        from agents.orchestrator import OrchestratorAgent
        orch = OrchestratorAgent()

    # Override the classify client on the instance so LLM label is controlled
    orch._client = mock_client
    return orch, mock_client


# ── Greeting routing ───────────────────────────────────────────────────────────

class TestGreetingRouting:
    def test_greeting_token_not_overridden(self):
        # LLM correctly labels "hello" as greeting — verify no override fires
        orch, _ = _make_orchestrator("greeting")
        labels = orch._route("hello")
        assert "greeting" in labels

    def test_hi_token_not_overridden(self):
        orch, _ = _make_orchestrator("greeting")
        labels = orch._route("hi")
        assert "greeting" in labels

    def test_data_question_overrides_greeting(self):
        # LLM correctly returns segmentation for a distribution query — not greeting
        orch, _ = _make_orchestrator("segmentation")
        labels = orch._route("show me the customer distribution")
        assert "greeting" not in labels

    def test_data_question_with_balance_overrides_greeting(self):
        # LLM correctly returns threshold for a stats query — not greeting
        orch, _ = _make_orchestrator("threshold")
        labels = orch._route("what is the average balance?")
        assert "greeting" not in labels

    def test_thanks_not_overridden_when_llm_says_greeting(self):
        orch, _ = _make_orchestrator("greeting")
        labels = orch._route("Thanks, that was helpful!")
        assert "greeting" in labels

    def test_got_it_thanks_not_overridden(self):
        orch, _ = _make_orchestrator("greeting")
        labels = orch._route("Got it, thanks")
        assert "greeting" in labels

    def test_thank_you_not_overridden(self):
        orch, _ = _make_orchestrator("greeting")
        labels = orch._route("Thank you")
        assert "greeting" in labels


# ── EU/UN policy keyword override ─────────────────────────────────────────────

class TestPolicyKeywordOverride:
    """Behavior change (production_switchover): when the LLM classifier fails,
    the keyword fallback now ONLY rescues threshold/segmentation queries (where
    we have local data to query against). Policy questions without those data
    keywords route to out_of_scope rather than risk a hallucinated answer from
    a backup heuristic. The LLM classifier handles these correctly in
    production — these mock-classifier tests cover the failure mode.
    See feedback_instruction_placement / production_switchover_plan."""

    def test_beneficial_ownership_falls_back_to_out_of_scope(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What are the EU requirements for beneficial ownership registers?")
        assert "out_of_scope" in labels

    def test_amld_falls_back_to_out_of_scope(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What does the 4th AMLD require for customer due diligence?")
        assert "out_of_scope" in labels

    def test_un_resolution_falls_back_to_out_of_scope(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What does UN Security Council Resolution 1373 require of banks?")
        assert "out_of_scope" in labels

    def test_fatf_falls_back_to_out_of_scope(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What are FATF recommendations for AML programs?")
        assert "out_of_scope" in labels

    def test_beneficial_owner_falls_back_to_out_of_scope(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What are the beneficial owner disclosure requirements?")
        assert "out_of_scope" in labels

    def test_tructuring_typo_falls_back_to_out_of_scope(self):
        # Typo for "structuring" — no threshold/seg keywords → fallback → out_of_scope
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("what is tructuring")
        assert "out_of_scope" in labels

    def test_smurfing_falls_back_to_out_of_scope(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What is smurfing?")
        assert "out_of_scope" in labels


# ── Segmentation keyword override ─────────────────────────────────────────────

class TestSegmentationKeywordOverride:
    def test_cluster_keyword_routes_to_segmentation(self):
        # "cluster" keyword in fallback list → segmentation when LLM fails
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("cluster all customers")
        assert "segmentation" in labels

    def test_kmeans_keyword_routes_to_segmentation(self):
        # "kmeans" keyword in fallback list → segmentation when LLM fails
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("run kmeans on the data")
        assert "segmentation" in labels

    def test_segmentation_keyword_routes_to_segmentation(self):
        # "segmentation" has no fallback keyword — relies on LLM correct classification
        orch, _ = _make_orchestrator("segmentation")
        labels = orch._route("run customer segmentation")
        assert "segmentation" in labels

    def test_treemap_keyword_routes_to_segmentation(self):
        # "treemap" keyword in fallback list → segmentation when LLM fails
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("show me the treemap")
        assert "segmentation" in labels

    def test_pure_segmentation_not_mixed_with_threshold(self):
        # "cluster" keyword → segmentation; no threshold keyword present
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("which cluster has the most activity?")
        assert "segmentation" in labels
        assert "threshold" not in labels

    def test_rule_performance_for_cluster_routes_to_threshold(self):
        # LLM correctly returns threshold for rule performance queries
        orch, _ = _make_orchestrator("threshold")
        labels = orch._route("Show all rule performance for Cluster 4")
        assert "threshold" in labels
        assert "segmentation" not in labels

    def test_which_rules_in_cluster_routes_to_threshold(self):
        # LLM correctly returns threshold for rule ranking queries
        orch, _ = _make_orchestrator("threshold")
        labels = orch._route("Which rules perform best in Cluster 2?")
        assert "threshold" in labels
        assert "segmentation" not in labels


# ── Threshold keyword override ────────────────────────────────────────────────

class TestThresholdKeywordOverride:
    def test_cluster_as_filter_routes_to_threshold(self):
        # "sar" keyword wins over "cluster" in the fallback priority order
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("show SAR backtest for Cluster 3")
        assert "threshold" in labels
        assert "segmentation" not in labels

    def test_fp_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("show fp tuning for Business")
        assert "threshold" in labels

    def test_fn_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("what happens to fn if I raise threshold?")
        assert "threshold" in labels

    def test_sar_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("what is the sar catch rate?")
        assert "threshold" in labels

    def test_backtest_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("run a backtest for Individual customers")
        assert "threshold" in labels

    def test_sweep_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("show rule sweep for Activity Deviation")
        assert "threshold" in labels

    def test_heatmap_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("show the heatmap for Elder Abuse")
        assert "threshold" in labels

    def test_threshold_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("what threshold catches 90% of SARs?")
        assert "threshold" in labels

    def test_threshold_keyword_in_fallback(self):
        # Exact threshold keyword in fallback (no typo) → threshold
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("show threshold analysis for Business")
        assert "threshold" in labels

    def test_rule_keyword_rescued_from_out_of_scope(self):
        # "fp" keyword triggers threshold fallback when LLM fails
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("which rules have the highest fp rate?")
        assert "threshold" in labels

    def test_list_rules_query_rescued_when_classifier_empty(self):
        # Regression: "show me a list of all AML rules in the system" was being
        # routed to out_of_scope when the LLM classifier returned empty string.
        # The keyword fallback list now includes "rule"/"rules" so list_rules-style
        # queries get rescued to threshold even on classifier failure.
        orch, _ = _make_orchestrator("")
        labels = orch._route("show me a list of all AML rules in the system")
        assert "threshold" in labels
        assert "out_of_scope" not in labels

    def test_list_rules_query_rescued_when_classifier_returns_out_of_scope(self):
        # Companion to the above: when the LLM mis-classifies a list-rules query
        # as out_of_scope (vs returning empty), the OOS rescue path catches it.
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("list all AML rules")
        assert "threshold" in labels
        assert "out_of_scope" not in labels

    def test_rule_with_threshold_and_policy_keeps_threshold(self):
        # Multi-label results are returned as-is; threshold must be present
        orch, _ = _make_orchestrator("threshold,policy")
        labels = orch._route("show rule performance for Activity Deviation rule")
        assert "threshold" in labels

    def test_cluster_sweep_query_is_threshold(self):
        # "sar" keyword wins over "cluster" in fallback priority
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("run SAR backtest for Activity Deviation ACH in Cluster 2")
        assert "threshold" in labels

    def test_2d_grid_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("Show me a 2D grid for Elder Abuse")
        assert "threshold" in labels

    def test_2d_analysis_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("Show 2D analysis for Detect Excessive Transaction Activity")
        assert "threshold" in labels

    def test_grid_analysis_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("Show grid analysis for Activity Deviation ACH")
        assert "threshold" in labels


# ── OFAC keyword override ──────────────────────────────────────────────────────

class TestOfacKeywordOverride:
    def test_ofac_keyword_routes_to_ofac(self):
        # LLM classifies as ofac; OFAC guard keeps it because "screen" is an action verb
        orch, _ = _make_orchestrator("ofac")
        labels = orch._route("run OFAC screening")
        assert "ofac" in labels

    def test_sdn_keyword_routes_to_ofac(self):
        orch, _ = _make_orchestrator("ofac")
        labels = orch._route("screen against the SDN list")
        assert "ofac" in labels

    def test_sanctions_keyword_routes_to_ofac(self):
        orch, _ = _make_orchestrator("ofac")
        labels = orch._route("show sanctioned country exposure")
        assert "ofac" in labels

    def test_ofac_data_query_falls_back_to_out_of_scope(self):
        # Behavior change (production_switchover): policy keyword fallback was
        # removed — only threshold/segmentation get keyword rescue. OFAC-related
        # queries with no data-query keywords route to out_of_scope when the
        # LLM classifier fails.
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("which customers have ofac hits?")
        assert "out_of_scope" in labels
        assert "ofac" not in labels

    def test_how_many_customers_ofac_falls_back_to_out_of_scope(self):
        orch, _ = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("which customers show ofac exposure?")
        assert "out_of_scope" in labels


# ── Empty label fallback ──────────────────────────────────────────────────────

class TestKeywordFallback:
    def test_threshold_fallback_when_empty_labels(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        # LLM returns invalid label → parsed labels list is empty → keyword fallback
        labels = orch._route("show me fp/fn tuning for Business")
        assert "threshold" in labels

    def test_segmentation_fallback_keyword(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("cluster all Individual customers")
        assert "segmentation" in labels

    # ── Policy-keyword fallback removed (production_switchover) ──────────────
    # Policy questions are now expected to be handled by the LLM classifier.
    # When the classifier fails AND no threshold/segmentation keywords match,
    # the fallback returns out_of_scope rather than risk an unrelated answer.

    def test_kyc_falls_back_to_out_of_scope(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("what is know your customer KYC?")
        assert "out_of_scope" in labels

    def test_canada_falls_back_to_out_of_scope(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What does AML compliance in Canada require?")
        assert "out_of_scope" in labels

    def test_fintrac_falls_back_to_out_of_scope(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What does FINTRAC require for suspicious transaction reporting?")
        assert "out_of_scope" in labels

    def test_typology_falls_back_to_out_of_scope(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What is AML typology?")
        assert "out_of_scope" in labels

    def test_layering_falls_back_to_out_of_scope(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What is AML layering?")
        assert "out_of_scope" in labels

    def test_ofac_label_preserved_through_guard(self):
        # When LLM returns "ofac" and a screening action is present, label persists
        orch, mock_client = _make_orchestrator("ofac")
        labels = orch._route("run ofac screen on portfolio")
        assert "ofac" in labels


# ── Segmentation→threshold rescue (single-label misroute on threshold-tool query) ───

class TestSegmentationRescue:
    """The empty-ack classifier (commit 5247fcc) is more decisive and sometimes
    commits to single-label `segmentation` on cluster-filtered threshold-tool
    queries — e.g. "run adaptive thresholds on Individual customers" or
    "Show Elder Abuse SAR backtest for Cluster 4". With single-label
    segmentation the Phase 5 multi-label suppression can't fire (it requires
    both threshold and segmentation). The rescue promotes to threshold when
    the query contains a threshold-tool keyword."""

    def test_adaptive_thresholds_with_segment_word(self):
        orch, _ = _make_orchestrator("segmentation")
        labels = orch._route("run adaptive thresholds on Individual customers")
        assert labels == ["threshold"]

    def test_backtest_for_cluster(self):
        orch, _ = _make_orchestrator("segmentation")
        labels = orch._route("Show Elder Abuse SAR backtest for Cluster 4")
        assert labels == ["threshold"]

    def test_2d_sweep_for_cluster(self):
        orch, _ = _make_orchestrator("segmentation")
        labels = orch._route("show 2D sweep for Activity Deviation ACH on Cluster 1")
        assert labels == ["threshold"]

    def test_legitimate_segmentation_unaffected(self):
        # No threshold-tool keyword present → rescue does NOT fire
        orch, _ = _make_orchestrator("segmentation")
        labels = orch._route("cluster Business customers by transaction behavior")
        assert labels == ["segmentation"]

    def test_alerts_distribution_unaffected(self):
        orch, _ = _make_orchestrator("segmentation")
        labels = orch._route("show alert distribution across segments")
        assert labels == ["segmentation"]

    def test_multi_label_unaffected_at_route(self):
        # _route returns raw classifier labels; Phase 5 suppression happens
        # later in run(). Verify rescue doesn't accidentally fire on
        # multi-label input.
        orch, _ = _make_orchestrator("threshold,segmentation")
        labels = orch._route("Show Elder Abuse SAR backtest for Cluster 4")
        assert set(labels) == {"threshold", "segmentation"}


# ── LLM classification error handling ────────────────────────────────────────

class TestClassificationErrorHandling:
    def test_llm_exception_defaults_to_policy(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("connection refused")

        with patch("openai.OpenAI", return_value=mock_client):
            from agents.orchestrator import OrchestratorAgent
            orch = OrchestratorAgent()
        orch._client = mock_client

        # LLM exception → except block sets labels=["policy"]; no keyword overrides for pizza
        labels = orch._route("some completely irrelevant question about pizza")
        assert "policy" in labels


# ── run() method — response construction ─────────────────────────────────────

class TestOrchestratorRun:
    def test_greeting_returns_greeting_text(self):
        orch, _ = _make_orchestrator("greeting")
        text, charts = orch.run("hello", tool_executor=MagicMock())
        assert "ARIA" in text
        assert charts == []

    def test_out_of_scope_returns_refusal(self):
        orch, _ = _make_orchestrator("out_of_scope")
        # Query with no keyword override signals → out_of_scope
        text, charts = orch.run("what is the weather today?", tool_executor=MagicMock())
        # Should return the out-of-scope response (no agents run)
        assert isinstance(text, str)
        assert isinstance(charts, list)

    def test_ofac_name_lookup_triggered_by_capitalised_name(self):
        orch, _ = _make_orchestrator("ofac")
        mock_executor = MagicMock(return_value=("John Smith OFAC result", None))
        # Run with a query that has a capitalised name
        text, charts = orch.run("screen John Smith against OFAC", tool_executor=mock_executor)
        # Should have called ofac_name_lookup
        mock_executor.assert_called_once()
        call_args = mock_executor.call_args
        assert call_args[0][0] == "ofac_name_lookup"

    def test_ofac_without_name_calls_ofac_screening(self):
        orch, _ = _make_orchestrator("ofac")
        mock_executor = MagicMock(return_value=("OFAC screening result", None))
        text, charts = orch.run("run OFAC screening", tool_executor=mock_executor)
        mock_executor.assert_called_once()
        call_args = mock_executor.call_args
        assert call_args[0][0] == "ofac_screening"

    def test_capability_returns_capability_text(self):
        orch, _ = _make_orchestrator("greeting")
        text, charts = orch.run("What can ARIA do?", tool_executor=MagicMock())
        assert "threshold" in text.lower() or "Threshold" in text
        assert charts == []

    def test_rule_list_injected_to_threshold_agent(self):
        orch, _ = _make_orchestrator("threshold")
        mock_run = MagicMock(return_value=("bottom 3 answer", []))
        orch.threshold_agent.run = mock_run

        rule_list = "=== RULE LIST ===\nRound-trip: precision=35%\n=== END RULE LIST ==="
        orch.run(
            "top 3 by precision",
            tool_executor=MagicMock(),
            last_rule_list=rule_list,
        )

        context_passed = mock_run.call_args[0][2]
        assert "=== RULE LIST ===" in context_passed
        assert "Round-trip" in context_passed

    def test_empty_rule_list_no_injection(self):
        orch, _ = _make_orchestrator("threshold")
        mock_run = MagicMock(return_value=("answer", []))
        orch.threshold_agent.run = mock_run

        orch.run(
            "what are the top 3 rules by precision?",
            tool_executor=MagicMock(),
            last_rule_list="",
        )

        context_passed = mock_run.call_args[0][2]
        assert "=== RULE LIST ===" not in context_passed

    def test_cluster_context_injected_for_segmentation_agent(self):
        orch, _ = _make_orchestrator("segmentation")
        mock_run = MagicMock(return_value=("segment answer", []))
        orch.segmentation_agent.run = mock_run

        cluster_result = "Cluster 1: 100 customers\nCluster 2: 200 customers"
        orch.run(
            "which cluster has the most alerts?",
            tool_executor=MagicMock(),
            last_cluster_result=cluster_result,
        )

        context_passed = mock_run.call_args[0][2]
        assert "[PREVIOUS CLUSTERING RESULT]" in context_passed
        assert "Cluster 1" in context_passed

    def test_cluster_result_without_cluster_keyword_not_injected(self):
        orch, _ = _make_orchestrator("segmentation")
        mock_run = MagicMock(return_value=("answer", []))
        orch.segmentation_agent.run = mock_run

        # last_cluster_result has no "Cluster" substring → injection skipped
        orch.run(
            "which cluster has the most alerts?",
            tool_executor=MagicMock(),
            last_cluster_result="no cluster data here",
        )

        context_passed = mock_run.call_args[0][2]
        assert "[PREVIOUS CLUSTERING RESULT]" not in context_passed

    def test_rule_list_injected_for_threshold_queries(self):
        """Behavior change (v22+): the orchestrator now ALWAYS injects last_rule_list
        for threshold queries when one is available, even for fresh "list all rules"
        requests. The model is trusted to call list_rules to refresh if needed —
        having stale context is preferable to forgetting prior context mid-session.
        Previous behavior tried to suppress injection on "list rules" queries; that
        guard was removed because it produced more bugs than it fixed."""
        orch, _ = _make_orchestrator("threshold")
        mock_run = MagicMock(return_value=("rules listed", []))
        orch.threshold_agent.run = mock_run
        rule_list = "=== RULE LIST ===\nRound-trip: precision=35%\n=== END RULE LIST ==="
        orch.run("show me all AML rules", tool_executor=MagicMock(), last_rule_list=rule_list)
        context_passed = mock_run.call_args[0][2]
        assert "=== RULE LIST ===" in context_passed

    def test_threshold_agent_receives_empty_history(self):
        orch, _ = _make_orchestrator("threshold")
        mock_run = MagicMock(return_value=("answer", []))
        orch.threshold_agent.run = mock_run
        history = [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "reply"}]
        orch.run("show sar backtest for Elder Abuse", tool_executor=MagicMock(), history=history)
        history_passed = mock_run.call_args[0][3]
        assert history_passed == []

    def test_segmentation_agent_receives_full_history(self):
        orch, _ = _make_orchestrator("segmentation")
        mock_run = MagicMock(return_value=("answer", []))
        orch.segmentation_agent.run = mock_run
        history = [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "reply"}]
        orch.run("cluster Business customers", tool_executor=MagicMock(), history=history)
        history_passed = mock_run.call_args[0][3]
        assert history_passed == history


# ── Multi-label suppression (cluster_segment_plan Phase 5) ────────────────────

class TestMultiLabelSuppression:
    """Classifier sometimes emits [threshold, segmentation] for queries that are
    really threshold-tool operations with a cluster filter. The segmentation
    agent has no tool for these and fabricates threshold numbers in parallel.
    Suppression drops segmentation when threshold-tool keywords are present.
    Per cluster_segment_plan.md Phase 5."""

    def test_backtest_for_cluster_routes_threshold_only(self):
        orch, _ = _make_orchestrator("threshold,segmentation")
        mock_threshold = MagicMock(return_value=("threshold answer", []))
        mock_segmentation = MagicMock(return_value=("segmentation answer", []))
        orch.threshold_agent.run = mock_threshold
        orch.segmentation_agent.run = mock_segmentation
        orch.run("Show Elder Abuse SAR backtest for Cluster 4", tool_executor=MagicMock())
        mock_threshold.assert_called_once()
        mock_segmentation.assert_not_called()

    def test_adaptive_thresholds_on_business_routes_threshold_only(self):
        orch, _ = _make_orchestrator("threshold,segmentation")
        mock_threshold = MagicMock(return_value=("threshold answer", []))
        mock_segmentation = MagicMock(return_value=("segmentation answer", []))
        orch.threshold_agent.run = mock_threshold
        orch.segmentation_agent.run = mock_segmentation
        orch.run("run adaptive thresholds on business customers", tool_executor=MagicMock())
        mock_threshold.assert_called_once()
        mock_segmentation.assert_not_called()

    def test_2d_sweep_for_cluster_routes_threshold_only(self):
        orch, _ = _make_orchestrator("threshold,segmentation")
        mock_threshold = MagicMock(return_value=("threshold answer", []))
        mock_segmentation = MagicMock(return_value=("segmentation answer", []))
        orch.threshold_agent.run = mock_threshold
        orch.segmentation_agent.run = mock_segmentation
        orch.run("Show 2D sweep for Velocity Single on Cluster 2", tool_executor=MagicMock())
        mock_threshold.assert_called_once()
        mock_segmentation.assert_not_called()

    def test_dual_intent_without_tool_keyword_runs_both(self):
        # No analytical-tool keyword present → multi-label preserved (rare case)
        orch, _ = _make_orchestrator("threshold,segmentation")
        mock_threshold = MagicMock(return_value=("threshold answer", []))
        mock_segmentation = MagicMock(return_value=("segmentation answer", []))
        orch.threshold_agent.run = mock_threshold
        orch.segmentation_agent.run = mock_segmentation
        orch.run("cluster customers and summarize", tool_executor=MagicMock())
        # Both should fire — no suppression keyword in the query
        mock_threshold.assert_called_once()
        mock_segmentation.assert_called_once()

    def test_single_label_threshold_unaffected(self):
        orch, _ = _make_orchestrator("threshold")
        mock_threshold = MagicMock(return_value=("answer", []))
        orch.threshold_agent.run = mock_threshold
        orch.run("Show Elder Abuse SAR backtest for Cluster 4", tool_executor=MagicMock())
        mock_threshold.assert_called_once()

    def test_single_label_segmentation_unaffected(self):
        orch, _ = _make_orchestrator("segmentation")
        mock_segmentation = MagicMock(return_value=("answer", []))
        orch.segmentation_agent.run = mock_segmentation
        orch.run("cluster Business customers by transaction behavior", tool_executor=MagicMock())
        mock_segmentation.assert_called_once()


# ── Conceptual label routing ──────────────────────────────────────────────────

class TestConceptualRouting:
    def test_operational_cluster_query_not_threshold(self):
        # Operational cluster query routes to segmentation, not threshold
        orch, _ = _make_orchestrator("segmentation")
        labels = orch._route("Cluster customers into groups")
        assert "segmentation" in labels
        assert labels != ["threshold"]

    def test_what_is_question_routes_to_policy(self):
        # "What is X?" conceptual questions → LLM returns policy, delegates to policy agent
        orch, _ = _make_orchestrator("policy")
        mock_run = MagicMock(return_value=("explanation text", []))
        orch.policy_agent.run = mock_run
        orch.run("What is threshold tuning?", tool_executor=MagicMock())
        mock_run.assert_called_once()


# ── Dataset summary keyword override ─────────────────────────────────────────

class TestDatasetSummaryRouting:
    def test_how_many_customers_routes_to_threshold(self):
        # Dataset count queries have no fallback keyword — LLM correctly returns threshold
        orch, _ = _make_orchestrator("threshold")
        labels = orch._route("how many customers are in the dataset?")
        assert "threshold" in labels

    def test_how_many_alerts_routes_to_threshold(self):
        orch, _ = _make_orchestrator("threshold")
        labels = orch._route("how many alerts does the system have?")
        assert "threshold" in labels

    def test_total_customers_routes_to_threshold(self):
        orch, _ = _make_orchestrator("threshold")
        labels = orch._route("total customers in the portfolio")
        assert "threshold" in labels

    def test_data_summary_routes_to_threshold(self):
        orch, _ = _make_orchestrator("threshold")
        labels = orch._route("give me a summary of the data")
        assert "threshold" in labels


# ── Extended sticky: cluster attribute follow-ups rescued from threshold ──────

class TestClusterAttributeStickyRouting:
    """Elliptical cluster attribute follow-ups misclassified as threshold must be
    rescued to segmentation when a prior cluster session is active."""

    _CLUSTER_CTX = (
        "=== PRE-COMPUTED CLUSTER STATS ===\n"
        "**Cluster 1**\n- Monthly Txn Volume: **$12,134**\n"
        "**Cluster 2**\n- Monthly Txn Volume: **$14,967**\n"
        "=== END PRE-COMPUTED CLUSTER STATS ==="
    )

    def _run_after_seg(self, query):
        """Run orchestrator with LLM returning 'threshold', last_agent=segmentation."""
        orch, _ = _make_orchestrator("threshold")   # LLM says threshold
        orch._last_agent = "segmentation"
        mock_seg = MagicMock(return_value=("cluster answer", []))
        orch.segmentation_agent.run = mock_seg
        orch.run(query, tool_executor=MagicMock(),
                 last_cluster_result=self._CLUSTER_CTX)
        return mock_seg

    def test_highest_monthly_rescued_to_segmentation(self):
        mock_seg = self._run_after_seg("which one has the highest monthly trxn amount")
        mock_seg.assert_called_once()

    def test_lowest_monthly_volume_rescued(self):
        mock_seg = self._run_after_seg("which one is the lowest by monthly volume")
        mock_seg.assert_called_once()

    def test_and_the_lowest_rescued(self):
        mock_seg = self._run_after_seg("and the lowest")
        mock_seg.assert_called_once()

    def test_which_one_is_the_highest_rescued(self):
        mock_seg = self._run_after_seg("which one is the highest")
        mock_seg.assert_called_once()

    def test_sar_backtest_not_rescued(self):
        """Query with a threshold tool keyword (sar backtest) must NOT be rescued to segmentation."""
        orch, _ = _make_orchestrator("threshold")
        orch._last_agent = "segmentation"
        mock_seg = MagicMock(return_value=("answer", []))
        mock_thr = MagicMock(return_value=("backtest answer", []))
        orch.segmentation_agent.run = mock_seg
        orch.threshold_agent.run = mock_thr
        # "sar backtest" is in _has_rule_or_tool → rescue guard must block
        orch.run("which one has the highest sar backtest",
                 tool_executor=MagicMock(), last_cluster_result=self._CLUSTER_CTX)
        mock_thr.assert_called_once()
        mock_seg.assert_not_called()

    def test_no_cluster_context_not_rescued(self):
        """Without active cluster context, threshold routing must stand."""
        orch, _ = _make_orchestrator("threshold")
        orch._last_agent = "segmentation"
        mock_seg = MagicMock(return_value=("answer", []))
        mock_thr = MagicMock(return_value=("threshold answer", []))
        orch.segmentation_agent.run = mock_seg
        orch.threshold_agent.run = mock_thr
        orch.run("which one has the highest monthly trxn amount",
                 tool_executor=MagicMock(), last_cluster_result="")  # no context
        mock_thr.assert_called_once()
        mock_seg.assert_not_called()


# ── Classifier <think> tag stripping ─────────────────────────────────────────

class TestClassifierThinkTagStripping:
    def test_think_tags_stripped_from_classifier_output(self):
        """When the classifier returns <think>...</think>label, label must be extracted."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "<think>Let me classify this.</think>\nthreshold"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        with patch("openai.OpenAI", return_value=mock_client):
            from agents.orchestrator import OrchestratorAgent
            orch = OrchestratorAgent()
        orch._client = mock_client
        labels = orch._route("show SAR backtest for Elder Abuse")
        assert "threshold" in labels

    def test_multiline_think_block_stripped(self):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "<think>\nLine 1\nLine 2\n</think>segmentation"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        with patch("openai.OpenAI", return_value=mock_client):
            from agents.orchestrator import OrchestratorAgent
            orch = OrchestratorAgent()
        orch._client = mock_client
        labels = orch._route("cluster Business customers")
        assert "segmentation" in labels

    def test_empty_after_think_strip_falls_to_keyword_fallback(self):
        """If think tags consume entire response, keyword fallback must still work."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "<think>only thinking, no label</think>"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        with patch("openai.OpenAI", return_value=mock_client):
            from agents.orchestrator import OrchestratorAgent
            orch = OrchestratorAgent()
        orch._client = mock_client
        labels = orch._route("show sar backtest for Business")
        assert "threshold" in labels
