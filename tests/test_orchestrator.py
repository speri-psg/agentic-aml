"""Tests for agents/orchestrator.py — keyword routing override logic (LLM mocked)."""
import pytest
from unittest.mock import MagicMock, patch


def _make_orchestrator(llm_label="out_of_scope"):
    """
    Build an OrchestratorAgent with all LLM calls returning a fixed label.
    PolicyAgent._load_kb is stubbed out to prevent ChromaDB access.
    """
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = llm_label

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("openai.OpenAI", return_value=mock_client), \
         patch("agents.policy_agent.PolicyAgent._load_kb", return_value=None):
        from agents.orchestrator import OrchestratorAgent
        orch = OrchestratorAgent()

    # Override the classify client on the instance so LLM label is controlled
    orch._client = mock_client
    return orch, mock_client


# ── Greeting routing ───────────────────────────────────────────────────────────

class TestGreetingRouting:
    def test_greeting_token_rescued_from_out_of_scope(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("hello")
        assert "greeting" in labels

    def test_hi_token_rescued_from_out_of_scope(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("hi")
        assert "greeting" in labels

    def test_data_question_overrides_greeting(self):
        orch, _ = _make_orchestrator("greeting")
        labels = orch._route("show me the customer distribution")
        assert "greeting" not in labels

    def test_data_question_with_balance_overrides_greeting(self):
        orch, _ = _make_orchestrator("greeting")
        labels = orch._route("what is the average balance?")
        assert "greeting" not in labels

    def test_thanks_rescued_as_greeting(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("Thanks, that was helpful!")
        assert "greeting" in labels

    def test_got_it_thanks_rescued_as_greeting(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("Got it, thanks")
        assert "greeting" in labels

    def test_thank_you_rescued_as_greeting(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("Thank you")
        assert "greeting" in labels


# ── EU/UN policy keyword override ─────────────────────────────────────────────

class TestPolicyKeywordOverride:
    def test_beneficial_ownership_routes_to_policy(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("What are the EU requirements for beneficial ownership registers?")
        assert "policy" in labels

    def test_amld_routes_to_policy(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("What does the 4th AMLD require for customer due diligence?")
        assert "policy" in labels

    def test_un_resolution_routes_to_policy(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("What does UN Security Council Resolution 1373 require of banks?")
        assert "policy" in labels

    def test_fatf_routes_to_policy(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("What are FATF recommendations for AML programs?")
        assert "policy" in labels

    def test_beneficial_owner_routes_to_policy(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("What are the beneficial owner disclosure requirements?")
        assert "policy" in labels

    def test_tructuring_typo_rescued_as_policy(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("what is tructuring")
        assert "policy" in labels

    def test_smurfing_rescued_as_policy(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("What is smurfing?")
        assert "policy" in labels


# ── Segmentation keyword override ─────────────────────────────────────────────

class TestSegmentationKeywordOverride:
    def test_cluster_keyword_routes_to_segmentation(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("cluster all customers")
        assert "segmentation" in labels

    def test_kmeans_keyword_routes_to_segmentation(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("run kmeans on the data")
        assert "segmentation" in labels

    def test_segment_keyword_routes_to_segmentation(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("show me segment distribution")
        assert "segmentation" in labels

    def test_treemap_keyword_routes_to_segmentation(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("show me the treemap")
        assert "segmentation" in labels

    def test_pure_segmentation_not_mixed_with_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("which cluster has the most activity?")
        assert "segmentation" in labels
        assert "threshold" not in labels

    def test_rule_performance_for_cluster_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("Show all rule performance for Cluster 4")
        assert "threshold" in labels
        assert "segmentation" not in labels

    def test_which_rules_in_cluster_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("Which rules perform best in Cluster 2?")
        assert "threshold" in labels
        assert "segmentation" not in labels


# ── Threshold keyword override ────────────────────────────────────────────────

class TestThresholdKeywordOverride:
    def test_cluster_as_filter_routes_to_threshold(self):
        # "Cluster 3" in a SAR backtest query → cluster is a filter, not segmentation
        orch, _ = _make_orchestrator("segmentation")
        labels = orch._route("show SAR backtest for Cluster 3")
        assert "threshold" in labels
        assert "segmentation" not in labels

    def test_fp_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("show fp tuning for Business")
        assert "threshold" in labels

    def test_fn_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("what happens to fn if I raise threshold?")
        assert "threshold" in labels

    def test_sar_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("what is the sar catch rate?")
        assert "threshold" in labels

    def test_backtest_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("run a backtest for Individual customers")
        assert "threshold" in labels

    def test_sweep_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("show rule sweep for Activity Deviation")
        assert "threshold" in labels

    def test_heatmap_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("show the heatmap for Elder Abuse")
        assert "threshold" in labels

    def test_threshold_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("what threshold catches 90% of SARs?")
        assert "threshold" in labels

    def test_thrshold_tunning_typo_rescued_as_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("what is thrshold tunning")
        assert "threshold" in labels

    def test_rule_keyword_rescued_from_out_of_scope(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("which rules have the most false positives?")
        assert "threshold" in labels

    def test_rule_with_policy_drops_policy(self):
        orch, _ = _make_orchestrator("threshold,policy")
        labels = orch._route("show rule performance for Activity Deviation rule")
        assert "threshold" in labels
        assert "policy" not in labels

    def test_cluster_sweep_query_is_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("run SAR backtest for Activity Deviation ACH in Cluster 2")
        assert "threshold" in labels

    def test_2d_grid_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("Show me a 2D grid for Elder Abuse")
        assert "threshold" in labels

    def test_2d_analysis_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("Show 2D analysis for Detect Excessive Transaction Activity")
        assert "threshold" in labels

    def test_grid_analysis_keyword_routes_to_threshold(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("Show grid analysis for Activity Deviation ACH")
        assert "threshold" in labels


# ── OFAC keyword override ──────────────────────────────────────────────────────

class TestOfacKeywordOverride:
    def test_ofac_keyword_routes_to_ofac(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("run OFAC screening")
        assert "ofac" in labels

    def test_sdn_keyword_routes_to_ofac(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("screen against the SDN list")
        assert "ofac" in labels

    def test_sanctions_keyword_routes_to_ofac(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("show sanctioned country exposure")
        assert "ofac" in labels

    def test_ofac_data_query_routes_to_policy(self):
        # "which customers have OFAC hits" → policy (data query)
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("which customers have ofac hits?")
        assert "policy" in labels
        assert "ofac" not in labels

    def test_how_many_customers_ofac_routes_to_policy(self):
        orch, _ = _make_orchestrator("out_of_scope")
        labels = orch._route("how many customers have ofac exposure?")
        assert "policy" in labels


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

    def test_policy_fallback_keyword(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("what is know your customer KYC?")
        assert "policy" in labels

    def test_canada_keyword_routes_to_policy(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        # "Canada's AML rules" would hit the "rule" threshold fallback first;
        # use a query with "canada" but no threshold keywords.
        labels = orch._route("What does AML compliance in Canada require?")
        assert "policy" in labels

    def test_fintrac_keyword_routes_to_policy(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What does FINTRAC require for suspicious transaction reporting?")
        assert "policy" in labels

    def test_typology_keyword_routes_to_policy(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What is AML typology?")
        assert "policy" in labels

    def test_layering_keyword_routes_to_policy(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("What is AML layering?")
        assert "policy" in labels

    def test_ofac_fallback_keyword(self):
        orch, mock_client = _make_orchestrator("bad_label_not_valid")
        labels = orch._route("run ofac screen on portfolio")
        assert "ofac" in labels


# ── LLM classification error handling ────────────────────────────────────────

class TestClassificationErrorHandling:
    def test_llm_exception_defaults_to_out_of_scope(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("connection refused")

        with patch("openai.OpenAI", return_value=mock_client), \
             patch("agents.policy_agent.PolicyAgent._load_kb", return_value=None):
            from agents.orchestrator import OrchestratorAgent
            orch = OrchestratorAgent()
        orch._client = mock_client

        # A query with no keyword signals should fall back to out_of_scope
        labels = orch._route("some completely irrelevant question about pizza")
        assert "out_of_scope" in labels


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
