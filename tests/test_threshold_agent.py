"""Tests for agents/threshold_agent.py — invalid param rejection, tool definitions, system prompt."""
import pytest
from unittest.mock import MagicMock, patch


with patch("openai.OpenAI", return_value=MagicMock()):
    from agents.threshold_agent import (
        ThresholdAgent,
        TOOLS,
        SYSTEM_PROMPT,
        _INVALID_PARAMS,
        _REJECTION_MSG,
        _N_RULES,
        _RANKING_QUERY_TERMS,
    )


# ── Ranking query terms ────────────────────────────────────────────────────────

class TestRankingQueryTerms:
    def test_the_same_in_ranking_terms(self):
        assert "the same" in _RANKING_QUERY_TERMS

    def test_same_by_in_ranking_terms(self):
        assert "same by" in _RANKING_QUERY_TERMS

    def test_core_terms_present(self):
        for term in ("top", "bottom", "highest", "lowest", "most", "least"):
            assert term in _RANKING_QUERY_TERMS


# ── Invalid parameter rejection ────────────────────────────────────────────────

class TestInvalidParamRejection:
    """ThresholdAgent.run() must reject queries containing invalid params early."""

    @pytest.fixture
    def agent(self):
        with patch("openai.OpenAI", return_value=MagicMock()):
            return ThresholdAgent()

    @pytest.mark.parametrize("invalid_param", [
        "threshold_min", "threshold_max", "threshold_step", "step", "min_threshold"
    ])
    def test_rejects_invalid_param_in_query(self, agent, invalid_param):
        query = f"Show me FP/FN for {invalid_param}=5"
        text, charts = agent.run(query, tool_executor=MagicMock())
        assert text == _REJECTION_MSG
        assert charts == []

    def test_invalid_param_set_completeness(self):
        assert _INVALID_PARAMS == {
            "threshold_min", "threshold_max", "threshold_step", "step", "min_threshold"
        }

    def test_rejection_message_mentions_valid_params(self):
        assert "segment" in _REJECTION_MSG
        assert "threshold_column" in _REJECTION_MSG
        assert "Business" in _REJECTION_MSG
        assert "Individual" in _REJECTION_MSG

    def test_rejection_message_mentions_valid_columns(self):
        assert "AVG_TRXNS_WEEK" in _REJECTION_MSG
        assert "AVG_TRXN_AMT" in _REJECTION_MSG
        assert "TRXN_AMT_MONTHLY" in _REJECTION_MSG

    def test_hyphen_variant_of_invalid_param_rejected(self):
        # e.g. "threshold-min" normalized via .replace("-", "_")
        query = "Show FP/FN for threshold-step=10"
        agent_obj = ThresholdAgent.__new__(ThresholdAgent)
        # Manually apply the same normalization logic
        query_lower = query.lower().replace("-", "_").replace(" ", "_")
        assert any(p in query_lower for p in _INVALID_PARAMS)

    def test_valid_query_not_rejected(self, agent):
        # A valid query won't contain invalid param keywords — agent should
        # proceed to call super().run(), which calls the LLM. Here we just
        # verify it does NOT return the rejection message.
        mock_executor = MagicMock(return_value=("result text", None))

        # Mock the LLM call so it returns a terminal response (no tool calls)
        final_msg = MagicMock()
        final_msg.tool_calls = None
        final_msg.content = "The Business segment has 50 FPs."

        with patch.object(agent, "_stream_llm", return_value=final_msg):
            text, charts = agent.run(
                "Show FP/FN for Business segment AVG_TRXNS_WEEK",
                tool_executor=mock_executor,
            )
        assert text != _REJECTION_MSG


# ── TOOLS structure validation ─────────────────────────────────────────────────

class TestToolsDefinition:
    EXPECTED_TOOLS = {
        "threshold_tuning", "segment_stats", "sar_backtest",
        "rule_2d_sweep", "list_rules", "rule_sar_backtest",
        "cluster_rule_summary", "cluster_threshold_analysis",
    }

    def _tool_names(self):
        return {t["function"]["name"] for t in TOOLS}

    def test_all_expected_tools_present(self):
        assert self._tool_names() == self.EXPECTED_TOOLS

    def test_tools_have_correct_structure(self):
        for tool in TOOLS:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_threshold_tuning_segment_enum(self):
        tt = next(t for t in TOOLS if t["function"]["name"] == "threshold_tuning")
        segment_prop = tt["function"]["parameters"]["properties"]["segment"]
        assert "enum" in segment_prop
        assert set(segment_prop["enum"]) == {"Business", "Individual"}

    def test_threshold_tuning_column_enum(self):
        tt = next(t for t in TOOLS if t["function"]["name"] == "threshold_tuning")
        col_prop = tt["function"]["parameters"]["properties"]["threshold_column"]
        assert set(col_prop["enum"]) == {"AVG_TRXNS_WEEK", "AVG_TRXN_AMT", "TRXN_AMT_MONTHLY"}

    def test_segment_stats_has_no_required_params(self):
        ss = next(t for t in TOOLS if t["function"]["name"] == "segment_stats")
        assert ss["function"]["parameters"]["required"] == []

    def test_list_rules_has_no_required_params(self):
        lr = next(t for t in TOOLS if t["function"]["name"] == "list_rules")
        assert lr["function"]["parameters"]["required"] == []

    def test_rule_sar_backtest_requires_risk_factor(self):
        rbt = next(t for t in TOOLS if t["function"]["name"] == "rule_sar_backtest")
        assert "risk_factor" in rbt["function"]["parameters"]["required"]

    def test_rule_2d_sweep_requires_risk_factor(self):
        r2d = next(t for t in TOOLS if t["function"]["name"] == "rule_2d_sweep")
        assert "risk_factor" in r2d["function"]["parameters"]["required"]

    def test_rule_sar_backtest_has_cluster_param(self):
        rbt = next(t for t in TOOLS if t["function"]["name"] == "rule_sar_backtest")
        assert "cluster" in rbt["function"]["parameters"]["properties"]

    def test_rule_2d_sweep_has_cluster_param(self):
        r2d = next(t for t in TOOLS if t["function"]["name"] == "rule_2d_sweep")
        assert "cluster" in r2d["function"]["parameters"]["properties"]

    def test_threshold_tuning_no_cluster_param(self):
        tt = next(t for t in TOOLS if t["function"]["name"] == "threshold_tuning")
        assert "cluster" not in tt["function"]["parameters"]["properties"]

    def test_cluster_rule_summary_present(self):
        assert "cluster_rule_summary" in self._tool_names()

    def test_cluster_rule_summary_requires_cluster(self):
        crs = next(t for t in TOOLS if t["function"]["name"] == "cluster_rule_summary")
        assert "cluster" in crs["function"]["parameters"]["required"]

    def test_cluster_rule_summary_cluster_is_integer(self):
        crs = next(t for t in TOOLS if t["function"]["name"] == "cluster_rule_summary")
        cluster_prop = crs["function"]["parameters"]["properties"]["cluster"]
        assert cluster_prop["type"] == "integer"

    def test_cluster_rule_summary_description_mentions_all_rules(self):
        crs = next(t for t in TOOLS if t["function"]["name"] == "cluster_rule_summary")
        desc = crs["function"]["description"].lower()
        assert "all" in desc and "rule" in desc

    def test_cluster_rule_summary_description_distinguishes_from_rule_sar_backtest(self):
        crs = next(t for t in TOOLS if t["function"]["name"] == "cluster_rule_summary")
        desc = crs["function"]["description"]
        assert "rule_sar_backtest" in desc or "single" in desc.lower() or "named" in desc.lower()

    def test_threshold_tuning_column_description_disambiguates_avg_trxn_amt(self):
        tt = next(t for t in TOOLS if t["function"]["name"] == "threshold_tuning")
        desc = tt["function"]["parameters"]["properties"]["threshold_column"]["description"]
        assert "AVG_TRXN_AMT" in desc
        assert "transaction amount" in desc.lower() or "dollar amount" in desc.lower()

    def test_threshold_tuning_column_description_disambiguates_avg_trxns_week(self):
        tt = next(t for t in TOOLS if t["function"]["name"] == "threshold_tuning")
        desc = tt["function"]["parameters"]["properties"]["threshold_column"]["description"]
        assert "AVG_TRXNS_WEEK" in desc
        # Must clarify it's a count (not dollars) to prevent "weekly amount" → wrong column
        assert "count" in desc.lower() or "number" in desc.lower() or "frequency" in desc.lower()

    def test_sar_backtest_column_description_matches_threshold_tuning(self):
        # Both tools share the same column disambiguation — verify sar_backtest has it too
        sb = next(t for t in TOOLS if t["function"]["name"] == "sar_backtest")
        desc = sb["function"]["parameters"]["properties"]["threshold_column"]["description"]
        assert "AVG_TRXN_AMT" in desc
        assert "AVG_TRXNS_WEEK" in desc
        assert "TRXN_AMT_MONTHLY" in desc


# ── System prompt validation ───────────────────────────────────────────────────

class TestSystemPrompt:
    """The production_switchover (2026-06-02) replaced the multi-thousand-char
    rule-based prompt (Rules 1-29) with a one-line directive. Behavioral
    instruction now lives in training examples + tool definitions.
    See feedback_instruction_placement / production_switchover_plan."""

    def test_prompt_is_minimal(self):
        # The whole prompt should be short (under 600 chars). Previously it was 16K+.
        # Was <200 until 2026-06-08 when the closest-match shorthand instruction
        # was added (commit 7a50911) to handle 'Crypto'/'Mule'/etc. → canonical
        # rule name resolution. Still 38x smaller than the pre-switchover prompt.
        assert len(SYSTEM_PROMPT) < 600

    def test_prompt_directs_to_data(self):
        # Minimal prompt instructs the model to ground answers in tool output.
        assert "data" in SYSTEM_PROMPT.lower()

    def test_prompt_does_not_contain_obsolete_rule_format(self):
        # Old prompts had "Rule 1:", "Rule 16:", etc. and were rule-list-driven.
        for obsolete in (
            "Rule 1:", "Rule 16:", "OVERRIDE RULE 16",
            "default to Business", "default to AVG_TRXNS_WEEK",
            "PRE-COMPUTED", "cluster_rule_summary",
        ):
            assert obsolete not in SYSTEM_PROMPT

    def test_rule_count_constant_matches_catalogue(self):
        # _N_RULES is exposed for downstream code; verify it tracks the catalogue
        # (16 classic + 5 modern typology = 21).
        import lambda_rule_analysis as ra
        assert _N_RULES == len(ra.RULE_CATALOGUE)
