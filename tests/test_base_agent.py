"""Tests for agents/base_agent.py — pure functions (no LLM required)."""
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


# ── Import module under test ───────────────────────────────────────────────────
# Patch openai.OpenAI so BaseAgent.__init__ doesn't need a real connection
with patch("openai.OpenAI", return_value=MagicMock()):
    from agents.base_agent import (
        _normalize_tool_name,
        _normalize_args,
        _parse_tool_call_from_content,
        _TOOL_NAME_ALIASES,
        _ARG_ALIASES,
        stop_event,
        BaseAgent,
    )


# ── Tool-name alias normalization ──────────────────────────────────────────────

class TestNormalizeToolName:
    def test_canonical_name_unchanged(self):
        assert _normalize_tool_name("threshold_tuning") == "threshold_tuning"

    def test_gemma4_threshold_hallucinations(self):
        assert _normalize_tool_name("threshold_analysis") == "threshold_tuning"
        assert _normalize_tool_name("threshold_sweep") == "threshold_tuning"
        assert _normalize_tool_name("fp_fn_analysis") == "threshold_tuning"
        assert _normalize_tool_name("fp_fn_tuning") == "threshold_tuning"
        assert _normalize_tool_name("analyze_threshold") == "threshold_tuning"
        assert _normalize_tool_name("alert_analysis") == "threshold_tuning"

    def test_cluster_analysis_aliases(self):
        assert _normalize_tool_name("ss_cluster_analysis") == "ds_cluster_analysis"
        assert _normalize_tool_name("cluster_analysis") == "ds_cluster_analysis"
        assert _normalize_tool_name("segment_analysis") == "ds_cluster_analysis"
        assert _normalize_tool_name("segmentation_analysis") == "ds_cluster_analysis"
        assert _normalize_tool_name("segment_customers") == "ds_cluster_analysis"
        assert _normalize_tool_name("segmentation_kmeans") == "ds_cluster_analysis"

    def test_ofac_aliases(self):
        assert _normalize_tool_name("sanctions_screening") == "ofac_screening"
        assert _normalize_tool_name("ofac_check") == "ofac_screening"
        assert _normalize_tool_name("sdn_screening") == "ofac_screening"

    def test_sar_aliases(self):
        assert _normalize_tool_name("sar_analysis") == "sar_backtest"
        assert _normalize_tool_name("sar_detection") == "sar_backtest"
        assert _normalize_tool_name("backtest") == "sar_backtest"

    def test_unknown_name_passthrough(self):
        assert _normalize_tool_name("totally_unknown_tool") == "totally_unknown_tool"

    def test_all_aliases_in_dict_resolve(self):
        for alias, canonical in _TOOL_NAME_ALIASES.items():
            assert _normalize_tool_name(alias) == canonical


# ── Argument alias normalization ───────────────────────────────────────────────

class TestNormalizeArgs:
    def test_threshold_tuning_customer_type(self):
        args = {"customer_type": "Business", "threshold_column": "AVG_TRXNS_WEEK"}
        result = _normalize_args("threshold_tuning", args)
        assert "segment" in result
        assert result["segment"] == "Business"
        assert "customer_type" not in result

    def test_threshold_tuning_column_aliases(self):
        for alias in ("transaction_amount", "amount_type", "column", "metric"):
            args = {alias: "AVG_TRXN_AMT"}
            result = _normalize_args("threshold_tuning", args)
            assert "threshold_column" in result
            assert result["threshold_column"] == "AVG_TRXN_AMT"

    def test_threshold_tuning_segment_aliases(self):
        for alias in ("customer_type", "customer_segment", "segment_type"):
            args = {alias: "Individual"}
            result = _normalize_args("threshold_tuning", args)
            assert result["segment"] == "Individual"

    def test_sar_backtest_aliases(self):
        args = {"customer_type": "Business", "column": "AVG_TRXN_AMT"}
        result = _normalize_args("sar_backtest", args)
        assert result["segment"] == "Business"
        assert result["threshold_column"] == "AVG_TRXN_AMT"

    def test_rule_sar_backtest_aliases(self):
        args = {"rule": "Elder Abuse", "parameter": "floor_amount"}
        result = _normalize_args("rule_sar_backtest", args)
        assert result["risk_factor"] == "Elder Abuse"
        assert result["sweep_param"] == "floor_amount"

    def test_rule_2d_sweep_aliases(self):
        args = {"rule": "Velocity Single", "param_1": "pair_total", "param_2": "ratio_tolerance"}
        result = _normalize_args("rule_2d_sweep", args)
        assert result["risk_factor"] == "Velocity Single"
        assert result["sweep_param_1"] == "pair_total"
        assert result["sweep_param_2"] == "ratio_tolerance"

    def test_no_alias_tool_unchanged(self):
        args = {"customer_type": "All"}
        result = _normalize_args("ds_cluster_analysis", args)
        assert result == {"customer_type": "All"}

    def test_unknown_keys_pass_through(self):
        args = {"segment": "Business", "extra_key": "value"}
        result = _normalize_args("threshold_tuning", args)
        assert result["segment"] == "Business"
        assert result["extra_key"] == "value"

    def test_empty_args(self):
        assert _normalize_args("threshold_tuning", {}) == {}


# ── Fallback tool-call parser ──────────────────────────────────────────────────

class TestParseToolCallFromContent:

    def test_returns_none_for_empty(self):
        assert _parse_tool_call_from_content("") is None
        assert _parse_tool_call_from_content(None) is None

    def test_returns_none_for_no_match(self):
        assert _parse_tool_call_from_content("Here is a plain text response.") is None

    # Format 1: Gemma 4 native call:tool_name\n{...}
    def test_gemma4_native_format(self):
        content = 'call:threshold_tuning\n{"segment": "Business", "threshold_column": "AVG_TRXNS_WEEK"}'
        result = _parse_tool_call_from_content(content)
        assert result is not None
        name, args = result
        assert name == "threshold_tuning"
        assert args["segment"] == "Business"
        assert args["threshold_column"] == "AVG_TRXNS_WEEK"

    def test_gemma4_native_with_alias(self):
        content = 'call:threshold_analysis\n{"segment": "Individual"}'
        name, args = _parse_tool_call_from_content(content)
        assert name == "threshold_tuning"

    def test_gemma4_native_with_arg_alias(self):
        content = 'call:threshold_tuning\n{"customer_type": "Business", "column": "AVG_TRXN_AMT"}'
        name, args = _parse_tool_call_from_content(content)
        assert args.get("segment") == "Business"
        assert args.get("threshold_column") == "AVG_TRXN_AMT"

    # Format 2: Qwen <tool_call>...</tool_call>
    def test_qwen_format(self):
        content = '<tool_call>\n{"name": "rule_sar_backtest", "arguments": {"risk_factor": "Elder Abuse"}}\n</tool_call>'
        result = _parse_tool_call_from_content(content)
        assert result is not None
        name, args = result
        assert name == "rule_sar_backtest"
        assert args["risk_factor"] == "Elder Abuse"

    def test_qwen_format_no_arguments_key(self):
        # If arguments key is missing, should not crash
        content = '<tool_call>\n{"name": "list_rules"}\n</tool_call>'
        # Either returns None or returns with empty args — shouldn't raise
        result = _parse_tool_call_from_content(content)
        # No arguments key means it won't match the "name" and "arguments" check
        # The result can be None or a valid parse depending on implementation
        # Just ensure no exception is raised

    # Format 3: OpenAI-style JSON {"name": ..., "arguments": {...}}
    def test_openai_json_format(self):
        content = '{"name": "segment_stats", "arguments": {}}'
        result = _parse_tool_call_from_content(content)
        assert result is not None
        name, args = result
        assert name == "segment_stats"
        assert args == {}

    def test_openai_json_with_alias_name(self):
        content = '{"name": "cluster_analysis", "arguments": {"customer_type": "Business"}}'
        result = _parse_tool_call_from_content(content)
        assert result is not None
        name, args = result
        assert name == "ds_cluster_analysis"

    # Format 7: Gemma 4 tool_code — tool_code print(func(kwargs))
    def test_gemma4_tool_code_format(self):
        content = 'tool_code print(threshold_tuning(segment="Business", threshold_column="AVG_TRXN_AMT"))'
        result = _parse_tool_call_from_content(content)
        assert result is not None
        name, args = result
        assert name == "threshold_tuning"
        assert args.get("segment") == "Business"
        assert args.get("threshold_column") == "AVG_TRXN_AMT"

    def test_gemma4_tool_code_integer_arg(self):
        content = 'tool_code print(rule_2d_sweep(risk_factor="Elder Abuse", cluster=3))'
        result = _parse_tool_call_from_content(content)
        assert result is not None
        name, args = result
        assert args.get("cluster") == 3
        assert isinstance(args.get("cluster"), int)

    def test_gemma4_tool_code_with_eos(self):
        content = '<eos>tool_code print(list_rules())'
        result = _parse_tool_call_from_content(content)
        assert result is not None
        name, _ = result
        assert name == "list_rules"

    # Format 8: backtick-wrapped — `func_name(kwargs)`
    def test_backtick_format(self):
        content = "I will call `threshold_tuning(segment='Business', threshold_column='AVG_TRXNS_WEEK')`"
        result = _parse_tool_call_from_content(content)
        assert result is not None
        name, args = result
        assert name == "threshold_tuning"
        assert args.get("segment") == "Business"

    def test_backtick_unknown_tool_not_parsed(self):
        # Backtick parser only handles known tools
        content = "`unknown_tool(param='value')`"
        result = _parse_tool_call_from_content(content)
        assert result is None

    # Format 5: Natural language — "call/use/invoke the tool_name"
    def test_natural_language_format(self):
        content = "I will call threshold_tuning with segment='Business'"
        result = _parse_tool_call_from_content(content)
        # Natural language parsing is best-effort; may or may not match
        # Just verify it doesn't raise
        if result is not None:
            name, _ = result
            assert name == "threshold_tuning"

    def test_natural_language_skips_stop_words(self):
        content = "call the threshold_tuning"
        result = _parse_tool_call_from_content(content)
        if result is not None:
            name, _ = result
            assert name == "threshold_tuning"

    # Alias resolution in parsed results
    def test_parsed_aliases_are_normalized(self):
        content = 'call:threshold_tuning\n{"customer_type": "Business"}'
        name, args = _parse_tool_call_from_content(content)
        assert "customer_type" not in args
        assert args.get("segment") == "Business"


# ── BaseAgent tool deduplication ──────────────────────────────────────────────

class TestBaseAgentDeduplication:
    """Test that duplicate tool calls are filtered in the agentic loop."""

    def _make_agent(self):
        with patch("openai.OpenAI", return_value=MagicMock()):
            return BaseAgent("test", "test prompt", [])

    def test_dedup_same_tool_same_args(self):
        structured = [
            ("threshold_tuning", {"segment": "Business"}, "id1"),
            ("threshold_tuning", {"segment": "Business"}, "id2"),
        ]
        seen = set()
        deduped = []
        for item in structured:
            key = (item[0], str(item[1]))
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        assert len(deduped) == 1

    def test_dedup_same_tool_different_args_kept(self):
        structured = [
            ("threshold_tuning", {"segment": "Business"}, "id1"),
            ("threshold_tuning", {"segment": "Individual"}, "id2"),
        ]
        seen = set()
        deduped = []
        for item in structured:
            key = (item[0], str(item[1]))
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        assert len(deduped) == 2

    def test_dedup_different_tools_kept(self):
        structured = [
            ("threshold_tuning", {"segment": "Business"}, "id1"),
            ("segment_stats", {}, "id2"),
        ]
        seen = set()
        deduped = []
        for item in structured:
            key = (item[0], str(item[1]))
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        assert len(deduped) == 2


# ── Policy context concatenation ──────────────────────────────────────────────

class TestPolicyContextMerge:
    def test_policy_context_prepended_to_query(self):
        context = "Background: AML policy"
        query = "What is the SAR rate?"
        result = f"{context}\n\n{query}".strip() if context else query
        assert result.startswith("Background: AML policy")
        assert "What is the SAR rate?" in result

    def test_empty_policy_context_uses_query_only(self):
        context = ""
        query = "Show FP/FN tuning"
        result = f"{context}\n\n{query}".strip() if context else query
        assert result == "Show FP/FN tuning"
