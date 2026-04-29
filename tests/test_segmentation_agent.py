"""Tests for agents/segmentation_agent.py — tool definitions and system prompt."""
import pytest
from unittest.mock import MagicMock, patch


with patch("openai.OpenAI", return_value=MagicMock()):
    from agents.segmentation_agent import (
        SegmentationAgent,
        TOOLS,
        SYSTEM_PROMPT,
    )


# ── TOOLS structure validation ─────────────────────────────────────────────────

class TestToolsDefinition:
    EXPECTED_TOOLS = {
        "cluster_analysis", "alerts_distribution",
        "prepare_segmentation_data", "ds_cluster_analysis",
    }

    def _tool_names(self):
        return {t["function"]["name"] for t in TOOLS}

    def _get_tool(self, name):
        return next(t for t in TOOLS if t["function"]["name"] == name)

    def test_all_expected_tools_present(self):
        assert self._tool_names() == self.EXPECTED_TOOLS

    def test_tools_have_correct_structure(self):
        for tool in TOOLS:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    # ── cluster_analysis ──────────────────────────────────────────────────────

    def test_cluster_analysis_has_customer_type_enum(self):
        ca = self._get_tool("cluster_analysis")
        ct = ca["function"]["parameters"]["properties"]["customer_type"]
        assert set(ct["enum"]) == {"Business", "Individual", "All"}

    def test_cluster_analysis_has_n_clusters_param(self):
        ca = self._get_tool("cluster_analysis")
        assert "n_clusters" in ca["function"]["parameters"]["properties"]

    def test_cluster_analysis_n_clusters_is_integer_type(self):
        ca = self._get_tool("cluster_analysis")
        n = ca["function"]["parameters"]["properties"]["n_clusters"]
        assert n["type"] == "integer"

    def test_cluster_analysis_only_customer_type_required(self):
        ca = self._get_tool("cluster_analysis")
        assert ca["function"]["parameters"]["required"] == ["customer_type"]

    # ── ds_cluster_analysis ───────────────────────────────────────────────────

    def test_ds_cluster_analysis_has_customer_type_enum(self):
        dca = self._get_tool("ds_cluster_analysis")
        ct = dca["function"]["parameters"]["properties"]["customer_type"]
        assert set(ct["enum"]) == {"Business", "Individual", "All"}

    def test_ds_cluster_analysis_has_n_clusters_param(self):
        dca = self._get_tool("ds_cluster_analysis")
        assert "n_clusters" in dca["function"]["parameters"]["properties"]

    def test_ds_cluster_analysis_n_clusters_is_integer_type(self):
        dca = self._get_tool("ds_cluster_analysis")
        n = dca["function"]["parameters"]["properties"]["n_clusters"]
        assert n["type"] == "integer"

    def test_ds_cluster_analysis_only_customer_type_required(self):
        dca = self._get_tool("ds_cluster_analysis")
        assert dca["function"]["parameters"]["required"] == ["customer_type"]

    def test_ds_cluster_analysis_n_clusters_description_mentions_user_request(self):
        # Must instruct model to pass exactly what the user requests (not always 4)
        dca = self._get_tool("ds_cluster_analysis")
        desc = dca["function"]["parameters"]["properties"]["n_clusters"]["description"]
        assert "user" in desc.lower() or "request" in desc.lower() or "exact" in desc.lower()

    # ── alerts_distribution ───────────────────────────────────────────────────

    def test_alerts_distribution_has_no_required_params(self):
        ad = self._get_tool("alerts_distribution")
        assert ad["function"]["parameters"]["required"] == []

    # ── prepare_segmentation_data ─────────────────────────────────────────────

    def test_prepare_segmentation_data_has_no_required_params(self):
        pd_ = self._get_tool("prepare_segmentation_data")
        assert pd_["function"]["parameters"]["required"] == []


# ── System prompt validation ───────────────────────────────────────────────────

class TestSystemPrompt:
    def test_default_customer_type_is_all(self):
        assert "default to All" in SYSTEM_PROMPT or "default is All" in SYSTEM_PROMPT \
               or ("not specify" in SYSTEM_PROMPT and "All" in SYSTEM_PROMPT)

    def test_n_clusters_default_mentioned(self):
        assert "4" in SYSTEM_PROMPT

    def test_display_clusters_directive_described(self):
        assert "DISPLAY_CLUSTERS" in SYSTEM_PROMPT

    def test_display_clusters_format_is_correct(self):
        assert "DISPLAY_CLUSTERS: N" in SYSTEM_PROMPT

    def test_system_prompt_has_english_only_rule(self):
        assert "English" in SYSTEM_PROMPT

    def test_do_not_invent_numbers_rule_present(self):
        assert "invent" in SYSTEM_PROMPT or "not in the tool result" in SYSTEM_PROMPT \
               or "only" in SYSTEM_PROMPT.lower()

    def test_ds_cluster_analysis_tool_name_in_prompt(self):
        # System prompt must spell out the exact tool name to prevent hallucination
        assert "ds_cluster_analysis" in SYSTEM_PROMPT

    def test_n_clusters_user_override_rule(self):
        # Rule 8 must instruct the model to pass the user's requested count exactly
        assert "n_clusters" in SYSTEM_PROMPT

    def test_display_clusters_only_when_user_asks_to_filter(self):
        # Directive must only fire when user asks to filter — rule should say so
        assert "does NOT ask to filter" in SYSTEM_PROMPT \
               or "do NOT include" in SYSTEM_PROMPT \
               or "not ask" in SYSTEM_PROMPT.lower()


# ── SegmentationAgent instantiation ───────────────────────────────────────────

class TestSegmentationAgentInit:
    @pytest.fixture
    def agent(self):
        with patch("openai.OpenAI", return_value=MagicMock()):
            return SegmentationAgent()

    def test_agent_name_is_segmentation(self, agent):
        assert agent.name == "segmentation"

    def test_agent_has_correct_tools(self, agent):
        names = {t["function"]["name"] for t in agent.tools}
        assert "ds_cluster_analysis" in names
        assert "cluster_analysis" in names

    def test_valid_query_proceeds_to_llm(self, agent):
        final_msg = MagicMock()
        final_msg.tool_calls = None
        final_msg.content = "Cluster 1 has the highest volume."

        with patch.object(agent, "_stream_llm", return_value=final_msg):
            text, charts = agent.run(
                "Cluster Business customers into 3 groups",
                tool_executor=MagicMock(return_value=("result", None)),
            )
        assert isinstance(text, str)
        assert text != ""
