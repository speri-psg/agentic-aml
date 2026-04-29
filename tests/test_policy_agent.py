"""Tests for agents/policy_agent.py — citation stripping (pure function, no LLM/ChromaDB required)."""
import pytest
from unittest.mock import patch, MagicMock


def _make_agent():
    """Create a PolicyAgent with ChromaDB loading disabled."""
    with patch("openai.OpenAI", return_value=MagicMock()), \
         patch("agents.policy_agent.PolicyAgent._load_kb", return_value=None):
        from agents.policy_agent import PolicyAgent
        agent = PolicyAgent()
    return agent


# ── Citation stripping — fabricated inline tokens ──────────────────────────────

class TestStripFabricatedCitations:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    # FIN-YYYY-AXXX patterns
    def test_strips_fincen_advisory_number(self, agent):
        text = "According to FIN-2020-A005, banks must file SARs."
        result = agent._strip_fabricated_citations(text, [])
        assert "FIN-2020-A005" not in result
        assert "banks must file SARs" in result

    def test_strips_fincen_advisory_different_year(self, agent):
        text = "See FIN-2014-A008 for guidance."
        result = agent._strip_fabricated_citations(text, [])
        assert "FIN-2014-A008" not in result

    # 31 CFR patterns
    def test_strips_cfr_with_section(self, agent):
        text = "Under 31 CFR 1010.314, customer identification is required."
        result = agent._strip_fabricated_citations(text, [])
        assert "31 CFR 1010.314" not in result

    def test_strips_cfr_part(self, agent):
        text = "31 CFR Part 1020 covers banks."
        result = agent._strip_fabricated_citations(text, [])
        assert "31 CFR Part 1020" not in result

    # U.S.C. patterns
    def test_strips_usc_section(self, agent):
        text = "31 U.S.C. § 5318 requires record-keeping."
        result = agent._strip_fabricated_citations(text, [])
        assert "U.S.C." not in result or "§ 5318" not in result

    def test_strips_standalone_usc(self, agent):
        text = "U.S.C. § 1234 applies here."
        result = agent._strip_fabricated_citations(text, [])
        assert "U.S.C. § 1234" not in result

    # OCC patterns
    def test_strips_occ_bulletin(self, agent):
        text = "OCC 2000-17 provides relevant guidance."
        result = agent._strip_fabricated_citations(text, [])
        assert "OCC 2000-17" not in result

    # BSA patterns
    def test_strips_bsa_code(self, agent):
        text = "BSA-04 covers exemptions."
        result = agent._strip_fabricated_citations(text, [])
        assert "BSA-04" not in result

    # Orphan subsection patterns
    def test_strips_orphan_subsection_full(self, agent):
        text = "Under (a)(2)(A), reporting is required."
        result = agent._strip_fabricated_citations(text, [])
        assert "(a)(2)(A)" not in result

    def test_strips_orphan_subsection_short(self, agent):
        text = "Pursuant to (b)(1), the rule applies."
        result = agent._strip_fabricated_citations(text, [])
        assert "(b)(1)" not in result

    # Source: line filtering
    def test_strips_fabricated_source_line(self, agent):
        text = "Here is information.\nSource: Some Unknown Document\nMore text."
        result = agent._strip_fabricated_citations(text, ["FFIEC BSA-AML Manual"])
        assert "Some Unknown Document" not in result

    def test_keeps_valid_source_line(self, agent):
        text = "Here is information.\nSource: FFIEC BSA-AML Manual\nMore text."
        result = agent._strip_fabricated_citations(text, ["FFIEC BSA-AML Manual"])
        assert "FFIEC BSA-AML Manual" in result

    # Text without citations should pass through unchanged
    def test_clean_text_unchanged(self, agent):
        text = "AML compliance requires customer due diligence and transaction monitoring."
        result = agent._strip_fabricated_citations(text, [])
        assert result == text

    def test_empty_text(self, agent):
        result = agent._strip_fabricated_citations("", [])
        assert result == ""

    def test_multiline_clean_text_preserved(self, agent):
        text = "Line one.\nLine two.\nLine three."
        result = agent._strip_fabricated_citations(text, [])
        assert "Line one." in result
        assert "Line two." in result
        assert "Line three." in result

    # Multiple citations in same text
    def test_multiple_citations_all_stripped(self, agent):
        text = "FIN-2020-A005 and 31 CFR 1010.314 and OCC 2000-17 all apply."
        result = agent._strip_fabricated_citations(text, [])
        assert "FIN-2020-A005" not in result
        assert "OCC 2000-17" not in result

    # Allowed source tokens are preserved
    def test_citation_in_allowed_source_name_preserved(self, agent):
        # If "31 CFR Part 1010" is literally part of an allowed source name,
        # it should be kept
        allowed = ["31 CFR Part 1010 (up to date as of 3-26-2026)"]
        text = "31 CFR Part 1010 covers general provisions."
        result = agent._strip_fabricated_citations(text, allowed)
        # The source name substring matches — token is kept
        assert "31 CFR Part 1010" in result

    # Cleanup of artifacts after stripping
    def test_empty_parens_removed_after_stripping(self, agent):
        text = "Compliance with () is important."
        # Stripping leaves empty parens which get cleaned up
        result = agent._strip_fabricated_citations(text, [])
        assert "()" not in result

    def test_double_spaces_collapsed(self, agent):
        text = "The  rule  applies."
        result = agent._strip_fabricated_citations(text, [])
        assert "  " not in result

    # ── EU citation patterns ───────────────────────────────────────────────────

    def test_strips_celex_identifier(self, agent):
        text = "As per CELEX 32015L0849, customer due diligence is required."
        result = agent._strip_fabricated_citations(text, [])
        assert "32015L0849" not in result

    def test_strips_celex_regulation(self, agent):
        text = "Regulation 32024R1624 sets out CDD obligations."
        result = agent._strip_fabricated_citations(text, [])
        assert "32024R1624" not in result

    def test_strips_oj_reference(self, agent):
        text = "Published in OJ L 141, the directive requires enhanced due diligence."
        result = agent._strip_fabricated_citations(text, [])
        assert "OJ L 141" not in result

    def test_strips_oj_date_reference(self, agent):
        text = "See OJ L 141, 5.6.2015 for the full text."
        result = agent._strip_fabricated_citations(text, [])
        assert "OJ L 141, 5.6.2015" not in result

    def test_strips_recital_number(self, agent):
        text = "Recital 22 of AMLD5 clarifies the scope of virtual assets."
        result = agent._strip_fabricated_citations(text, [])
        assert "Recital 22" not in result

    def test_strips_recitals_range(self, agent):
        text = "Recitals 10-12 provide context for beneficial ownership."
        result = agent._strip_fabricated_citations(text, [])
        assert "Recitals 10" not in result

    def test_does_not_strip_article_text_in_allowed_source(self, agent):
        allowed = ["EU_4th_AMLD_2015_849.pdf"]
        text = "EU_4th_AMLD_2015_849.pdf covers customer due diligence."
        result = agent._strip_fabricated_citations(text, allowed)
        assert "EU_4th_AMLD_2015_849.pdf" in result

    # ── UN citation patterns ───────────────────────────────────────────────────

    def test_strips_non_1373_sres(self, agent):
        text = "S/RES/1267 established the Al-Qaeda sanctions regime."
        result = agent._strip_fabricated_citations(text, [])
        assert "S/RES/1267" not in result

    def test_keeps_resolution_1373_sres(self, agent):
        # 1373 is in the KB — should not be stripped
        text = "S/RES/1373 requires states to criminalise terrorist financing."
        result = agent._strip_fabricated_citations(text, [])
        assert "S/RES/1373" in result

    def test_strips_non_1373_resolution_word(self, agent):
        text = "Resolution 1267 targeted Al-Qaeda finances."
        result = agent._strip_fabricated_citations(text, [])
        assert "Resolution 1267" not in result

    def test_keeps_resolution_1373_word(self, agent):
        text = "Resolution 1373 is the primary counter-terrorism financing instrument."
        result = agent._strip_fabricated_citations(text, [])
        assert "Resolution 1373" in result


# ── Retrieval logic ────────────────────────────────────────────────────────────

class TestPolicyAgentRetrieval:

    def test_retrieve_no_collection_returns_empty(self):
        with patch("openai.OpenAI", return_value=MagicMock()), \
             patch("agents.policy_agent.PolicyAgent._load_kb", return_value=None), \
             patch("agents.policy_agent._upload_kb.retrieve", return_value=("", [])):
            from agents.policy_agent import PolicyAgent
            agent = PolicyAgent()
            agent.collection = None  # ensure no KB
            ctx, sources = agent.retrieve("What is AML?")
        assert ctx == ""
        assert sources == []

    def test_retrieve_merges_upload_context(self):
        upload_ctx = "Uploaded document content about AML."
        upload_sources = ["my_document.pdf"]
        with patch("openai.OpenAI", return_value=MagicMock()), \
             patch("agents.policy_agent.PolicyAgent._load_kb", return_value=None), \
             patch("agents.policy_agent._upload_kb.retrieve", return_value=(upload_ctx, upload_sources)):
            from agents.policy_agent import PolicyAgent
            agent = PolicyAgent()
            agent.collection = None
            ctx, sources = agent.retrieve("What is AML?")
        assert upload_ctx in ctx
        assert "my_document.pdf" in sources

    def test_retrieve_deduplicates_sources(self):
        """Sources from regulatory KB and uploads that overlap shouldn't appear twice."""
        upload_ctx = "Some context."
        upload_sources = ["shared_source.pdf"]
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc chunk"]],
            "metadatas": [[{"source": "shared_source.pdf"}]],
        }
        with patch("openai.OpenAI", return_value=MagicMock()), \
             patch("agents.policy_agent.PolicyAgent._load_kb", return_value=None), \
             patch("agents.policy_agent._upload_kb.retrieve", return_value=(upload_ctx, upload_sources)):
            from agents.policy_agent import PolicyAgent
            agent = PolicyAgent()
            agent.collection = mock_collection
            _, sources = agent.retrieve("What is AML?")
        # "shared_source.pdf" should appear only once
        assert sources.count("shared_source.pdf") == 1
