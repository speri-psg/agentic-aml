"""Tests for config.py — file discovery and environment variable overrides."""
import os
import sys
import importlib
import tempfile
import pytest
from unittest.mock import patch


# ── _find_file ─────────────────────────────────────────────────────────────────

class TestFindFile:
    """Test the glob-based file discovery helper."""

    @pytest.fixture(autouse=True)
    def _import_find_file(self):
        import config
        self._find_file = config._find_file

    def test_returns_first_matching_file(self, tmp_path):
        target = tmp_path / "test_segment_data.csv"
        target.write_text("a,b,c")
        result = self._find_file(str(tmp_path), "*segment*.csv")
        assert result is not None
        assert "test_segment_data.csv" in result

    def test_returns_none_when_no_match_and_no_fallback(self, tmp_path):
        result = self._find_file(str(tmp_path), "*nonexistent*.csv")
        assert result is None

    def test_returns_fallback_when_no_match(self, tmp_path):
        fallback = str(tmp_path / "fallback.csv")
        result = self._find_file(str(tmp_path), "*nonexistent*.csv", fallback=fallback)
        assert result == fallback

    def test_first_matching_pattern_wins(self, tmp_path):
        (tmp_path / "sar_data.csv").write_text("x")
        (tmp_path / "segment_data.csv").write_text("y")
        # Pattern "*sar*" should match first
        result = self._find_file(str(tmp_path), "*sar*.csv", "*segment*.csv")
        assert "sar_data.csv" in result

    def test_falls_through_to_second_pattern_when_first_misses(self, tmp_path):
        (tmp_path / "segment_data.csv").write_text("y")
        result = self._find_file(str(tmp_path), "*sar*.csv", "*segment*.csv")
        assert "segment_data.csv" in result

    def test_empty_directory_returns_none(self, tmp_path):
        result = self._find_file(str(tmp_path), "*.csv")
        assert result is None

    def test_nonexistent_directory_returns_fallback(self, tmp_path):
        fallback = "/some/fallback.csv"
        result = self._find_file(str(tmp_path / "nonexistent_dir"), "*.csv", fallback=fallback)
        assert result == fallback


# ── Environment variable overrides ────────────────────────────────────────────

class TestEnvironmentOverrides:
    """Test that config values honour environment variable overrides."""

    def _reload_config(self):
        """Reload config module to pick up env var changes."""
        import config
        importlib.reload(config)
        return config

    def test_default_ollama_url(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove both env vars if set
            env = {k: v for k, v in os.environ.items()
                   if k not in ("OLLAMA_BASE_URL", "VLLM_BASE_URL")}
            with patch.dict(os.environ, env, clear=True):
                cfg = self._reload_config()
                assert cfg.OLLAMA_BASE_URL == "http://localhost:11434/v1"

    def test_ollama_base_url_override(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://myserver:8000/v1"}):
            cfg = self._reload_config()
            assert cfg.OLLAMA_BASE_URL == "http://myserver:8000/v1"

    def test_vllm_base_url_fallback(self):
        env = {k: v for k, v in os.environ.items() if k != "OLLAMA_BASE_URL"}
        env["VLLM_BASE_URL"] = "http://vllm:8000/v1"
        with patch.dict(os.environ, env, clear=True):
            cfg = self._reload_config()
            assert cfg.OLLAMA_BASE_URL == "http://vllm:8000/v1"

    def test_ollama_model_override(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "my-custom-model"}):
            cfg = self._reload_config()
            assert cfg.OLLAMA_MODEL == "my-custom-model"

    def test_max_tokens_tool_override(self):
        with patch.dict(os.environ, {"MAX_TOKENS_TOOL": "4096"}):
            cfg = self._reload_config()
            assert cfg.MAX_TOKENS_TOOL == 4096

    def test_max_tokens_policy_override(self):
        with patch.dict(os.environ, {"MAX_TOKENS_POLICY": "8192"}):
            cfg = self._reload_config()
            assert cfg.MAX_TOKENS_POLICY == 8192

    def test_max_tool_calls_override(self):
        with patch.dict(os.environ, {"MAX_TOOL_CALLS": "10"}):
            cfg = self._reload_config()
            assert cfg.MAX_TOOL_CALLS == 10

    def test_max_tokens_are_integers(self):
        import config
        assert isinstance(config.MAX_TOKENS_TOOL, int)
        assert isinstance(config.MAX_TOKENS_POLICY, int)
        assert isinstance(config.MAX_TOOL_CALLS, int)

    def test_aria_data_dir_override(self, tmp_path):
        with patch.dict(os.environ, {"ARIA_DATA_DIR": str(tmp_path)}):
            cfg = self._reload_config()
            assert cfg.ARIA_DATA_DIR == str(tmp_path)

    def test_synth_data_dir_legacy_alias(self):
        import config
        # SYNTH_DATA_DIR must equal ARIA_DATA_DIR (legacy alias)
        assert config.SYNTH_DATA_DIR == config.ARIA_DATA_DIR

    def test_legacy_vllm_model_alias(self):
        with patch.dict(os.environ, {"VLLM_MODEL": "legacy-model"}):
            env = {k: v for k, v in os.environ.items() if k != "OLLAMA_MODEL"}
            env["VLLM_MODEL"] = "legacy-model"
            with patch.dict(os.environ, env, clear=True):
                cfg = self._reload_config()
                assert cfg.OLLAMA_MODEL == "legacy-model"


# ── Static path correctness ───────────────────────────────────────────────────

class TestStaticPaths:
    def test_chroma_path_ends_with_chroma_db(self):
        import config
        assert config.CHROMA_PATH.endswith("chroma_db")

    def test_docs_dir_ends_with_docs(self):
        import config
        assert config.DOCS_DIR.endswith("docs")

    def test_ss_files_dir_ends_with_ss_files(self):
        import config
        assert config.SS_FILES_DIR.endswith("ss_files")

    def test_ofac_db_ends_with_sdn_db(self):
        import config
        assert "ofac_sdn.db" in config.OFAC_DB
