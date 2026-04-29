"""
config.py — Central configuration for ARIA.

Override LLM endpoint via environment variables:
    set OLLAMA_BASE_URL=http://localhost:11434/v1   (default — Ollama)
    set OLLAMA_MODEL=qwen2.5:7b                     (default)

To use a fine-tuned model via vLLM after training:
    set OLLAMA_BASE_URL=http://localhost:8000/v1
    set OLLAMA_MODEL=qwen-framl

Data directory:
    set ARIA_DATA_DIR=./aria_synth          (default — synthetic data)
    set ARIA_DATA_DIR=./my_bank_data        (user's own data)
    set ARIA_DATA_DIR=./docs                (original production data)

    Files are discovered by glob pattern within ARIA_DATA_DIR.
    Column names are normalised automatically via column_mapper.py.
"""

import os
import glob as _glob

# ── LLM endpoint ──────────────────────────────────────────────────────────────
# Honour legacy VLLM_* names too so existing docs still work.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",
                            os.getenv("VLLM_BASE_URL", "http://localhost:11434/v1"))
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",
                            os.getenv("VLLM_MODEL", "qwen2.5:7b"))

# ── LLM generation parameters ────────────────────────────────────────────────
MAX_TOKENS_TOOL   = int(os.getenv("MAX_TOKENS_TOOL",   "2048"))  # threshold / segmentation agents
MAX_TOKENS_POLICY = int(os.getenv("MAX_TOKENS_POLICY", "2048"))  # policy agent (longer KB responses)
MAX_TOOL_CALLS    = int(os.getenv("MAX_TOOL_CALLS",    "6"))      # max agentic loop iterations

# ── Data directory ─────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
ARIA_DATA_DIR = os.getenv("ARIA_DATA_DIR", os.path.join(_HERE, "aria_synth"))


def _find_file(data_dir, *patterns, fallback=None):
    """Return first file in data_dir matching any glob pattern, else fallback path."""
    for pattern in patterns:
        matches = _glob.glob(os.path.join(data_dir, pattern))
        if matches:
            return matches[0]
    return fallback


# ── Data file paths — discovered from ARIA_DATA_DIR, fallback to docs/ ────────
DS_CSV = (
    _find_file(ARIA_DATA_DIR, "*segment*.csv", "*segmentation*.csv")
    or _find_file(os.path.join(_HERE, "docs"), "ds_segmentation_synth.csv")
    or os.path.join(_HERE, "docs", "ds_segmentation_data.csv")
)

SAR_CSV = _find_file(
    ARIA_DATA_DIR,
    "*sar*.csv",
    fallback=os.path.join(_HERE, "docs", "sar_simulation.csv"),
)

# ALERTS_CSV — main dashboard data (customer/account/alert summary rows)
ALERTS_CSV = _find_file(
    ARIA_DATA_DIR,
    "aria_custs_alerts.csv", "*custs*alert*.csv",
    fallback=os.path.join(_HERE, "docs", "custs_accts_txns_alerts.csv"),
)

# RULE_SWEEP_CSV — alert-level data for rule SAR sweep / backtest
RULE_SWEEP_CSV = _find_file(
    ARIA_DATA_DIR,
    "*alert*.csv", "*rule*.csv", "*sweep*.csv",
    fallback=os.path.join(_HERE, "docs", "rule_sweep_data.csv"),
)

# Cluster labels — derived at startup from K-Means if not found
CLUSTER_LABELS_CSV = _find_file(
    ARIA_DATA_DIR,
    "*cluster*label*.csv", "*cluster*.csv",
    fallback=os.path.join(_HERE, "docs", "customer_cluster_labels.csv"),
)

# ── Non-data paths (not user-switchable) ──────────────────────────────────────
SS_FILES_DIR = os.path.join(_HERE, "ss_files")
CHROMA_PATH  = os.path.join(_HERE, "chroma_db")
DOCS_DIR     = os.path.join(_HERE, "docs")
OFAC_DB      = os.path.join(_HERE, "data", "ofac_sdn.db")
CUSTOMERS_CSV = os.path.join(_HERE, "ss_files_anon", "aml_s_customers.csv")

# Legacy alias — kept so any code still importing SYNTH_DATA_DIR doesn't break
SYNTH_DATA_DIR = ARIA_DATA_DIR
