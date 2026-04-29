# ARIA — Changes Log
**Project:** `framlagents_ftLLM`
**Date:** 2026-03-23
**Objective:** Build a self-contained FRAML chat interface using Qwen2.5 via Ollama with threshold tuning and dynamic segmentation

---

## 1. Project Made Self-Contained (Option B)

Previously `framlagents_ftLLM` depended on the sibling `framlagents/` folder for all data and analytics code. All dependencies are now local.

### Files Copied In
| Source | Destination | Purpose |
|--------|-------------|---------|
| `framlagents/docs/custs_accts_txns_alerts.csv` | `docs/` | 44,617-row alerts dataset |
| `framlagents/docs/ss_segmentation_data.csv` | `docs/` | Pre-processed segmentation data (1,090,815 rows) |
| `framlagents/docs/AML Static Rule Tuning...pdf` | `docs/` | Policy PDF for ChromaDB RAG |
| `framlagents/ss_files/aml_s_customers.csv` | `ss_files/` | Raw customer demographics |
| `framlagents/ss_files/aml_s_accounts.csv` | `ss_files/` | Raw account features |
| `framlagents/ss_files/aml_s_account_relationship.csv` | `ss_files/` | Customer-account mapping |
| `framlagents/ss_files/aml_s_transactions.csv` | `ss_files/` | Raw transaction records |
| `framlagents/ingest.py` | `./ ` | ChromaDB ingestion script |
| `framlagents/lambda_ss_performance.py` | `./` | Analytics engine (threshold tuning, clustering, treemaps) |
| `framlagents/ss_data_prep.py` | `./` | Data preparation pipeline (joins 4 CSVs) |

---

## 2. Python Environment Upgraded to 3.12

- Old venv (Python 3.10.2) deleted
- New venv created with `py -3.12` (Python 3.12.7)
- All dependencies reinstalled from `requirements.txt`
- `pypdf` added to `requirements.txt` (required by `ingest.py`)

---

## 3. `config.py` Created (New File)

**`config.py`** — single source of truth for all settings:

```python
OLLAMA_BASE_URL  # default: http://localhost:11434/v1 (override with env var)
OLLAMA_MODEL     # default: qwen2.5:7b (override with env var)
ALERTS_CSV       # local: docs/custs_accts_txns_alerts.csv
SS_CSV           # local: docs/ss_segmentation_data.csv
SS_FILES_DIR     # local: ss_files/
CHROMA_PATH      # local: chroma_db/
DOCS_DIR         # local: docs/
```

Supports both `OLLAMA_*` and legacy `VLLM_*` env var names.

---

## 4. `agents/base_agent.py` Updated

- Imports `OLLAMA_BASE_URL` and `OLLAMA_MODEL` from `config.py` instead of inline `os.getenv`
- Legacy aliases `VLLM_BASE_URL` / `MODEL_NAME` kept for backward compatibility

---

## 5. `agents/policy_agent.py` Fixed

- `CHROMA_PATH` now imported from `config.py` → points to local `chroma_db/`
- Fixed `api_key="EMPTY"` → `api_key="ollama"` (Ollama requires this exact value)
- Imports `OLLAMA_BASE_URL` / `OLLAMA_MODEL` from `config.py`

---

## 6. System Prompts Strengthened for Qwen2.5 Reliability

### `agents/threshold_agent.py`
- Added explicit rules: ALWAYS call tools, never answer from memory
- Exact enum values listed (`AVG_TRXNS_WEEK`, `AVG_TRXN_AMT`, `TRXN_AMT_MONTHLY`)
- Defaults specified for missing parameters (segment → Business, column → AVG_TRXNS_WEEK)
- **Key addition:** Model instructed to copy the PRE-COMPUTED ANALYSIS verbatim and add only ONE AML insight sentence

### `agents/segmentation_agent.py`
- Added explicit rules: ALWAYS call tools, prefer `ss_cluster_analysis` over `cluster_analysis`
- Exact enum values for `customer_type` (`Business`, `Individual`, `All`)
- Explicit: do NOT call multiple tools for one request

---

## 7. `lambda_ss_performance.py` — AWS Import Fix

The file had top-level `import boto3` and `import botocore` which caused `ModuleNotFoundError` since these are AWS packages not in the local venv. Fixed with conditional imports:

```python
try:
    import boto3
except ImportError:
    boto3 = None  # only needed for show_ss_performance() (legacy AWS Lambda)

try:
    import botocore
except ImportError:
    botocore = None
```

---

## 8. `application.py` — Full Rewrite

### UI Changes
**Before:** Single full-width `ChatComponent` with no sidebar.

**After:** Two-panel layout:
- **Left sidebar (3/12):** App title, model/endpoint badge, dataset summary badges (total accounts, business, individual, alert count, FP count), 6 clickable suggested prompt buttons
- **Right panel (9/12):** Full-height `ChatComponent`
- **Welcome message** at startup with dataset stats and usage instructions
- **Suggested prompt buttons** wire to chat via `dcc.Store` with timestamp (clicking same button twice still fires)

### Suggested Prompts
1. Show FP/FN threshold tuning for Business customers — weekly transaction count
2. Show FP/FN threshold tuning for Individual customers — monthly transaction amount
3. Cluster all customers into behavioral segments and show the treemap
4. Cluster Business customers into 4 segments
5. Show alerts and false positive distribution across segments
6. What does AML policy say about structuring detection thresholds?

### Data Loading
- Removed `sys.path` manipulation for sibling folder — now uses local paths via `config.py`
- Both datasets loaded at startup with printed summary

---

## 9. `compute_threshold_stats()` — Factual Accuracy Fix

**Problem:** Qwen2.5-7b was hallucinating threshold values and getting the FP/FN direction wrong (e.g., claiming "both FP and FN decrease as threshold rises" which is incorrect — FN always increases as threshold rises).

**Root cause:** The model was generating interpretations from pattern-matching rather than reading the actual data values.

**Fix:** Python now generates the factual interpretation deterministically. The tool result contains a `PRE-COMPUTED ANALYSIS` section with exact, accurate statements:

```
At the lowest threshold (0), there are 1486 false positives.
False positives decrease as the threshold rises.
False negatives are zero for all thresholds from 0 up to and including 3.
False negatives first become non-zero at threshold 4 (FN=3).
False negatives increase as the threshold continues to rise...
The crossover point is at threshold 45 (FP=12, FN=11).
The optimal zone spans threshold 38 to 52.
```

The model is instructed to copy this verbatim and only add one AML insight sentence — eliminating hallucination of factual numbers entirely.

**Key facts computed:**
- FP at lowest threshold (baseline)
- Threshold where FN first becomes non-zero
- Threshold range where FN = 0
- Threshold where FP first reaches zero
- Crossover point (min |FP - FN|)
- Optimal zone (both FP and FN < 20% of their respective maximums)

---

## 10. ChromaDB Knowledge Base Built

```bash
python ingest.py
# → 10 chunks from AML policy PDF stored in chroma_db/
```

---

## 11. Ollama Setup

- Ollama installed on local machine
- Model pulled: `qwen2.5:7b` (Q4 quantized, ~4.7GB, stored in `~/.ollama/models/`)
- Running on CPU (NVIDIA 940MX = 2GB VRAM, insufficient for 7B model)
- Response time on CPU: ~30-120 seconds per response
- For faster inference: deploy on vast.ai GPU and set `OLLAMA_BASE_URL=http://<ip>:11434/v1`

---

## Running the App

```bash
# First time only — build ChromaDB
python ingest.py

# Start the app
.venv/Scripts/python.exe application.py
# → http://127.0.0.1:5000
```

---

## Known Issues / Future Work

| Issue | Status | Recommendation |
|-------|--------|---------------|
| CPU inference slow (~1-2 min/response) | Open | Deploy on vast.ai RTX 3090 for 3-5 sec response |
| Base Qwen2.5-7b tool calling unreliable | Mitigated | System prompts improved; fine-tuning on `finetune/` will fix fully |
| Model still adds paraphrased text despite verbatim instruction | Open | Fine-tune on correct interpretation examples |
| `ss_segmentation_data.csv` dtype warning on load | Minor | Add `dtype={'marital_status': str}` to `pd.read_csv` call |
