#!/bin/bash
set -e

# ── Configuration ─────────────────────────────────────────────────────────────
HF_REPO="speri420/aria-v3"
GGUF_FILE="aria-v3-q8.gguf"
GGUF_PATH="/data/${GGUF_FILE}"
MODEL_NAME="aria-v3"

# Bind Ollama directly to the HF Space public port so the local Dash app can
# reach it at: OLLAMA_BASE_URL=https://speri420-agentic-aml-demo.hf.space/v1
export OLLAMA_HOST=0.0.0.0:7860

# ── 1. Start Ollama daemon ────────────────────────────────────────────────────
ollama serve &
echo "[startup] Ollama starting on port 7860..."
until curl -s http://localhost:7860/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "[startup] Ollama ready."

# ── 2. Download GGUF if not cached in persistent storage ─────────────────────
if [ ! -f "$GGUF_PATH" ]; then
    echo "[startup] Downloading ${GGUF_FILE} from HuggingFace Hub (${HF_REPO})..."
    python3 -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id='${HF_REPO}',
    filename='${GGUF_FILE}',
    local_dir='/data',
    token=os.environ.get('HF_TOKEN'),
)
print('[startup] Download complete.')
"
else
    echo "[startup] GGUF already cached at ${GGUF_PATH} — skipping download."
fi

# ── 3. Write Modelfile ────────────────────────────────────────────────────────
python3 << 'PYEOF'
SYSTEM_PROMPT = (
    "You are ARIA, an AML (Anti-Money Laundering) analytics AI assistant built by Xceed. "
    "You analyze false positive/false negative trade-offs in AML alert thresholds, "
    "perform customer behavioral segmentation, and interpret clustering results. "
    "Use the available tools to retrieve data, then provide clear, analytical insights. "
    "IMPORTANT: You MUST respond entirely in English. "
    "Be concise and reference specific numbers when interpreting results."
)

lines = [
    "FROM /data/aria-v3-q8.gguf",
    "",
    "PARAMETER num_ctx 8192",
    "PARAMETER temperature 0.1",
    "PARAMETER top_p 0.9",
    "PARAMETER stop <turn|>",
    "PARAMETER stop <eos>",
    "",
    'TEMPLATE """',
    "{{ if .System }}<|turn>user",
    "{{ .System }}<turn|>",
    "{{ end }}",
    "{{ range .Messages }}",
    '{{ if eq .Role "user" }}<|turn>user',
    "{{ .Content }}<turn|>",
    "<|turn>model",
    '{{ else if eq .Role "assistant" }}{{ .Content }}<turn|>',
    '{{ else if eq .Role "tool" }}<|turn>tool',
    "{{ .Content }}<turn|>",
    "<|turn>model",
    "{{ end }}",
    "{{ end }}",
    '"""',
    "",
    f'SYSTEM "{SYSTEM_PROMPT}"',
]

with open("/tmp/Modelfile.aml", "w") as f:
    f.write("\n".join(lines))
print("[startup] Modelfile written.")
PYEOF

# ── 4. Register model with Ollama ─────────────────────────────────────────────
if ollama list | grep -q "^${MODEL_NAME}"; then
    echo "[startup] Model ${MODEL_NAME} already registered — skipping create."
else
    echo "[startup] Registering model as ${MODEL_NAME}..."
    ollama create "${MODEL_NAME}" -f /tmp/Modelfile.aml
    echo "[startup] Model registered."
fi

echo "[startup] aria-v2 is live."
echo "[startup] Connect your local app with:"
echo "  OLLAMA_BASE_URL=https://speri420-agentic-aml-demo.hf.space/v1"
echo "  OLLAMA_MODEL=aria-v2"

# Keep the container alive — Ollama is already running in background
wait
