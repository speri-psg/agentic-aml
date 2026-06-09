#!/bin/bash
# Download GGUF on first run (stored in named volume — not re-downloaded on restart).
# On subsequent starts: GGUF already present, skip download.
# Always re-register the Ollama model so changes to TEMPLATE / SYSTEM / PARAMETERs
# in this script take effect on next container restart (ollama create overwrites).

set -euo pipefail

ARIA_MODEL="${OLLAMA_MODEL:-aria-v22}"
HF_GGUF_FILE="${HF_GGUF_FILE:-${ARIA_MODEL}-q4km.gguf}"
HF_REPO="${HF_REPO:-speri420/${ARIA_MODEL}}"
GGUF_PATH="/root/.ollama/gguf/${HF_GGUF_FILE}"

mkdir -p /root/.ollama/gguf

if [ ! -f "${GGUF_PATH}" ]; then
    echo "[ARIA] GGUF not found — downloading ${HF_GGUF_FILE} from ${HF_REPO} ..."
    curl -L --fail --progress-bar \
        ${HF_TOKEN:+-H "Authorization: Bearer ${HF_TOKEN}"} \
        "https://huggingface.co/${HF_REPO}/resolve/main/${HF_GGUF_FILE}" \
        -o "${GGUF_PATH}.tmp"
    mv "${GGUF_PATH}.tmp" "${GGUF_PATH}"
    echo "[ARIA] Download complete: ${GGUF_PATH}"
else
    echo "[ARIA] GGUF already present: ${GGUF_PATH}"
fi

echo "[ARIA] Model: ${ARIA_MODEL}  GGUF: ${GGUF_PATH}"

ollama serve &
SERVE_PID=$!

echo "[ARIA] Waiting for Ollama..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "[ARIA] Ollama ready."

# Always (re-)create the model so Modelfile edits in this script take effect.
# ollama create overwrites an existing model with the same name; GGUF blob is
# already cached so this is fast (~seconds).
# Modelfile mirrors the production local-host setup: Gemma 4 chat TEMPLATE,
# stop tokens, full-GPU layer offload, and the ARIA persona SYSTEM message
# so bare /api/generate or `ollama run` calls still surface the persona.
cat > /tmp/Modelfile << EOF
FROM ${GGUF_PATH}
PARAMETER num_ctx ${OLLAMA_NUM_CTX:-8192}
PARAMETER temperature 0.0
PARAMETER top_p 0.9
PARAMETER stop <turn|>
PARAMETER stop <eos>
PARAMETER num_gpu 999

TEMPLATE """
{{ if .System }}<|turn>user
{{ .System }}<turn|>
{{ end }}
{{ range .Messages }}
{{ if eq .Role "user" }}<|turn>user
{{ .Content }}<turn|>
<|turn>model
{{ else if eq .Role "assistant" }}{{ .Content }}<turn|>
{{ else if eq .Role "tool" }}<|turn>tool
{{ .Content }}<turn|>
<|turn>model
{{ end }}
{{ end }}
"""

SYSTEM "You are ARIA, an AML (Anti-Money Laundering) analytics AI assistant built by Princeton Strategy Group. You analyze false positive/false negative trade-offs in AML alert thresholds, perform customer behavioral segmentation, and interpret clustering results. Use the available tools to retrieve data, then provide clear, analytical insights. IMPORTANT: You MUST respond entirely in English. Plain text only — no LaTeX, no dollar-sign math wrappers. Write '90%' or '90 percent', not '\$90\%\$' or '\$Z\%\$'. Be concise and reference specific numbers when interpreting results."
EOF

echo "[ARIA] (Re-)registering model '${ARIA_MODEL}'..."
ollama create "${ARIA_MODEL}" -f /tmp/Modelfile
echo "[ARIA] Model '${ARIA_MODEL}' ready."

wait "${SERVE_PID}"
