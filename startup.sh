#!/bin/bash
set -e

GGUF_PATH="/data/qwen-framl-q4km.gguf"
MODEL_NAME="qwen-aml"

# ── 1. Start Ollama daemon ────────────────────────────────────────────────────
ollama serve &
echo "[startup] Ollama starting..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "[startup] Ollama ready."

# ── 2. Download GGUF if not cached in persistent storage ─────────────────────
if [ ! -f "$GGUF_PATH" ]; then
    echo "[startup] Downloading GGUF (~4GB) from HuggingFace Hub..."
    python3 -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id='speri420/qwen-framl-v13',
    filename='qwen-framl-q4km.gguf',
    local_dir='/data',
    token=os.environ.get('HF_TOKEN'),
)
print('[startup] Download complete.')
"
else
    echo "[startup] GGUF already cached — skipping download."
fi

# ── 3. Write Modelfile via Python (avoids heredoc escaping issues) ────────────
python3 << 'PYEOF'
modelfile = r"""FROM /data/qwen-framl-q4km.gguf

TEMPLATE """
modelfile += '"""' + r"""{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- end }}<|im_start|>assistant
""" + '"""'

modelfile += """

PARAMETER num_ctx 4096
PARAMETER num_predict 1024
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"

SYSTEM "You are an AML (Anti-Money Laundering) analytics AI assistant. You analyze false positive/false negative trade-offs in AML alert thresholds, perform customer behavioral segmentation, and interpret clustering results. Use the available tools to retrieve data, then provide clear, analytical insights. Be concise and reference specific numbers when interpreting results."
"""

with open("/tmp/Modelfile.aml", "w") as f:
    f.write(modelfile)
print("[startup] Modelfile written.")
PYEOF

# ── 4. Register model with Ollama (skip if already registered) ───────────────
if ollama list | grep -q "^$MODEL_NAME"; then
    echo "[startup] Model $MODEL_NAME already registered — skipping create."
else
    echo "[startup] Registering model as $MODEL_NAME..."
    ollama create $MODEL_NAME -f /tmp/Modelfile.aml
    echo "[startup] Model registered."
fi

# ── 5. Launch Dash app ────────────────────────────────────────────────────────
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_MODEL=$MODEL_NAME
echo "[startup] Starting Dash app on port 7860..."
exec python3 /app/application.py
