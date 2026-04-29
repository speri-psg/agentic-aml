---
language:
- en
license: other
license_name: aria-openrail-m
license_link: https://github.com/speri-psg/agentic-aml/blob/main/LICENSE_MODEL
base_model: google/gemma-3-4b-it
tags:
- aml
- finance
- compliance
- tool-calling
- gguf
- ollama
- anti-money-laundering
pipeline_tag: text-generation
---

# ARIA-v1 — Agentic Risk Intelligence for AML

ARIA-v1 is a fine-tuned **Gemma 4 4B** model purpose-built for AML (Anti-Money Laundering)
compliance analytics. It acts as an agentic tool-calling assistant that routes user queries
to the correct analytics function, calls it with correctly structured parameters, and
interprets the pre-computed results without hallucinating numbers.

> **Fine-tuning on a bank's own alert history continuously improves ARIA's precision, reducing the false positive burden on AML investigators over time.**

## Model Details

| Property | Value |
|---|---|
| **Base model** | `google/gemma-3-4b-it` (via `unsloth/gemma-4-E4B-it`) |
| **Fine-tune method** | QLoRA (LoRA r=16, α=32) with Unsloth SFT |
| **Training examples** | 933 domain-specific AML analytics conversations |
| **Quantization** | Q8_0 GGUF |
| **Context length** | 8,192 tokens |
| **Inference** | Ollama (OpenAI-compatible API) |
| **License** | Modified OpenRAIL-M — see [LICENSE_MODEL](https://github.com/speri-psg/agentic-aml/blob/main/LICENSE_MODEL) |

## What it does

The model learns four capabilities from the training data:

1. **Intent classification** — determines which analytics agent handles the query
   (threshold tuning, segmentation, policy Q&A, OFAC screening, or out-of-scope)
2. **Tool calling** — emits correctly structured JSON tool calls for Python analytics functions
3. **Result narration** — copies pre-computed numbers verbatim into the response
   (eliminates hallucinated figures — a key requirement in compliance contexts)
4. **Policy Q&A** — answers AML regulatory questions from retrieved document chunks
   without fabricating citations or statute numbers

## Agents and Tools

| Agent | Tools |
|---|---|
| **ThresholdAgent** | `threshold_tuning`, `sar_backtest`, `rule_2d_sweep`, `list_rules`, `rule_sar_backtest`, `cluster_rule_summary` |
| **SegmentationAgent** | `ds_cluster_analysis`, `cluster_analysis`, `alerts_distribution` |
| **PolicyAgent** | ChromaDB RAG over FFIEC, Wolfsberg, FinCEN, FATF, EU AMLD 4/5/6, AMLR 2024 |
| **OFACAgent** | `ofac_screening`, `ofac_name_lookup` |

## How to Use

### With Ollama

```bash
# 1. Download the GGUF and create a Modelfile
cat > /tmp/Modelfile.aria << 'EOF'
FROM aria-v1-q8.gguf

PARAMETER num_ctx 8192
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop <turn|>
PARAMETER stop <eos>

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
EOF

ollama create aria-v1 -f /tmp/Modelfile.aria
ollama run aria-v1
```

### With the full agentic app

Clone the [agentic-aml](https://github.com/speri-psg/agentic-aml) repository and follow
the setup instructions in the README. The app provides a Plotly Dash UI that wires the
model to the analytics tools.

```bash
git clone https://github.com/speri-psg/agentic-aml
cd agentic-aml
pip install -r requirements.txt
export OLLAMA_MODEL=aria-v1
python application.py
```

## Training Details

- **Base model**: `unsloth/gemma-4-E4B-it`
- **Method**: QLoRA with Unsloth — LoRA r=16, α=32, dropout=0.05
- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Epochs**: 3 | **Effective batch size**: 8 (batch=2, grad_accum=4)
- **Learning rate**: 2e-4 with cosine scheduler, 5% warmup
- **Hardware**: NVIDIA RTX 3090 / RTX 4090 / RTX 5090
- **Inference requirements**: 16 GB VRAM minimum (model is 8 GB+ at Q8_0; 16 GB leaves headroom for context and KV cache)
- **Training data**: 933 examples across threshold tuning, segmentation, policy Q&A,
  OFAC screening, greetings, and out-of-scope handling

## Data Privacy and Air-Gap Deployment

ARIA-v1 is designed to run **entirely on-premises** inside a bank's own infrastructure.
All analytics — threshold sweeps, segmentation, SAR backtests, OFAC screening — are
executed by local Python functions against the bank's own data. The model never sends
customer data, transaction records, or alert information to any external API or cloud LLM.

When a bank deploys ARIA-v1 on their own hardware, customer data stays within their
environment at all times. There is no dependency on OpenAI, Anthropic, or any third-party
inference service. This makes ARIA-v1 suitable for regulated institutions where data
residency, confidentiality, and audit requirements prohibit sending customer data outside
the bank's perimeter.

## Intended Use

ARIA-v1 is designed for AML compliance teams and researchers building or evaluating
AI-assisted transaction monitoring tools. It is intended to be used with the
[agentic-aml](https://github.com/speri-psg/agentic-aml) application stack which
supplies the analytics tools and data layer.

## Limitations

- Designed for the synthetic dataset schema provided in `agentic-aml` — adapting to a
  different schema requires re-mapping column names via `column_map.yaml`
- Policy Q&A quality depends on the documents ingested into ChromaDB — only documents
  in the knowledge base can be cited
- Not a substitute for qualified AML compliance advice

## License

This model is released under a **modified OpenRAIL-M license**.
Free for personal use, academic research, and organizations with annual revenue and
total funding each below **USD $2 million**.
Commercial use above that threshold requires a separate license.
The model may not be used to build a competing AML transaction monitoring or
financial crime analytics product or service.

Full license: [LICENSE_MODEL](https://github.com/speri-psg/agentic-aml/blob/main/LICENSE_MODEL)
