FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer-cached unless requirements.txt changes)
COPY requirements.txt .
# CPU-only torch — avoids pulling 400MB+ NVIDIA CUDA wheels (app has no GPU).
# Separate layer so a retry doesn't re-download torch if only requirements.txt fails.
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        -r requirements.txt huggingface_hub

# Application code
COPY agents/                    ./agents/
COPY config.py                  .
COPY column_mapper.py           .
COPY application.py             .
COPY config/                    ./config/
COPY lambda_rule_analysis.py    .
COPY lambda_cluster_threshold.py .
COPY lambda_ds_performance.py   .
COPY lambda_ofac.py             .
COPY build_ofac_db.py           .
COPY ofac_fuzzy.py              .
COPY make_figures.py            .
COPY sar_scorer.py              .
COPY ds_data_prep.py            .
COPY ingest.py                  .
COPY upload_kb.py               .
# Bank-scale upgrade (2026-06-04): warm-cache layer + 21-product taxonomy module.
# cache_layer is a HARD import in application.py — startup will crash without it.
# synth_products is used by ds_data_prep + the Synthetic Data Profile sidebar panel.
# synth_recurring is build-time only but small; include so ds_data_prep can rebuild
# in-container if needed.
COPY cache_layer.py             .
COPY synth_products.py          .
COPY synth_recurring.py         .

# OFAC SQLite database (tracked in git, auto-refreshes at runtime)
COPY data/                      ./data/

# Demo data download — pulls the full bank-scale XL dataset (aria_synth_xl/)
# from HF Hub at build time so the resulting image is self-contained and
# runnable on a user machine with no extra downloads. The HF dataset
# `speri420/aria-demo-data` holds the XL shape (100K customers, 21 rules
# incl. modern typology, network features pre-computed). Upload via
# upload_demo_data.py before rebuilding.
#
# HF_TOKEN is only required if the dataset is private; for the public
# dataset this can be empty (snapshot_download skips auth).
ARG HF_TOKEN=
ARG HF_DATA_VERSION=2
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
import os; \
token = '${HF_TOKEN}' if '${HF_TOKEN}' else None; \
snapshot_download( \
    repo_id='speri420/aria-demo-data', \
    repo_type='dataset', \
    local_dir='/app', \
    token=token, \
    ignore_patterns=['README.md', '.gitattributes'], \
); \
print('[build] Demo data downloaded.')"

# ChromaDB is created empty on first use — no pre-built store needed
RUN mkdir -p /app/chroma_db

EXPOSE 8050

ENV ARIA_DATA_DIR=/app/aria_synth_xl
ENV CUSTOMERS_CSV=/app/aria_synth_xl/aria_customers.csv

CMD ["python", "application.py"]
