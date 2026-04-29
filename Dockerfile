FROM python:3.12

# uv — fast package installer (from HF Plotly template)
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

# System deps + Ollama (must run as root)
RUN apt-get update && apt-get install -y curl zstd && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

# Non-root user (HF Spaces recommendation)
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Store Ollama models on persistent storage so they survive restarts
ENV OLLAMA_MODELS=/data/ollama_models

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install -r requirements.txt huggingface_hub

# Copy application files
COPY --chown=user . /app
RUN chmod +x /app/startup.sh

USER user

EXPOSE 7860

CMD ["/app/startup.sh"]
