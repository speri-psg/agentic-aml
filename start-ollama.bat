@echo off
echo Starting ARIA with Ollama backend...
echo The GGUF (~4 GB) will be downloaded from HF Hub during the Docker build.
echo This only happens once -- subsequent starts use the cached image layer.
echo.
copy /Y .env.ollama .env > nul
docker compose --profile ollama up --build
