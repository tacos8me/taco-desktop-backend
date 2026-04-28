#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Add cu130 nvidia libs to LD_LIBRARY_PATH (needed for nvrtc builtins)
VENV_NVIDIA=".venv/lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="${VENV_NVIDIA}/cudnn/lib:${VENV_NVIDIA}/cu13/lib:${VENV_NVIDIA}/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

# Reduce CUDA memory fragmentation (PYTORCH_CUDA_ALLOC_CONF deprecated in torch 2.9+)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Lazy-load CUDA modules (faster startup, lower memory)
export CUDA_MODULE_LOADING=LAZY

# Silence tokenizers fork warning
export TOKENIZERS_PARALLELISM=false

# Load environment variables (optional, for TACO_API_KEY override)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# v1.16.1: explicit HTTP concurrency + backlog. Default uvicorn has
# `limit_concurrency=None` (unbounded request handlers) and backlog=2048
# socket queue. Under 28+ concurrent client polls the kernel SYN backlog
# overflowed and clients saw "Connection reset by peer" instead of clean
# 503/queue. `--limit-concurrency 200` caps in-flight at the ASGI layer
# (clean 503 when full); `--backlog 4096` doubles the socket queue.
exec uv run --no-sync uvicorn server:app --host 0.0.0.0 --port 8090 --no-access-log \
    --limit-concurrency 200 --backlog 4096
