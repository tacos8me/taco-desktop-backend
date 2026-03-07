#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Add cu130 nvidia libs to LD_LIBRARY_PATH (needed for nvrtc builtins)
VENV_NVIDIA=".venv/lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="${VENV_NVIDIA}/cu13/lib:${VENV_NVIDIA}/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec uv run --no-sync uvicorn server:app --host 0.0.0.0 --port 8090
