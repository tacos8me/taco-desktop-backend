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

exec uv run --no-sync uvicorn server:app --host 0.0.0.0 --port 8090
