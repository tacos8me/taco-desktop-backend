import os
from pathlib import Path


def _load_api_keys() -> set[str]:
    """Load API keys from .api_keys file and/or TACO_API_KEY env var."""
    keys: set[str] = set()
    keys_file = Path(__file__).parent / ".api_keys"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.add(line)
    env_key = os.environ.get("TACO_API_KEY", "").strip()
    if env_key:
        keys.add(env_key)
    return keys


API_KEYS: set[str] = _load_api_keys()

# Checkpoint paths
CHECKPOINTS_DIR = Path("/mnt/nvme-1/huggingface/ltx-2.3-checkpoints")
DISTILLED_CHECKPOINT = str(CHECKPOINTS_DIR / "ltx-2.3-22b-distilled.safetensors")
DEV_CHECKPOINT = str(CHECKPOINTS_DIR / "ltx-2.3-22b-dev.safetensors")
DISTILLED_LORA = str(CHECKPOINTS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors")
SPATIAL_UPSAMPLER = str(CHECKPOINTS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

# Text encoder — point to the HF snapshot directory containing model*.safetensors
GEMMA_ROOT = "/mnt/nvme-1/huggingface/hub/models--google--gemma-3-12b-pt/snapshots/295efb63d01a7017928f273a94ebb86105c9526f"

# GPU devices
LTX_DEVICE = "cuda:0"    # LTX-2 video generation (~59GB)
FLUX_DEVICE = "cuda:1"   # Flux 2 image generation (~79GB FP8)
CHAT_API_BASE = "http://192.168.1.80:8080"  # External llama-swap server
CHAT_MODEL = "gemma-3-12b-nvfp4"           # Model ID on the external server

# SplitModelManager: encoder hub + denoiser on single GPU
GPU_DEVICES = [LTX_DEVICE]

# Flux model
FLUX_MODEL_REPO = "black-forest-labs/FLUX.2-dev"
HF_CACHE_DIR = "/mnt/nvme-1/huggingface/hub"

# Upload storage
UPLOAD_DIR = Path("/mnt/nvme-1/servers/taco-backend/uploads")

# LoRA storage
LORAS_DIR = Path("/mnt/nvme-1/servers/taco-backend/loras")
MAX_LORA_SIZE_BYTES = 500 * 1024 * 1024  # 500MB

# Job queue
MAX_QUEUE_DEPTH = 10
JOB_RESULT_TTL_SECONDS = 600  # 10 minutes

# Server
HOST = "0.0.0.0"
PORT = 8090

# PyTorch performance settings
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
