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
SPATIAL_UPSAMPLER = str(CHECKPOINTS_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

# Text encoder — point to the HF snapshot directory containing model*.safetensors
# Set GEMMA_VARIANT=sikaworld in .env to use the abliterated FP4 text encoder
_GEMMA_VARIANTS = {
    "default": "/mnt/nvme-1/huggingface/hub/models--google--gemma-3-12b-pt/snapshots/295efb63d01a7017928f273a94ebb86105c9526f",
    "sikaworld": "/mnt/nvme-1/huggingface/gemma-3-12b-sikaworld",
}
GEMMA_VARIANT = os.environ.get("GEMMA_VARIANT", "default")
GEMMA_ROOT = _GEMMA_VARIANTS.get(GEMMA_VARIANT, _GEMMA_VARIANTS["default"])

# GPU devices
LTX_DEVICE = "cuda:1"    # LTX-2 video generation (~69GB) — moved to free cuda:0
FLUX_DEVICE = "cuda:0"   # Flux 2 image generation (not loaded by default)
LOAD_FLUX = os.environ.get("LOAD_FLUX", "").lower() in ("1", "true", "yes")
CHAT_API_BASE = "http://192.168.1.80:8080"  # External llama-swap server
CHAT_MODEL = "gemma-3-12b-nvfp4"           # Model ID on the external server
CHAR_VISION_MODEL = "gemma-4-31b-it"       # Vision model for Char mode ranking

# SplitModelManager: encoder hub + denoiser on single GPU
GPU_DEVICES = [LTX_DEVICE]

# Flux models
FLUX_MODELS = {
    "flux2-dev": "black-forest-labs/FLUX.2-dev",
    "flux2-klein": "black-forest-labs/FLUX.2-klein-9b-kv",
}
HF_CACHE_DIR = "/mnt/nvme-1/huggingface/hub"

# Turbo LoRA (fal.ai FLUX.2-dev-Turbo — 8-step distilled)
FLUX_TURBO_LORA = "fal/FLUX.2-dev-Turbo"
FLUX_TURBO_LORA_WEIGHT = "flux.2-turbo-lora.safetensors"
FLUX_TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

# Upload storage
UPLOAD_DIR = Path("/mnt/nvme-1/servers/taco-backend/uploads")

# Approved images (curated feed from noodle-i for noodle-v)
APPROVED_IMAGES_DIR = Path("/mnt/nvme-1/servers/taco-backend/approved-images")

# Generation history
HISTORY_DB = Path("/mnt/nvme-1/servers/taco-backend/history.db")
HISTORY_RETENTION_DAYS = 30
THUMBNAIL_DIR = Path("/mnt/nvme-1/servers/taco-backend/thumbnails")

# LoRA storage
LORAS_DIR = Path("/mnt/nvme-1/servers/taco-backend/loras")
MAX_LORA_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
FLUX_LORAS_DIR = Path("/mnt/nvme-1/servers/taco-backend/flux_loras")

# Job queue
MAX_QUEUE_DEPTH = 10
JOB_RESULT_TTL_SECONDS = 600  # 10 minutes

# Server
HOST = "0.0.0.0"
PORT = 8090

# PyTorch performance settings
import torch
torch.backends.cuda.matmul.allow_tf32 = False  # Full float32 precision for VAE decode
torch.backends.cudnn.allow_tf32 = False         # Full float32 precision for VAE convolutions
torch.backends.cudnn.deterministic = True  # Stable algorithm selection
