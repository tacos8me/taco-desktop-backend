from pathlib import Path

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

# SplitModelManager: encoder hub + denoiser on single GPU
GPU_DEVICES = [LTX_DEVICE]

# Flux model
FLUX_MODEL_REPO = "black-forest-labs/FLUX.2-dev"
HF_CACHE_DIR = "/mnt/nvme-1/huggingface/hub"

# Upload storage
UPLOAD_DIR = Path("/mnt/nvme-1/servers/taco-backend/uploads")

# Server
HOST = "0.0.0.0"
PORT = 8090
