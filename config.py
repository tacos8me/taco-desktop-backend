from pathlib import Path

# Checkpoint paths
CHECKPOINTS_DIR = Path("/mnt/nvme-1/huggingface/ltx-2.3-checkpoints")
DISTILLED_CHECKPOINT = str(CHECKPOINTS_DIR / "ltx-2.3-22b-distilled.safetensors")
DEV_CHECKPOINT = str(CHECKPOINTS_DIR / "ltx-2.3-22b-dev.safetensors")
DISTILLED_LORA = str(CHECKPOINTS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors")
SPATIAL_UPSAMPLER = str(CHECKPOINTS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

# Text encoder — point to the HF snapshot directory containing model*.safetensors
GEMMA_ROOT = "/mnt/nvme-1/huggingface/hub/models--google--gemma-3-12b-pt/snapshots/295efb63d01a7017928f273a94ebb86105c9526f"

# GPU devices for inference (both RTX PRO 6000 Blackwell 96GB)
GPU_DEVICES = ["cuda:0", "cuda:1"]

# Upload storage
UPLOAD_DIR = Path("/mnt/nvme-1/servers/taco-backend/uploads")

# Split-GPU mode: use SplitModelManager instead of per-GPU pipeline copies
USE_SPLIT_GPU = len(GPU_DEVICES) >= 2

# Server
HOST = "0.0.0.0"
PORT = 8090
