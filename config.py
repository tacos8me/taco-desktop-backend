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


def _load_admin_keys() -> set[str]:
    """Load admin keys from .admin_keys file and/or TACO_ADMIN_KEY env var.

    v1.8.2 / SEC P0-2: admin-gated endpoints (system pause/resume/turbo,
    config mutation, un/reload, pool scaling) check this set. Empty → admin
    auth is in the "backwards-compat bridge" state: every entry in
    ``API_KEYS`` is treated as admin. Populate ``.admin_keys`` with the
    actual operator bearer(s) to lock the 12 mutation endpoints down.
    """
    keys: set[str] = set()
    keys_file = Path(__file__).parent / ".admin_keys"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.add(line)
    env_key = os.environ.get("TACO_ADMIN_KEY", "").strip()
    if env_key:
        keys.add(env_key)
    return keys


API_KEYS: set[str] = _load_api_keys()
ADMIN_KEYS: set[str] = _load_admin_keys()

# Checkpoint paths
CHECKPOINTS_DIR = Path("/mnt/nvme-1/huggingface/ltx-2.3-checkpoints")
DISTILLED_CHECKPOINT = str(CHECKPOINTS_DIR / "ltx-2.3-22b-distilled-1.1.safetensors")
DEV_CHECKPOINT = str(CHECKPOINTS_DIR / "ltx-2.3-22b-dev.safetensors")
DISTILLED_LORA = str(CHECKPOINTS_DIR / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
SPATIAL_UPSAMPLER = str(CHECKPOINTS_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

# Text encoder — point to the HF snapshot directory containing model*.safetensors
# Set GEMMA_VARIANT=sikaworld in .env to use the abliterated FP4 text encoder
_GEMMA_VARIANTS = {
    "default": "/mnt/nvme-1/huggingface/hub/models--google--gemma-3-12b-pt/snapshots/295efb63d01a7017928f273a94ebb86105c9526f",
    "sikaworld": "/mnt/nvme-1/huggingface/gemma-3-12b-sikaworld",
}
GEMMA_VARIANT = os.environ.get("GEMMA_VARIANT", "default")
GEMMA_ROOT = _GEMMA_VARIANTS.get(GEMMA_VARIANT, _GEMMA_VARIANTS["default"])

# GPU devices — SINGLE-GPU SWAP MODE
# Both LTX and Flux live on cuda:0 but are mutually exclusive (LTX active ~79 GB
# + Flux active ~81 GB > 96 GB). The server dispatcher auto-swaps by evicting
# LTX before any Flux forward pass and (re)loading LTX before any video request.
# cuda:1 runs ACE + JoyAI sidecars (or LTX sidecar in turbo/dual-GPU mode).
LTX_DEVICE = "cuda:0"
FLUX_DEVICE = "cuda:0"
LOAD_FLUX = os.environ.get("LOAD_FLUX", "").lower() in ("1", "true", "yes")

# JoyAI image-edit sidecar — out-of-process isolated inference (v1.1.8).
# See docs/API.md v1.1.8 changelog and /mnt/nvme-1/servers/joyai-sidecar/ for the
# sidecar itself. Empty LOAD_JOYAI means feature is disabled — requests for
# model="joyai-edit" will return 503.
JOYAI_SIDECAR_URL = os.environ.get("JOYAI_SIDECAR_URL", "http://127.0.0.1:8092")
LOAD_JOYAI = os.environ.get("LOAD_JOYAI", "").lower() in ("1", "true", "yes")

# ERNIE-Image text-to-image sidecar on cuda:1 (v1.0).
# 8B DiT, Apache 2.0. Swaps with JoyAI on cuda:1.
ERNIE_SIDECAR_URL = os.environ.get("ERNIE_SIDECAR_URL", "http://127.0.0.1:8094")
LOAD_ERNIE = os.environ.get("LOAD_ERNIE", "").lower() in ("1", "true", "yes")

# LTX video sidecar — independent LTX pipeline on cuda:1 for turbo mode (v1.2).
# Managed via systemctl; taco-backend calls /load and /unload to control GPU memory.
LTX_SIDECAR_URL = os.environ.get("LTX_SIDECAR_URL", "http://127.0.0.1:8093")

# Optional SECOND LTX sidecar in the turbo pool (v1.5 — Modal RTX Pro 6000, etc.).
# When set, turbo mode spins up an EXTRA concurrent worker dispatching to this
# URL, giving 3 concurrent video workers total (main cuda:0 + local cuda:1 sidecar
# + remote). Leave empty to disable. Token is the Bearer value — NOT an env-var name.
LTX_REMOTE_SIDECAR_URL = os.environ.get("LTX_REMOTE_SIDECAR_URL", "").strip()
LTX_REMOTE_SIDECAR_TOKEN = os.environ.get("LTX_REMOTE_SIDECAR_TOKEN", "").strip()
# v1.6: upper bound on concurrent remote workers (each = 1 Modal container).
# User can scale 0..MAX via `POST /v1/system/pool/remote-workers`. Must not
# exceed the Modal function's `max_containers` or we queue forever.
LTX_REMOTE_SIDECAR_MAX_WORKERS = int(os.environ.get("LTX_REMOTE_SIDECAR_MAX_WORKERS", "4"))

# ACE-Step music generation sidecar on cuda:1 (v1.2).
ACE_SIDECAR_URL = os.environ.get("ACE_SIDECAR_URL", "http://127.0.0.1:8001")
LOAD_ACE = os.environ.get("LOAD_ACE", "").lower() in ("1", "true", "yes")
MAX_MUSIC_PENDING = int(os.environ.get("MAX_MUSIC_PENDING", "5"))

CHAT_API_BASE = "http://192.168.1.80:8080"  # External llama-swap server
CHAT_MODEL = "gemma-3-12b-nvfp4"           # Model ID on the external server
CHAR_VISION_MODEL = "gemma-4-31b-it"       # Vision model for Char mode ranking

# Dual-GPU LTX: dedicate BOTH GPUs to LTX video generation.
# Two independent users can generate videos simultaneously (one per GPU).
# Disables Flux, ACE, and JoyAI — cuda:1 is fully allocated to LTX.
DUAL_GPU_LTX = os.environ.get("DUAL_GPU_LTX", "").lower() in ("1", "true", "yes")

# SplitModelManager: encoder hub + denoiser on cuda:0 (in-process).
# DUAL_GPU_LTX uses the LTX sidecar on cuda:1 for the second worker
# (separate process avoids CUDA illegal memory access from concurrent
# multi-GPU ops in the same process).
GPU_DEVICES = [LTX_DEVICE]
if DUAL_GPU_LTX:
    LOAD_FLUX = False
    LOAD_JOYAI = False
    LOAD_ACE = False

# Flux models
FLUX_MODELS = {
    "flux2-dev": "black-forest-labs/FLUX.2-dev",
    "flux2-klein": "black-forest-labs/FLUX.2-klein-9b-kv",
}
HF_CACHE_DIR = "/mnt/nvme-1/huggingface/hub"

# Turbo sigma schedule: 8-step distilled flow from fal.ai FLUX.2-dev-Turbo.
# Used when `turbo: true` on Flux image requests. Composable with an explicit
# `flux2-turbo` folder-drop LoRA (see flux_loras/) for the fully-fused turbo
# behavior. Sigma schedule alone also works without the LoRA, at slightly
# different quality.
FLUX_TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

# Upload storage
UPLOAD_DIR = Path("/mnt/nvme-1/servers/taco-backend/uploads")

# Temp dir for intermediate MP4 encode buffer. PyAV needs a path (can't use BytesIO
# without a format= hint), and /tmp on this host is ext-backed — each encode pays
# NVMe write + read roundtrip. /dev/shm is tmpfs (pure RAM), falling back to /tmp.
MP4_TMPDIR = Path("/dev/shm") if Path("/dev/shm").exists() else Path("/tmp")

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

# v1.8.2 / SEC P1-3: per-API-key queue caps. These apply BEFORE the global
# MAX_QUEUE_DEPTH / MAX_MUSIC_PENDING / MAX_BATCH_QUEUE_DEPTH caps, so one
# bearer can't single-tenant the whole queue. 429 with per_key_queue_full
# on breach. Override via env.
PER_KEY_QUEUE_CAP = int(os.environ.get("PER_KEY_QUEUE_CAP", "3"))
PER_KEY_MUSIC_CAP = int(os.environ.get("PER_KEY_MUSIC_CAP", "2"))
PER_KEY_BATCH_CAP = int(os.environ.get("PER_KEY_BATCH_CAP", "2"))

# v1.8.2 / SEC P2-3+P2-4: per-API-key upload + LoRA quotas. The upload cap
# is a rolling 24h byte counter keyed by sha256(api_key); the LoRA cap is
# total active LoRAs in the registry owned by this key.
PER_KEY_UPLOAD_BYTES_PER_DAY = int(
    os.environ.get("PER_KEY_UPLOAD_BYTES_PER_DAY", str(10 * 1024 * 1024 * 1024))
)
PER_KEY_LORA_COUNT = int(os.environ.get("PER_KEY_LORA_COUNT", "20"))

# Turbo mode — dual-GPU inference
AUTO_TURBO_IDLE_MINUTES = int(os.environ.get("AUTO_TURBO_IDLE_MINUTES", "15"))

# torch.compile: compile transformer blocks for Inductor-optimized inference.
# First request after load takes ~60-120s warmup. Default OFF — set TORCH_COMPILE=1 to enable.
ENABLE_TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "").lower() in ("1", "true", "yes")

# Batch queue
MAX_BATCH_QUEUE_DEPTH = 5                   # max concurrent batch submissions
MAX_BATCH_ITEMS = 50                        # max items per batch
BATCH_RESULT_TTL_SECONDS = 1800             # 30 min (batches are larger, keep longer)

# Server
HOST = "0.0.0.0"
PORT = 8090

# PyTorch performance settings
import torch
torch.backends.cuda.matmul.allow_tf32 = False  # Full float32 precision for VAE decode
torch.backends.cudnn.allow_tf32 = False         # Full float32 precision for VAE convolutions
torch.backends.cudnn.deterministic = True  # Stable algorithm selection
# bf16 reduced precision accumulation: leave at PyTorch default (True).
# LTX-2 was trained with default bf16 accumulation — forcing float32 accumulation
# creates a training/inference precision mismatch that compounds across 56 transformer
# layers × 20 denoising steps, causing character movement artifacts. ComfyUI also
# uses the default. The old False setting was based on Flux VAE analysis but
# incorrectly applied globally to the LTX transformer.
