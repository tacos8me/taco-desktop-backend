# Flux 2 Image Server + LTX Single-GPU Revert Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Flux 2 Dev image generation on cuda:1 while running LTX-2 video on cuda:0.

**Architecture:** Same FastAPI server, two independent GPU workloads. LTX uses SplitModelManager with `GPU_DEVICES = ["cuda:0"]` — loads one encoder hub + one denoiser worker, all on cuda:0 (~59GB). FluxManager loads Flux2Pipeline with FP8 transformer on cuda:1 (~79GB). Both run inference in thread pool executors, fully parallel.

**Why SplitModelManager, not PipelineManager:** PipelineManager creates 4 separate pipeline objects per GPU (DistilledPipeline, TI2VidTwoStagesPipeline, RetakePipeline, A2VidPipelineTwoStage), each with its own ModelLedger + DummyRegistry = each loads its own transformer (~22GB × 4+ = OOM). SplitModelManager loads ONE transformer and swaps between dev/distilled/dev_lora states.

**Tech Stack:** FastAPI, diffusers (Flux2Pipeline, Flux2Transformer2DModel), ltx-pipelines, torch cu130, Pillow

---

### Task 1: Update config.py for dual-workload GPU assignment

**Files:**
- Modify: `config.py`

**Step 1: Update config.py**

Replace the current GPU config with explicit per-workload device assignment:

```python
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
# (PipelineManager can't be used — loads 4 pipelines with 4 transformers = OOM)
GPU_DEVICES = [LTX_DEVICE]
USE_SPLIT_GPU = True

# Flux model
FLUX_MODEL_REPO = "black-forest-labs/FLUX.2-dev"
HF_CACHE_DIR = "/mnt/nvme-1/huggingface/hub"

# Upload storage
UPLOAD_DIR = Path("/mnt/nvme-1/servers/taco-backend/uploads")

# Server
HOST = "0.0.0.0"
PORT = 8090
```

**Step 2: Verify config is valid**

Run: `uv run --no-sync python -c "import config; print(config.LTX_DEVICE, config.FLUX_DEVICE)"`
Expected: `cuda:0 cuda:1`

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat: split GPU config — LTX on cuda:0, Flux on cuda:1"
```

---

### Task 2: Add diffusers dependency to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependencies**

Add `diffusers` and `sentencepiece` to the dependencies list:

```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "pydantic>=2.0",
    "ltx-core",
    "ltx-pipelines",
    "diffusers>=0.37.0.dev0",
    "sentencepiece",
]
```

**Step 2: Install**

diffusers 0.37.0.dev0 is a pre-release. Use `--prerelease=allow` with uv:

Run: `uv sync --prerelease=allow` (or add `--no-sync` to preserve cu130 torch)

If uv sync changes torch or has issues, fall back to pip:
`uv pip install --pre "diffusers>=0.37.0.dev0" sentencepiece`

**Step 3: Verify import**

Run: `uv run --no-sync python -c "from diffusers import Flux2Pipeline; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add diffusers and sentencepiece for Flux 2 support"
```

---

### Task 3: Create flux_manager.py

**Files:**
- Create: `flux_manager.py`

**Step 1: Write FluxManager**

Adapt from tacojourney's `flux_inference.py`. Key differences from tacojourney:
- No preview callback (taco-backend returns final result only)
- No base64 image decoding (uses storage:// URIs via upload_store)
- Uses `config.FLUX_DEVICE` and `config.FLUX_MODEL_REPO`
- Singleton pattern not needed (server.py creates one instance)

```python
"""Flux 2 Dev image generation manager.

Loads Flux2Pipeline with FP8 layerwise casting on a dedicated GPU.
FP8 transformer (~32GB) + bf16 Mistral text encoder (~47GB) = ~79GB.
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import os
import time

import torch
from PIL import Image

import config

# Ensure HF uses our NVMe cache
os.environ.setdefault("HF_HOME", "/mnt/nvme-1/huggingface")

logger = logging.getLogger(__name__)

# Fix cuBLAS strided batched GEMM bug on Blackwell (SM 12.0)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
    except Exception:
        pass


class FluxManager:
    """Manages Flux 2 pipeline lifecycle and inference on a dedicated GPU."""

    def __init__(self) -> None:
        self._pipe = None
        self._device = config.FLUX_DEVICE
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        """Load the Flux 2 pipeline with FP8 transformer."""
        from diffusers import Flux2Pipeline
        from diffusers.models import Flux2Transformer2DModel

        logger.info("Loading Flux2 pipeline on %s ...", self._device)
        t0 = time.monotonic()

        # FP8 layerwise casting: stores weights in float8_e4m3fn (~32GB)
        # but computes in bf16. Halves VRAM vs full bf16 (~64GB).
        transformer = Flux2Transformer2DModel.from_pretrained(
            config.FLUX_MODEL_REPO,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            cache_dir=config.HF_CACHE_DIR,
        )
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )

        pipe = Flux2Pipeline.from_pretrained(
            config.FLUX_MODEL_REPO,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            cache_dir=config.HF_CACHE_DIR,
        )
        pipe = pipe.to(self._device)
        self._pipe = pipe

        elapsed = time.monotonic() - t0
        logger.info("Flux2 pipeline loaded in %.1fs on %s", elapsed, self._device)

    def unload(self) -> None:
        """Free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Flux2 pipeline unloaded")

    @torch.inference_mode()
    def _generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> bytes:
        """Generate an image (txt2img) and return PNG bytes."""
        generator = torch.Generator(device=self._device).manual_seed(seed)

        try:
            result = self._pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            logger.exception("Flux generate OOM, unloading pipeline")
            self.unload()
            raise

        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @torch.inference_mode()
    def _img2img(
        self,
        prompt: str,
        image_path: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> bytes:
        """Edit an image using Flux 2 Kontext reference latents."""
        generator = torch.Generator(device=self._device).manual_seed(seed)
        ref_image = Image.open(image_path).convert("RGB")

        try:
            result = self._pipe(
                image=ref_image,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            logger.exception("Flux img2img OOM, unloading pipeline")
            self.unload()
            raise

        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    # --- Async API ---

    async def generate_text_to_image(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 0,
    ) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._generate, prompt, width, height,
                num_inference_steps, guidance_scale, seed,
            )

    async def generate_image_to_image(
        self,
        prompt: str,
        image_path: str,
        width: int,
        height: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 0,
    ) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._img2img, prompt, image_path, width, height,
                num_inference_steps, guidance_scale, seed,
            )
```

**Step 2: Verify syntax**

Run: `uv run --no-sync python -c "import flux_manager; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add flux_manager.py
git commit -m "feat: add FluxManager for Flux 2 Dev image generation"
```

---

### Task 4: Update server.py — add Flux endpoints and simplify LTX init

**Files:**
- Modify: `server.py`

**Step 1: Add Flux manager and image request models**

At the top of server.py, after existing imports, add:

```python
from flux_manager import FluxManager
```

Simplify the globals section — always use SplitModelManager for LTX, add FluxManager:

```python
from split_model_manager import SplitModelManager
from flux_manager import FluxManager

manager = SplitModelManager()
flux = FluxManager()
uploads = UploadStore(config.UPLOAD_DIR)
```

Remove the `USE_SPLIT_GPU` conditional, the `PipelineManager` import, and the `_duration_to_frames`/`_resolution_to_dims` import (move those to a helpers import or inline from pipeline_manager).

**Step 2: Update lifespan to load both**

```python
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading LTX pipelines on %s ...", config.GPU_DEVICES)
    manager.load_all()
    logger.info("LTX pipelines ready.")

    logger.info("Loading Flux pipeline on %s ...", config.FLUX_DEVICE)
    flux.load()
    logger.info("Flux pipeline ready.")
    yield
```

**Step 3: Add image request models**

```python
ImageModelName = Literal["flux2-dev"]


class TextToImageRequest(BaseModel):
    prompt: str
    model: ImageModelName = "flux2-dev"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    seed: int | None = None


class ImageToImageRequest(BaseModel):
    prompt: str
    image_uri: str
    model: ImageModelName = "flux2-dev"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    seed: int | None = None
```

**Step 4: Add image generation endpoints**

```python
@app.post("/v1/text-to-image")
async def text_to_image(body: TextToImageRequest) -> Response:
    if not flux.is_ready:
        return _error(500, "Flux pipeline not loaded")
    try:
        # Snap to multiples of 16
        width = (body.width // 16) * 16
        height = (body.height // 16) * 16
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

        image_bytes = await flux.generate_text_to_image(
            prompt=body.prompt,
            width=width,
            height=height,
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            seed=seed,
        )
        return Response(content=image_bytes, media_type="image/png")
    except Exception as exc:
        logger.exception("text-to-image failed")
        return _error(500, str(exc))


@app.post("/v1/image-to-image")
async def image_to_image(body: ImageToImageRequest) -> Response:
    if not flux.is_ready:
        return _error(500, "Flux pipeline not loaded")
    try:
        image_path = str(uploads.resolve(body.image_uri))
        width = (body.width // 16) * 16
        height = (body.height // 16) * 16
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

        image_bytes = await flux.generate_image_to_image(
            prompt=body.prompt,
            image_path=image_path,
            width=width,
            height=height,
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            seed=seed,
        )
        return Response(content=image_bytes, media_type="image/png")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("image-to-image failed")
        return _error(500, str(exc))
```

**Step 5: Update health endpoint to report both**

```python
@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "ltx": "ready" if manager.is_ready else "not_loaded",
        "flux": "ready" if flux.is_ready else "not_loaded",
    }
```

**Step 6: Commit**

```bash
git add server.py
git commit -m "feat: add Flux image endpoints, use SplitModelManager for LTX"
```

---

### Task 5: Update run.sh with expandable_segments

**Files:**
- Modify: `run.sh`

**Step 1: Add PYTORCH_CUDA_ALLOC_CONF**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Add cu130 nvidia libs to LD_LIBRARY_PATH (needed for nvrtc builtins)
VENV_NVIDIA=".venv/lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="${VENV_NVIDIA}/cu13/lib:${VENV_NVIDIA}/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec uv run --no-sync uvicorn server:app --host 0.0.0.0 --port 8090
```

**Step 2: Commit**

```bash
git add run.sh
git commit -m "feat: add expandable_segments to reduce CUDA fragmentation"
```

---

### Task 6: Smoke test — start server, verify VRAM, test endpoints

**Step 1: Start the server**

```bash
./run.sh
```

Or in background:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ./run.sh > /tmp/taco-backend.log 2>&1 &
```

**Step 2: Wait for models to load, check health**

```bash
curl -s http://localhost:8090/health | python3 -m json.tool
```

Expected:
```json
{"status": "ok", "ltx": "ready", "flux": "ready"}
```

**Step 3: Verify VRAM distribution**

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

Expected:
- GPU 0 (cuda:0): ~55-70GB (LTX SplitModelManager: encoder hub + denoiser)
- GPU 1 (nvidia-smi 1, 24GB): ~0GB (unused)
- GPU 2 (cuda:1): ~75-82GB (Flux FP8)

**Step 4: Test Flux text-to-image**

```bash
curl -s -o /tmp/test-flux-t2i.png -w "\nHTTP %{http_code} | Time: %{time_total}s | Size: %{size_download}\n" \
  -X POST http://localhost:8090/v1/text-to-image \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A majestic mountain landscape at golden hour","width":1024,"height":1024}'
```

Expected: HTTP 200, PNG file, ~30-60s generation time

**Step 5: Verify output**

```bash
file /tmp/test-flux-t2i.png
identify /tmp/test-flux-t2i.png  # if imagemagick installed
```

Expected: PNG image, 1024x1024

**Step 6: Test LTX text-to-video (fast model)**

```bash
curl -s -o /tmp/test-ltx-t2v.mp4 -w "\nHTTP %{http_code} | Time: %{time_total}s | Size: %{size_download}\n" \
  -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cat walking","model":"ltx-2-3-fast","resolution":"1920x1080","duration":2.0,"fps":24.0}'
```

Expected: HTTP 200, MP4 file, ~15-20s

**Step 7: Test concurrent LTX + Flux**

Run both in parallel:
```bash
curl -s -o /tmp/concurrent-flux.png -w "Flux: HTTP %{http_code} | %{time_total}s\n" \
  -X POST http://localhost:8090/v1/text-to-image \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A serene lake","width":1024,"height":1024}' &

curl -s -o /tmp/concurrent-ltx.mp4 -w "LTX: HTTP %{http_code} | %{time_total}s\n" \
  -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Ocean waves","model":"ltx-2-3-fast","resolution":"1920x1080","duration":2.0,"fps":24.0}' &

wait
```

Expected: Both return 200, running truly in parallel on separate GPUs.

**Step 8: Commit (if any fixes needed)**

---

### Task 7: Test LTX pro model on single GPU

**Step 1: Test pro model text-to-video**

```bash
curl -s -o /tmp/test-ltx-pro.mp4 -w "\nHTTP %{http_code} | Time: %{time_total}s | Size: %{size_download}\n" \
  -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A dog running on a beach at sunset","model":"ltx-2-3-pro","resolution":"1920x1080","duration":2.0,"fps":24.0}'
```

Expected: HTTP 200, MP4 file, ~60-70s. SplitModelManager on a single GPU has ~27-37GB headroom for activations + transformer swaps. The transformer swap (dev → dev_lora → dev) frees the old before loading the new, and all references (local `transformer` variable, closures) are deleted + `cleanup_memory()` called between stages.

**Step 2: Verify VRAM during/after**

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

GPU 0 should peak ~80-85GB during pro model inference, then return to ~60GB baseline.

---

## Validation Findings

**Issue found and fixed:** Plan v1 proposed reverting to PipelineManager for LTX on single GPU. Validators identified that PipelineManager creates 4 separate pipeline objects, each loading its own transformer via DummyRegistry (~22GB × 4+ = OOM). **Fix:** Keep SplitModelManager which loads one encoder hub + one denoiser worker with a single swappable transformer.

**Additional fixes from validation:**
- Added `os.environ.setdefault("HF_HOME", "/mnt/nvme-1/huggingface")` to flux_manager.py
- Kept `USE_SPLIT_GPU = True` in config.py instead of removing it
- Updated VRAM expectations in smoke tests
