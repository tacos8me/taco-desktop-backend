# LTX-Compatible Inference Server — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a FastAPI server implementing the taco-desktop LTX-compatible API contract, wrapping the existing `ltx-pipelines` package with dual-GPU concurrent inference.

**Architecture:** FastAPI app with an asyncio-based GPU dispatch queue. Two pipeline instances (one per RTX PRO 6000 on `cuda:0` and `cuda:2`) handle requests concurrently. Uploads stored to disk with UUID-based `storage://` URIs. All generation endpoints return raw MP4 bytes.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, ltx-core + ltx-pipelines (from `/mnt/nvme-1/repos/LTX-2`), PyTorch 2.9, PyAV

---

## Reference Paths

| Resource | Path |
|---|---|
| LTX-2 repo | `/mnt/nvme-1/repos/LTX-2` |
| Distilled checkpoint | `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/ltx-2.3-22b-distilled.safetensors` |
| Dev checkpoint | `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/ltx-2.3-22b-dev.safetensors` |
| Distilled LoRA | `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors` |
| Spatial upsampler x2 | `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors` |
| Gemma 3 12B | `/mnt/nvme-1/huggingface/hub/models--google--gemma-3-12b-pt` (HF cache layout) |
| GPU 0 | RTX PRO 6000 Blackwell, 96GB (`cuda:0`) |
| GPU 1 | RTX PRO 4000 Blackwell, 24GB (`cuda:1`) — **not used** |
| GPU 2 | RTX PRO 6000 Blackwell, 96GB (`cuda:2`) |
| Project dir | `/mnt/nvme-1/servers/taco-backend` |
| API spec | `/mnt/nvme-1/servers/taco-backend/ltx-compatible-server.md` |

## Resolution & Frame Mapping

taco-desktop sends `resolution` as `"WxH"` and `duration`/`fps`. The server must convert:

```python
width, height = map(int, resolution.split("x"))
num_frames = round(duration * fps)
# Snap to 8k+1 constraint
num_frames = ((num_frames - 1) // 8) * 8 + 1
```

## Model Mapping

| `model` field | Pipeline | Checkpoint | Notes |
|---|---|---|---|
| `ltx-2-3-fast` | `DistilledPipeline` | `ltx-2.3-22b-distilled.safetensors` | 8-step, no CFG |
| `ltx-2-3-pro` | `TI2VidTwoStagesPipeline` | `ltx-2.3-22b-dev.safetensors` + distilled LoRA | 30-step, CFG guided, two-stage upscale |

---

## Task 1: Project Setup & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `CLAUDE.md`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "taco-backend"
version = "0.1.0"
description = "LTX-compatible inference server for taco-desktop"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "pydantic>=2.0",
]

[tool.uv.sources]
ltx-core = { path = "/mnt/nvme-1/repos/LTX-2/packages/ltx-core" }
ltx-pipelines = { path = "/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines" }
```

Note: `ltx-core` and `ltx-pipelines` are installed as path dependencies from the local LTX-2 repo. Their transitive deps (torch, av, etc.) are pulled in automatically.

**Step 2: Create CLAUDE.md**

```markdown
# taco-backend

LTX-compatible inference server for taco-desktop.

## Structure
- `server.py` — FastAPI app, all HTTP endpoints
- `pipeline_manager.py` — Dual-GPU pipeline loading and request dispatch
- `config.py` — Paths, model mapping, resolution tables
- `upload_store.py` — UUID file storage for uploads

## Key commands
- Run: `uv run uvicorn server:app --host 0.0.0.0 --port 8080`
- Test: `uv run pytest tests/ -v`

## Conventions
- Use ltx-pipelines classes directly (DistilledPipeline, TI2VidTwoStagesPipeline, etc.)
- All generation runs under `@torch.inference_mode()`
- Return raw MP4 bytes with `Content-Type: video/mp4`
```

**Step 3: Initialize git and install deps**

```bash
cd /mnt/nvme-1/servers/taco-backend
git init
uv sync
```

**Step 4: Commit**

```bash
git add pyproject.toml CLAUDE.md
git commit -m "chore: project setup with ltx-pipelines path deps"
```

---

## Task 2: Config Module

**Files:**
- Create: `config.py`

**Step 1: Write config.py**

```python
from pathlib import Path

# Checkpoint paths
CHECKPOINTS_DIR = Path("/mnt/nvme-1/huggingface/ltx-2.3-checkpoints")
DISTILLED_CHECKPOINT = str(CHECKPOINTS_DIR / "ltx-2.3-22b-distilled.safetensors")
DEV_CHECKPOINT = str(CHECKPOINTS_DIR / "ltx-2.3-22b-dev.safetensors")
DISTILLED_LORA = str(CHECKPOINTS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors")
SPATIAL_UPSAMPLER = str(CHECKPOINTS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

# Text encoder
GEMMA_ROOT = "/mnt/nvme-1/huggingface/hub/models--google--gemma-3-12b-pt"

# GPU devices for inference (both RTX PRO 6000 Blackwell 96GB)
GPU_DEVICES = ["cuda:0", "cuda:2"]

# Upload storage
UPLOAD_DIR = Path("/mnt/nvme-1/servers/taco-backend/uploads")

# Server
HOST = "0.0.0.0"
PORT = 8080
```

**Step 2: Commit**

```bash
git add config.py
git commit -m "feat: add config module with model paths and GPU mapping"
```

---

## Task 3: Upload Store

**Files:**
- Create: `upload_store.py`
- Create: `tests/test_upload_store.py`

**Step 1: Write the failing test**

```python
import tempfile
from pathlib import Path
from upload_store import UploadStore


def test_create_returns_uuid_and_storage_uri():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UploadStore(Path(tmpdir))
        upload_id, storage_uri = store.create()
        assert storage_uri.startswith("storage://")
        assert upload_id in storage_uri


def test_save_and_resolve():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UploadStore(Path(tmpdir))
        upload_id, storage_uri = store.create()
        store.save(upload_id, b"fake image data")
        resolved = store.resolve(storage_uri)
        assert resolved.exists()
        assert resolved.read_bytes() == b"fake image data"


def test_resolve_unknown_uri_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UploadStore(Path(tmpdir))
        try:
            store.resolve("storage://nonexistent")
            assert False, "Should have raised"
        except FileNotFoundError:
            pass
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_upload_store.py -v`
Expected: FAIL — `upload_store` module not found

**Step 3: Write upload_store.py**

```python
import uuid
from pathlib import Path


class UploadStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(self) -> tuple[str, str]:
        upload_id = uuid.uuid4().hex
        return upload_id, f"storage://{upload_id}"

    def save(self, upload_id: str, data: bytes) -> Path:
        path = self.base_dir / upload_id
        path.write_bytes(data)
        return path

    def resolve(self, storage_uri: str) -> Path:
        if not storage_uri.startswith("storage://"):
            raise ValueError(f"Invalid storage URI: {storage_uri}")
        upload_id = storage_uri[len("storage://"):]
        path = self.base_dir / upload_id
        if not path.exists():
            raise FileNotFoundError(f"Upload not found: {storage_uri}")
        return path
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_upload_store.py -v`
Expected: all 3 PASS

**Step 5: Commit**

```bash
git add upload_store.py tests/test_upload_store.py
git commit -m "feat: add upload store with UUID-based storage:// URIs"
```

---

## Task 4: Pipeline Manager (Dual-GPU Dispatch)

**Files:**
- Create: `pipeline_manager.py`

This is the core module. It:
1. Loads pipeline instances onto two GPUs at startup
2. Uses an `asyncio.Queue` to dispatch requests to whichever GPU is free
3. Runs inference in a thread pool (since pipeline `__call__` is blocking)

**Step 1: Write pipeline_manager.py**

```python
import asyncio
import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import torch

from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import Audio
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.retake import RetakePipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
from ltx_pipelines.utils.media_io import encode_video, load_image_conditioning, decode_audio_from_file
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import detect_params

import config

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    FAST = "ltx-2-3-fast"
    PRO = "ltx-2-3-pro"


@dataclass
class GPUWorker:
    device: torch.device
    fast_pipeline: DistilledPipeline
    pro_pipeline: TI2VidTwoStagesPipeline
    retake_pipeline: RetakePipeline
    a2v_pipeline: A2VidPipelineTwoStage
    lock: asyncio.Lock


def _resolution_to_dims(resolution: str) -> tuple[int, int]:
    w, h = map(int, resolution.split("x"))
    return w, h


def _duration_to_frames(duration: float, fps: float) -> int:
    num_frames = round(duration * fps)
    # Snap to 8k+1 constraint
    num_frames = ((num_frames - 1) // 8) * 8 + 1
    return max(num_frames, 9)  # minimum 9 frames (1*8+1)


def _video_to_bytes(video: torch.Tensor | Iterator[torch.Tensor], fps: float, audio: Audio | None, num_frames: int) -> bytes:
    buf = io.BytesIO()
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    encode_video(
        video=video,
        fps=int(fps),
        audio=audio,
        output_path=buf,
        video_chunks_number=video_chunks_number,
    )
    buf.seek(0)
    return buf.read()


def _load_worker(device_str: str) -> GPUWorker:
    device = torch.device(device_str)
    logger.info(f"Loading pipelines on {device_str}...")

    distilled_lora = LoraPathStrengthAndSDOps(
        path=config.DISTILLED_LORA,
        strength=1.0,
    )

    fast_pipeline = DistilledPipeline(
        distilled_checkpoint_path=config.DISTILLED_CHECKPOINT,
        gemma_root=config.GEMMA_ROOT,
        spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
        loras=(),
        device=device,
    )

    pro_pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=config.DEV_CHECKPOINT,
        distilled_lora=[distilled_lora],
        spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
        gemma_root=config.GEMMA_ROOT,
        loras=(),
        device=device,
    )

    retake_pipeline = RetakePipeline(
        checkpoint_path=config.DEV_CHECKPOINT,
        gemma_root=config.GEMMA_ROOT,
        loras=(),
        device=device,
    )

    a2v_pipeline = A2VidPipelineTwoStage(
        checkpoint_path=config.DEV_CHECKPOINT,
        distilled_lora=[distilled_lora],
        spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
        gemma_root=config.GEMMA_ROOT,
        loras=(),
        device=device,
    )

    logger.info(f"Pipelines loaded on {device_str}")
    return GPUWorker(
        device=device,
        fast_pipeline=fast_pipeline,
        pro_pipeline=pro_pipeline,
        retake_pipeline=retake_pipeline,
        a2v_pipeline=a2v_pipeline,
        lock=asyncio.Lock(),
    )


class PipelineManager:
    def __init__(self) -> None:
        self.workers: list[GPUWorker] = []

    def load_all(self) -> None:
        for device_str in config.GPU_DEVICES:
            worker = _load_worker(device_str)
            self.workers.append(worker)
        logger.info(f"All workers loaded: {len(self.workers)} GPUs ready")

    async def _acquire_worker(self) -> GPUWorker:
        """Return the first unlocked worker, or wait for one."""
        while True:
            for worker in self.workers:
                if not worker.lock.locked():
                    await worker.lock.acquire()
                    return worker
            # All busy — wait briefly and retry
            await asyncio.sleep(0.1)

    async def generate_text_to_video(
        self,
        prompt: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        camera_motion: str | None = None,
        seed: int = -1,
    ) -> bytes:
        width, height = _resolution_to_dims(resolution)
        num_frames = _duration_to_frames(duration, fps)

        effective_prompt = prompt
        if camera_motion:
            effective_prompt = f"{prompt}, camera motion: {camera_motion}"

        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_event_loop()
            video_bytes = await loop.run_in_executor(
                None,
                _run_t2v,
                worker,
                effective_prompt,
                model,
                width,
                height,
                num_frames,
                fps,
                seed,
            )
            return video_bytes
        finally:
            worker.lock.release()

    async def generate_image_to_video(
        self,
        prompt: str,
        image_path: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        seed: int = -1,
    ) -> bytes:
        width, height = _resolution_to_dims(resolution)
        num_frames = _duration_to_frames(duration, fps)

        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_event_loop()
            video_bytes = await loop.run_in_executor(
                None,
                _run_i2v,
                worker,
                prompt,
                image_path,
                model,
                width,
                height,
                num_frames,
                fps,
                seed,
            )
            return video_bytes
        finally:
            worker.lock.release()

    async def generate_audio_to_video(
        self,
        prompt: str,
        audio_path: str,
        image_path: str | None,
        model: str,
        resolution: str,
        seed: int = -1,
    ) -> bytes:
        width, height = _resolution_to_dims(resolution)

        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_event_loop()
            video_bytes = await loop.run_in_executor(
                None,
                _run_a2v,
                worker,
                prompt,
                audio_path,
                image_path,
                model,
                width,
                height,
                seed,
            )
            return video_bytes
        finally:
            worker.lock.release()

    async def retake(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        mode: str,
        prompt: str | None = None,
        seed: int = -1,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_event_loop()
            video_bytes = await loop.run_in_executor(
                None,
                _run_retake,
                worker,
                video_path,
                start_time,
                duration,
                mode,
                prompt or "",
                seed,
            )
            return video_bytes
        finally:
            worker.lock.release()


@torch.inference_mode()
def _run_t2v(
    worker: GPUWorker,
    prompt: str,
    model: str,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    seed: int,
) -> bytes:
    effective_seed = seed if seed >= 0 else torch.randint(0, 2**31, (1,)).item()
    tiling_config = TilingConfig.default()

    if model == ModelType.FAST:
        video, audio = worker.fast_pipeline(
            prompt=prompt,
            seed=effective_seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            images=[],
            tiling_config=tiling_config,
        )
    else:
        video, audio = worker.pro_pipeline(
            prompt=prompt,
            seed=effective_seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            images=[],
            tiling_config=tiling_config,
        )

    return _video_to_bytes(video, fps, audio, num_frames)


@torch.inference_mode()
def _run_i2v(
    worker: GPUWorker,
    prompt: str,
    image_path: str,
    model: str,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    seed: int,
) -> bytes:
    effective_seed = seed if seed >= 0 else torch.randint(0, 2**31, (1,)).item()
    tiling_config = TilingConfig.default()
    images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]

    if model == ModelType.FAST:
        video, audio = worker.fast_pipeline(
            prompt=prompt,
            seed=effective_seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            images=images,
            tiling_config=tiling_config,
        )
    else:
        video, audio = worker.pro_pipeline(
            prompt=prompt,
            seed=effective_seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            images=images,
            tiling_config=tiling_config,
        )

    return _video_to_bytes(video, fps, audio, num_frames)


@torch.inference_mode()
def _run_a2v(
    worker: GPUWorker,
    prompt: str,
    audio_path: str,
    image_path: str | None,
    model: str,
    width: int,
    height: int,
    seed: int,
) -> bytes:
    effective_seed = seed if seed >= 0 else torch.randint(0, 2**31, (1,)).item()
    tiling_config = TilingConfig.default()

    # Read audio to determine duration and frame count
    audio_input = decode_audio_from_file(audio_path, worker.device)
    duration = audio_input.waveform.shape[-1] / audio_input.sampling_rate
    fps = 24.0
    num_frames = _duration_to_frames(duration, fps)

    images = []
    if image_path:
        images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]

    video, audio = worker.a2v_pipeline(
        prompt=prompt,
        seed=effective_seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=fps,
        images=images,
        audio_path=audio_path,
        tiling_config=tiling_config,
    )

    return _video_to_bytes(video, fps, audio, num_frames)


@torch.inference_mode()
def _run_retake(
    worker: GPUWorker,
    video_path: str,
    start_time: float,
    duration: float,
    mode: str,
    prompt: str,
    seed: int,
) -> bytes:
    effective_seed = seed if seed >= 0 else torch.randint(0, 2**31, (1,)).item()
    end_time = start_time + duration
    tiling_config = TilingConfig.default()

    regenerate_video = mode in ("replace_audio_and_video", "replace_video", "replace_video_only")
    regenerate_audio = mode in ("replace_audio_and_video", "replace_audio")

    params = detect_params(config.DEV_CHECKPOINT)

    video, audio = worker.retake_pipeline(
        video_path=video_path,
        prompt=prompt,
        start_time=start_time,
        end_time=end_time,
        seed=effective_seed,
        regenerate_video=regenerate_video,
        regenerate_audio=regenerate_audio,
        video_guider_params=params.video_guider_params,
        audio_guider_params=params.audio_guider_params,
        tiling_config=tiling_config,
    )

    from ltx_pipelines.utils.media_io import get_videostream_metadata
    fps, num_frames, _, _ = get_videostream_metadata(video_path)

    return _video_to_bytes(video, fps, audio, num_frames)
```

**Important notes for the implementer:**
- `encode_video` in `ltx_pipelines` currently writes to a file path string. We pass a `BytesIO` — if this fails, we'll need to write to a temp file and read back. Verify this during testing.
- The `DistilledPipeline` and `TI2VidTwoStagesPipeline` constructors load model *builders* (via `ModelLedger`) — they don't load weights to GPU yet. Weights are loaded lazily on first `__call__`. This means startup is fast but first request per GPU is slow.
- Camera motion is appended to the prompt as text — this matches how the official LTX API handles it.

**Step 2: Commit**

```bash
git add pipeline_manager.py
git commit -m "feat: add pipeline manager with dual-GPU dispatch"
```

---

## Task 5: FastAPI Server (All Endpoints)

**Files:**
- Create: `server.py`

**Step 1: Write server.py**

```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
from pipeline_manager import PipelineManager
from upload_store import UploadStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

manager = PipelineManager()
store = UploadStore(config.UPLOAD_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.load_all()
    yield


app = FastAPI(title="taco-backend", lifespan=lifespan)


# --- Request Models ---

class TextToVideoRequest(BaseModel):
    prompt: str
    model: str = "ltx-2-3-fast"
    resolution: str = "1920x1080"
    duration: float = 6.0
    fps: float = 24.0
    generate_audio: bool = False
    camera_motion: str | None = None


class ImageToVideoRequest(BaseModel):
    prompt: str
    image_uri: str
    model: str = "ltx-2-3-fast"
    resolution: str = "1920x1080"
    duration: float = 6.0
    fps: float = 24.0
    generate_audio: bool = False


class AudioToVideoRequest(BaseModel):
    prompt: str
    audio_uri: str
    image_uri: str | None = None
    model: str = "ltx-2-3-fast"
    resolution: str = "1920x1080"


class RetakeRequest(BaseModel):
    video_uri: str
    start_time: float
    duration: float
    mode: str = "replace_audio_and_video"
    prompt: str | None = None


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/text-to-video")
async def text_to_video(req: TextToVideoRequest):
    try:
        video_bytes = await manager.generate_text_to_video(
            prompt=req.prompt,
            model=req.model,
            resolution=req.resolution,
            duration=req.duration,
            fps=req.fps,
            generate_audio=req.generate_audio,
            camera_motion=req.camera_motion,
        )
        return Response(content=video_bytes, media_type="video/mp4")
    except Exception:
        logger.exception("text-to-video failed")
        return JSONResponse(status_code=500, content={"error": "Generation failed"})


@app.post("/v1/image-to-video")
async def image_to_video(req: ImageToVideoRequest):
    try:
        image_path = str(store.resolve(req.image_uri))
        video_bytes = await manager.generate_image_to_video(
            prompt=req.prompt,
            image_path=image_path,
            model=req.model,
            resolution=req.resolution,
            duration=req.duration,
            fps=req.fps,
            generate_audio=req.generate_audio,
        )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Image not found"})
    except Exception:
        logger.exception("image-to-video failed")
        return JSONResponse(status_code=500, content={"error": "Generation failed"})


@app.post("/v1/audio-to-video")
async def audio_to_video(req: AudioToVideoRequest):
    try:
        audio_path = str(store.resolve(req.audio_uri))
        image_path = str(store.resolve(req.image_uri)) if req.image_uri else None
        video_bytes = await manager.generate_audio_to_video(
            prompt=req.prompt,
            audio_path=audio_path,
            image_path=image_path,
            model=req.model,
            resolution=req.resolution,
        )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Upload not found"})
    except Exception:
        logger.exception("audio-to-video failed")
        return JSONResponse(status_code=500, content={"error": "Generation failed"})


@app.post("/v1/retake")
async def retake(req: RetakeRequest):
    try:
        video_path = str(store.resolve(req.video_uri))
        video_bytes = await manager.retake(
            video_path=video_path,
            start_time=req.start_time,
            duration=req.duration,
            mode=req.mode,
            prompt=req.prompt,
        )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    except Exception:
        logger.exception("retake failed")
        return JSONResponse(status_code=422, content={"error": "Content rejected or generation failed"})


@app.post("/v1/upload")
async def create_upload():
    upload_id, storage_uri = store.create()
    host = f"http://{config.HOST}:{config.PORT}"
    return {
        "upload_url": f"{host}/uploads/put/{upload_id}",
        "storage_uri": storage_uri,
        "required_headers": {},
    }


@app.put("/uploads/put/{upload_id}")
async def put_upload(upload_id: str, request: Request):
    body = await request.body()
    store.save(upload_id, body)
    return Response(status_code=201)
```

**Step 2: Commit**

```bash
git add server.py
git commit -m "feat: add FastAPI server with all taco-desktop endpoints"
```

---

## Task 6: Integration Test — Health + Upload

**Files:**
- Create: `tests/test_server.py`

**Step 1: Write basic integration tests**

```python
import pytest
from fastapi.testclient import TestClient

# Import with pipeline loading disabled for unit tests
import config
config.GPU_DEVICES = []  # Skip GPU loading in tests

from server import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_flow():
    # Step 1: Create upload
    response = client.post("/v1/upload")
    assert response.status_code == 200
    data = response.json()
    assert "upload_url" in data
    assert "storage_uri" in data
    assert data["storage_uri"].startswith("storage://")

    # Step 2: PUT the file
    upload_url = data["upload_url"]
    # Extract path from URL
    path = "/" + "/".join(upload_url.split("/")[3:])
    response = client.put(path, content=b"fake image bytes", headers={"Content-Type": "image/png"})
    assert response.status_code == 201
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (health and upload tests work without GPU)

**Step 3: Commit**

```bash
git add tests/test_server.py
git commit -m "test: add health and upload integration tests"
```

---

## Task 7: Smoke Test — Full Pipeline on GPU

**Files:**
- Create: `tests/test_smoke.py`

This test requires GPU and runs actual inference. Mark it so it can be skipped in CI.

**Step 1: Write smoke test**

```python
"""Smoke test — requires GPU. Run with: uv run pytest tests/test_smoke.py -v -s"""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("No CUDA GPU available", allow_module_level=True)

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_text_to_video_fast():
    response = client.post("/v1/text-to-video", json={
        "prompt": "a red ball bouncing on a white floor",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 6.0,
        "fps": 24.0,
        "generate_audio": False,
    })
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 1000  # Should be a real video
```

**Step 2: Run smoke test**

Run: `uv run pytest tests/test_smoke.py -v -s`
Expected: PASS with actual video bytes returned

This is the first real end-to-end validation. If `encode_video` doesn't accept `BytesIO`, fix `_video_to_bytes` in pipeline_manager.py to write to a temp file instead.

**Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: add GPU smoke test for text-to-video"
```

---

## Task 8: Validate All Endpoints Against Spec

**Files:**
- Read: `ltx-compatible-server.md`

Manual verification checklist:

1. `POST /v1/text-to-video` — all fields from spec accepted, returns raw MP4
2. `POST /v1/image-to-video` — resolves `image_uri` from upload store
3. `POST /v1/audio-to-video` — resolves `audio_uri` and optional `image_uri`
4. `POST /v1/retake` — resolves `video_uri`, handles all mode values
5. `POST /v1/upload` — returns `upload_url`, `storage_uri`, `required_headers`
6. `PUT /uploads/put/{id}` — accepts raw bytes, returns 201
7. `GET /health` — returns `{"status": "ok"}`
8. Error responses include `error` field
9. 422 on retake failure (content safety)

**Step 1: Review server.py against spec, fix any gaps**

**Step 2: Commit any fixes**

```bash
git commit -am "fix: align endpoints with taco-desktop API contract"
```

---

## Task 9: Documentation & Run Script

**Files:**
- Create: `run.sh`

**Step 1: Write run.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run uvicorn server:app --host 0.0.0.0 --port 8080
```

**Step 2: Make executable and commit**

```bash
chmod +x run.sh
git add run.sh
git commit -m "chore: add run script"
```

---

## Dependency Graph

```
Task 1 (setup) → Task 2 (config) → Task 3 (upload store)
                                  → Task 4 (pipeline manager) → Task 5 (server) → Task 6 (tests) → Task 7 (smoke) → Task 8 (validate) → Task 9 (docs)
```

Tasks 3 and 4 are independent of each other and can run in parallel after Task 2.
