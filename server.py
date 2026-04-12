"""FastAPI server implementing the LTX-compatible API for taco-desktop."""

from __future__ import annotations

import asyncio
import time
import torch
import logging
import random
import secrets as _secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import config
from split_model_manager import SplitModelManager
from flux_manager import FluxManager, FluxLoraError
from joyai_client import joyai, JoyAIError
from ace_client import ace, AceError
from chat_manager import ChatManager
from helpers import _duration_to_frames, _resolution_to_dims
from upload_store import UploadStore
from lora_registry import LoRARegistry
from flux_lora_registry import FluxLoRARegistry
from job_queue import (
    Job, JobStatus, JobType, JobStore, make_job_id, make_flux_callback,
    worker_loop, cleanup_loop,
    BatchStatus, BatchItemResult, BatchJob, BatchStore, make_batch_id,
    _MEDIA_TYPES as _JOB_MEDIA_TYPES,
)
from history_store import HistoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

manager = SplitModelManager()
flux = FluxManager()
uploads = UploadStore(config.UPLOAD_DIR)
lora_registry = LoRARegistry(config.LORAS_DIR)
flux_lora_registry = FluxLoRARegistry(config.FLUX_LORAS_DIR)
chat = ChatManager()
history = HistoryStore()

# Shared inference lock: FP8 layerwise casting in diffusers causes CUBLAS_STATUS_INTERNAL_ERROR
# when Flux and LTX run CUDA inference concurrently in the same process.
_inference_lock = asyncio.Lock()
_paused = False

# Job queue
job_store = JobStore()
_job_queue: asyncio.Queue[str] = asyncio.Queue()

# Batch queue
batch_store = BatchStore()
_batch_queue: asyncio.Queue[str] = asyncio.Queue()

# Turbo mode — dual-GPU LTX inference (2 concurrent video jobs)
_turbo_active: bool = False
_turbo_worker_task: asyncio.Task | None = None  # second worker_loop for concurrent dispatch

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


# Two-tenant mutual exclusion on cuda:0 (v1.2): LTX, Flux. JoyAI and ACE are
# on cuda:1 sidecars — no swap needed. `_last_gpu_tenant` tracks the current
# holder; `_evict_other_tenants(new)` idempotently flips from any state to `new`.
# Must be called while holding `_inference_lock`.
_last_gpu_tenant: str | None = None  # "ltx" | "flux" | None


async def _evict_other_tenants(new: str) -> None:
    """Evict any tenant that isn't `new`. Caller MUST hold _inference_lock."""
    global _last_gpu_tenant
    if _last_gpu_tenant == "ltx" and new != "ltx":
        logger.info("Auto-swap: evicting LTX from %s for %s", config.LTX_DEVICE, new)
        manager.evict_all()
    if _last_gpu_tenant == "flux" and new != "flux":
        logger.info("Auto-swap: unloading Flux from %s for %s", config.FLUX_DEVICE, new)
        flux.unload()
    _last_gpu_tenant = new


async def _ensure_ltx_resident() -> None:
    """Ensure LTX is loaded on cuda:0. Caller must hold _inference_lock."""
    await _evict_other_tenants("ltx")
    if not manager.is_ready:
        logger.info("Auto-swap: loading LTX on %s", config.LTX_DEVICE)
        manager.load_all()


async def _ensure_flux_ready() -> None:
    """Ensure Flux is ready (pipeline exists) on cuda:0. Caller must hold _inference_lock."""
    await _evict_other_tenants("flux")
    if not flux.is_ready:
        logger.info("Auto-swap: loading Flux on %s", config.FLUX_DEVICE)
        flux.load()


async def _run_music_job(job: Job) -> None:
    """Standalone async task for music on cuda:1. No _inference_lock needed."""
    job.status = JobStatus.PROCESSING
    job.started_at = time.monotonic()
    job.phase = "generating"
    result_bytes: bytes | None = None
    try:
        p = job.params
        ace_params = _build_ace_params(p)
        est = _estimate_music_time(p)

        def on_progress(elapsed: float) -> None:
            if job.status == JobStatus.CANCELLED:
                return
            job.progress = min(elapsed / max(est, 1), 0.90)

        result_bytes = await ace.generate(params=ace_params, on_progress=on_progress)
        job.phase = "saving"
        job.progress = 0.99
        upload_id, storage_uri = uploads.create()
        uploads.save(upload_id, result_bytes)
        job.result_uri = storage_uri
        job.result_media_type = _AUDIO_MEDIA_TYPES.get(p.get("audio_format", "mp3"), "audio/mpeg")
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.phase = None
        elapsed = time.monotonic() - (job.started_at or job.created_at)
        logger.info("Music job %s completed in %.1fs", job.id, elapsed)
    except AceError as exc:
        job.status = JobStatus.FAILED
        job.phase = None
        job.error = str(exc)[:500]
        job.error_code = "ace_error"
        logger.exception("Music job %s failed", job.id)
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.phase = None
        job.error = str(exc)[:500]
        job.error_code = "generation_failed"
        logger.exception("Music job %s failed", job.id)
    finally:
        job.completed_at = time.monotonic()
        if history and job.api_key and result_bytes is not None:
            _params = job.params or {}
            _captured = dict(
                job_id=job.id, api_key=job.api_key, job_type=job.type,
                prompt=_params.get("prompt", ""), model=None,
                width=0, height=0, turbo=False,
                status=job.status, result_uri=job.result_uri,
                result_bytes=result_bytes, created_at=time.time(),
                completed_at=time.time(), error=job.error,
            )

            async def _save():
                try:
                    await asyncio.to_thread(history.save, **_captured)
                except Exception:
                    logger.warning("Failed to save music job %s to history", job.id, exc_info=True)

            asyncio.create_task(_save())


async def _dispatch_job(job: Job) -> bytes:
    """Route a job to the correct manager and return result bytes."""
    if _paused:
        raise RuntimeError("System is paused for maintenance")
    p = job.params

    def on_progress(progress: float, phase: str | None = None) -> None:
        # Guard against stale callbacks firing after a DELETE /v2/jobs/{id}
        # cancel — the denoiser runs in an executor and can tick one or two
        # more times before it observes the cancelled flag.
        if job.status == JobStatus.CANCELLED:
            return
        job.progress = progress
        if phase is not None:
            job.phase = phase

    match job.type:
        case JobType.TEXT_TO_VIDEO:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            return await manager.generate_text_to_video(**p, on_progress=on_progress)
        case JobType.IMAGE_TO_VIDEO:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            return await manager.generate_image_to_video(**p, on_progress=on_progress)
        case JobType.AUDIO_TO_VIDEO:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            return await manager.generate_audio_to_video(**p, on_progress=on_progress)
        case JobType.RETAKE:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            return await manager.retake(**p, on_progress=on_progress)
        case JobType.TEXT_TO_IMAGE:
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            cb = make_flux_callback(job, p.get("num_inference_steps", 50))
            return await flux.generate_text_to_image(**p, callback_on_step_end=cb, phase_sink=on_progress)
        case JobType.IMAGE_TO_IMAGE:
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            cb = make_flux_callback(job, p.get("num_inference_steps", 50))
            return await flux.generate_image_to_image(**p, callback_on_step_end=cb, phase_sink=on_progress)
        case JobType.IMAGE_EDIT:
            model = p.get("model", "flux2-klein")
            if model == "joyai-edit":
                # JoyAI is on cuda:1 sidecar -- no GPU swap needed
                if _turbo_active:
                    raise JoyAIError("turbo_mode_active: JoyAI unavailable while turbo is enabled", 503)
                if not config.LOAD_JOYAI:
                    raise JoyAIError("joyai_disabled: set LOAD_JOYAI=1 to enable", 503)
                # joyai-edit: exactly one image_path, ignores lora, chat-template
                # prompt wrap is server-side in the sidecar.
                image_paths = p.get("image_paths") or []
                if len(image_paths) != 1:
                    raise ValueError(
                        f"joyai-edit requires exactly one image_uri, got {len(image_paths)}"
                    )
                # Phase callback: JoyAI is a single opaque HTTP call so we just
                # set one phase transition. Client UIs will show "encoding" for
                # the whole duration.
                on_progress(0.90, phase="encoding")
                return await joyai.edit(
                    prompt=p["prompt"],
                    image_path=image_paths[0],
                    width=p["width"],
                    height=p["height"],
                    num_inference_steps=p.get("num_inference_steps", 30),
                    guidance_scale=p.get("guidance_scale", 4.0),
                    seed=p.get("seed"),
                )
            else:
                await _ensure_flux_ready()
                torch.cuda.set_device(config.FLUX_DEVICE)
                cb = make_flux_callback(job, p.get("num_inference_steps", 4))
                return await flux.generate_image_edit(**p, callback_on_step_end=cb, phase_sink=on_progress)
        case JobType.EXPORT_COMPOSITION:
            from export_handler import export_composition
            return await asyncio.get_running_loop().run_in_executor(
                None, lambda: export_composition(p["clips"], p["transitions"], uploads)
            )
        case _:
            raise ValueError(f"Unknown job type: {job.type}")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _last_gpu_tenant
    logger.info("Loading LTX pipelines on %s ...", config.GPU_DEVICES)
    manager.load_all()
    logger.info("LTX pipelines ready.")
    _last_gpu_tenant = "ltx"

    if config.LOAD_FLUX:
        logger.info("Loading Flux pipeline on %s ...", config.FLUX_DEVICE)
        flux.load()
        logger.info("Flux pipeline ready.")
        # Flux uses enable_model_cpu_offload, so its weights live on pinned CPU
        # at idle. GPU tenancy stays "ltx" until an actual Flux forward pass.
    else:
        logger.info("Flux loading disabled (LOAD_FLUX not set)")

    chat.load()
    logger.info("Chat proxy ready.")

    if config.LOAD_JOYAI:
        try:
            health = await joyai.health()
            logger.info("JoyAI sidecar reachable: %s", health)
        except Exception as exc:
            logger.warning(
                "JoyAI sidecar unreachable at %s — joyai-edit will return 503: %s",
                config.JOYAI_SIDECAR_URL,
                exc,
            )

    if config.LOAD_ACE:
        for attempt in range(3):
            try:
                ace_health = await ace.health()
                logger.info("ACE sidecar reachable: %s", ace_health)
                break
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    logger.warning("ACE sidecar unreachable at %s: %s", config.ACE_SIDECAR_URL, exc)

    worker_task = asyncio.create_task(
        worker_loop(job_store, _job_queue, _inference_lock, _dispatch_job, uploads, history,
                    turbo_check=lambda: _turbo_active),
        name="queue-worker",
    )
    cleanup_task = asyncio.create_task(
        cleanup_loop(job_store, uploads),
        name="queue-cleanup",
    )
    batch_worker_task = asyncio.create_task(
        batch_worker(),
        name="batch-worker",
    )
    batch_cleanup_task = asyncio.create_task(
        batch_cleanup_loop(),
        name="batch-cleanup",
    )
    logger.info("Job queue + batch worker started.")

    yield

    worker_task.cancel()
    cleanup_task.cancel()
    batch_worker_task.cancel()
    batch_cleanup_task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|192\.168\.\d+\.\d+)(:\d+)?$",
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if not config.API_KEYS:
        return await call_next(request)
    if request.url.path in ("/health", "/v1/approved-images/events", "/dashboard", "/v1/system/gpu"):
        return await call_next(request)
    # SSE job streams: EventSource can't set custom headers, so these endpoints
    # accept a `?token=` query param and do their own auth inside the handler.
    if request.url.path.startswith("/v2/jobs/") and request.url.path.endswith("/stream"):
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""

    if not token or not any(
        _secrets.compare_digest(token, key) for key in config.API_KEYS
    ):
        return _error(401, "Invalid or missing API key")

    return await call_next(request)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


ModelName = Literal["ltx-2-3-fast", "ltx-2-3-pro", "ltx-2-3-hq"]


class LoRAInput(BaseModel):
    id: str = Field(description="LoRA ID from /v1/loras")
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
Resolution = Literal["1920x1080", "1080x1920", "2560x1440", "1440x2560", "3840x2160", "2160x3840"]
RetakeMode = Literal["replace_audio_and_video", "replace_video", "replace_video_only", "replace_audio"]
ImageModelName = Literal["flux2-dev", "flux2-klein", "joyai-edit"]


class TextToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False
    camera_motion: str | None = Field(default=None, max_length=200)
    lora: LoRAInput | None = None
    enhance_prompt: bool = False


class KeyframeInput(BaseModel):
    image_uri: str
    frame_index: int | Literal["first", "middle", "last"] = Field(default=0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class ImageToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str | None = None
    image_strength: float = Field(default=0.85, ge=0.0, le=1.0)
    keyframes: list[KeyframeInput] | None = None
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False
    lora: LoRAInput | None = None
    enhance_prompt: bool = False


class AudioToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    audio_uri: str
    image_uri: str | None = None
    model: ModelName
    resolution: Resolution
    duration: float = Field(default=6.0, gt=0, le=30)
    fps: float = Field(default=24.0, gt=0, le=60)
    lora: LoRAInput | None = None
    enhance_prompt: bool = False


class RetakeRequest(BaseModel):
    video_uri: str
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0, le=30)
    mode: RetakeMode
    prompt: str | None = Field(default=None, max_length=10000)
    lora: LoRAInput | None = None


class TextToImageRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    model: ImageModelName = "flux2-dev"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None
    turbo: bool = False
    lora: LoRAInput | None = None


class ImageToImageRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str
    model: ImageModelName = "flux2-dev"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None
    turbo: bool = False
    lora: LoRAInput | None = None


class ImageEditRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uris: list[str] = Field(min_length=1, max_length=10)
    model: ImageModelName = "flux2-klein"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None
    lora: LoRAInput | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list  # str for text, list for multimodal


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3-12b-nvfp4"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)


class CharRankRequest(BaseModel):
    rank_image_uri: str
    generated_image_uri: str
    prompt: str = Field(max_length=10000)


class MusicGenerationRequest(BaseModel):
    # Core
    prompt: str = Field(max_length=10000, description="Music description / caption")
    lyrics: str = Field(default="[Instrumental]", max_length=50000)
    duration: float = Field(default=60.0, gt=0, le=600)
    audio_format: Literal["mp3", "flac", "wav", "wav32", "opus", "aac"] = "mp3"
    seed: int | None = None
    # Music theory
    bpm: int | None = Field(default=None, ge=30, le=300)
    key_scale: str | None = None
    time_signature: str | None = None
    vocal_language: str | None = None
    # Diffusion (xl-base defaults)
    num_inference_steps: int = Field(default=50, ge=1, le=200)
    guidance_scale: float = Field(default=7.0, ge=0, le=15)
    shift: float = Field(default=3.0, ge=1.0, le=5.0)
    infer_method: Literal["ode", "sde"] = "ode"
    use_adg: bool = False
    cfg_interval_start: float = Field(default=0.0, ge=0, le=1)
    cfg_interval_end: float = Field(default=1.0, ge=0, le=1)
    batch_size: int = Field(default=1, ge=1, le=8)
    # Task type
    task_type: Literal["text2music", "cover", "repaint", "extract", "lego", "complete"] = "text2music"
    source_audio_uri: str | None = None
    reference_audio_uri: str | None = None
    audio_cover_strength: float = Field(default=1.0, ge=0, le=1)
    repainting_start: float = Field(default=0.0, ge=0)
    repainting_end: float | None = None
    repaint_mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
    repaint_strength: float = Field(default=0.5, ge=0, le=1)
    track_name: str | None = None
    # LM / thinking
    thinking: bool = False
    sample_mode: bool = False
    sample_query: str | None = None
    lm_temperature: float = Field(default=0.85, ge=0, le=2)
    lm_top_p: float = Field(default=0.9, ge=0, le=1)


_AUDIO_MEDIA_TYPES = {
    "mp3": "audio/mpeg", "flac": "audio/flac", "wav": "audio/wav",
    "wav32": "audio/wav", "opus": "audio/opus", "aac": "audio/aac",
}


# ---------------------------------------------------------------------------
# Batch request models
# ---------------------------------------------------------------------------

_BATCH_ITEM_TYPES = Literal[
    "text-to-image", "image-to-image", "image-edit",
    "text-to-video", "image-to-video",
]


class BatchItem(BaseModel):
    """One generation request inside a batch."""
    type: _BATCH_ITEM_TYPES
    params: dict[str, Any]


class BatchRequest(BaseModel):
    """Submit a batch of generation jobs."""
    items: list[BatchItem] = Field(..., min_length=1, max_length=50)
    priority: Literal["normal", "high"] = "normal"
    callback_url: str | None = None


# Map batch item type string -> (JobType, Pydantic validator model)
_BATCH_TYPE_MAP: dict[str, tuple[JobType, type[BaseModel]]] = {
    "text-to-image": (JobType.TEXT_TO_IMAGE, TextToImageRequest),
    "image-to-image": (JobType.IMAGE_TO_IMAGE, ImageToImageRequest),
    "image-edit": (JobType.IMAGE_EDIT, ImageEditRequest),
    "text-to-video": (JobType.TEXT_TO_VIDEO, TextToVideoRequest),
    "image-to-video": (JobType.IMAGE_TO_VIDEO, ImageToVideoRequest),
}


def _is_image_type(type_str: str) -> bool:
    """Return True for image generation types."""
    return type_str in ("text-to-image", "image-to-image", "image-edit")


def _batch_item_to_job(item: BatchItem, api_key: str) -> Job:
    """Create a synthetic Job from a BatchItem for dispatch."""
    job_type, _ = _BATCH_TYPE_MAP[item.type]
    return Job(
        id=make_job_id(),
        type=job_type,
        params=item.params,
        api_key=api_key,
    )


def _build_ace_params(p: dict) -> dict:
    """Translate MusicGenerationRequest params to ACE /release_task fields."""
    ace_p = {
        "caption": p["prompt"],
        "lyrics": p.get("lyrics", "[Instrumental]"),
        "audio_duration": p.get("duration", 60.0),
        "audio_format": p.get("audio_format", "mp3"),
        "inference_steps": p.get("num_inference_steps", 50),
        "guidance_scale": p.get("guidance_scale", 7.0),
        "shift": p.get("shift", 3.0),
        "infer_method": p.get("infer_method", "ode"),
        "use_adg": p.get("use_adg", False),
        "cfg_interval_start": p.get("cfg_interval_start", 0.0),
        "cfg_interval_end": p.get("cfg_interval_end", 1.0),
        "batch_size": p.get("batch_size", 1),
        "task_type": p.get("task_type", "text2music"),
        "audio_cover_strength": p.get("audio_cover_strength", 1.0),
        "thinking": p.get("thinking", False),
        "sample_mode": p.get("sample_mode", False),
    }
    # Seed handling: ACE uses seed=-1 for random
    if p.get("seed") is not None:
        ace_p["seed"] = p["seed"]
        ace_p["use_random_seed"] = False
    else:
        ace_p["seed"] = -1
        ace_p["use_random_seed"] = True
    # Optional fields -- only send if provided
    for key in ("bpm", "key_scale", "time_signature", "vocal_language",
                "sample_query", "repainting_start", "repainting_end",
                "repaint_mode", "repaint_strength", "track_name",
                "lm_temperature", "lm_top_p"):
        if p.get(key) is not None:
            ace_p[key] = p[key]
    # Resolved audio paths (already converted from storage:// URIs by the handler)
    if p.get("source_audio_path"):
        ace_p["src_audio_path"] = p["source_audio_path"]
    if p.get("reference_audio_path"):
        ace_p["reference_audio_path"] = p["reference_audio_path"]
    return ace_p


def _estimate_music_time(p: dict) -> float:
    """Rough estimate of music generation time in seconds."""
    steps = p.get("num_inference_steps", 50)
    dur = p.get("duration", 60.0)
    if steps <= 8:
        return max(5, dur * 0.1)
    elif steps <= 32:
        return max(15, dur * 0.3)
    else:
        return max(25, dur * 0.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt(prompt: str, camera_motion: str | None) -> str:
    """Append camera-motion tag to prompt if specified."""
    if camera_motion:
        return f"{prompt} [{camera_motion}]"
    return prompt


def _resolve_keyframes(body: ImageToVideoRequest, num_frames: int) -> list[dict] | JSONResponse:
    """Resolve keyframes, converting symbolic/negative frame indices to absolute values."""
    if body.keyframes and body.image_uri:
        return _error(422, "Cannot specify both image_uri and keyframes")
    if body.keyframes:
        if len(body.keyframes) == 0:
            return _error(422, "keyframes list must not be empty")
        if len(body.keyframes) > 8:
            return _error(422, "At most 8 keyframes are allowed")
        keyframe_inputs = []
        for kf in body.keyframes:
            fi = kf.frame_index
            if fi == "first":
                fi = 0
            elif fi == "middle":
                fi = num_frames // 2
            elif fi == "last":
                fi = num_frames - 1
            elif isinstance(fi, int) and fi < 0:
                fi = num_frames + fi
            if isinstance(fi, int) and (fi < 0 or (num_frames > 0 and fi >= num_frames)):
                return _error(422, f"Resolved frame_index {fi} is out of range for {num_frames} frames")
            path = str(uploads.resolve(kf.image_uri))
            keyframe_inputs.append({"image_path": path, "frame_index": fi, "strength": kf.strength})
        frame_indices = [kf["frame_index"] for kf in keyframe_inputs]
        if len(frame_indices) != len(set(frame_indices)):
            return _error(422, "Duplicate frame_index values after resolution")
        return keyframe_inputs
    elif body.image_uri:
        path = str(uploads.resolve(body.image_uri))
        return [{"image_path": path, "frame_index": 0, "strength": body.image_strength}]
    else:
        return _error(422, "Either image_uri or keyframes is required")


def _resolve_lora(body) -> tuple[str | None, float] | JSONResponse:
    """Resolve optional LoRA from request. Returns (path, strength) or JSONResponse on error."""
    if not getattr(body, "lora", None):
        return None, 1.0
    info = lora_registry.get(body.lora.id)
    if info is None:
        return _error(404, f"LoRA not found: {body.lora.id}")
    return str(lora_registry.resolve_path(body.lora.id)), body.lora.strength


def _resolve_flux_lora(body) -> tuple[str | None, float] | JSONResponse:
    """Resolve optional Flux LoRA from request. Returns (path, strength) or JSONResponse on error."""
    if not getattr(body, "lora", None):
        return None, 1.0
    info = flux_lora_registry.get(body.lora.id)
    if info is None:
        return _error(404, f"Flux LoRA not found: {body.lora.id}")
    return str(flux_lora_registry.resolve_path(body.lora.id)), body.lora.strength


def _error(status: int, msg: str) -> JSONResponse:
    # Avoid leaking internal filesystem paths in error responses
    text = msg[:500]
    if "/mnt/" in text or "/home/" in text or "/tmp/" in text:
        text = "Internal server error"
    return JSONResponse(status_code=status, content={"error": text, "message": text, "detail": text})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    """Serve the GPU management dashboard (static HTML SPA)."""
    from fastapi.responses import HTMLResponse
    dash_path = Path(__file__).parent / "dashboard.html"
    if not dash_path.exists():
        return _error(404, "dashboard.html not found")
    return HTMLResponse(content=dash_path.read_text(), media_type="text/html")


_gpu_cache: dict | None = None
_gpu_cache_time: float = 0.0


async def _query_gpu_info() -> list[dict]:
    """Run nvidia-smi and parse GPU info."""
    import subprocess
    result = await asyncio.to_thread(
        subprocess.run,
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, timeout=5,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "temperature_c": int(parts[4]) if parts[4] not in ("[N/A]", "[Not Supported]") and parts[4].isdigit() else None,
                "utilization_pct": int(parts[5]) if parts[5] not in ("[N/A]", "[Not Supported]") and parts[5].isdigit() else None,
                "power_draw_w": float(parts[6]) if parts[6] not in ("[N/A]", "[Not Supported]") else None,
            })
        elif len(parts) >= 5:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "temperature_c": int(parts[4]) if parts[4].isdigit() else None,
            })
    return gpus


@app.get("/v1/system/gpu")
async def system_gpu() -> dict:
    """nvidia-smi GPU telemetry for the dashboard (2s cache)."""
    global _gpu_cache, _gpu_cache_time
    now = time.monotonic()
    if _gpu_cache is None or (now - _gpu_cache_time) > 2.0:
        try:
            gpus = await _query_gpu_info()
            _gpu_cache = {
                "gpus": gpus,
                "turbo": _turbo_active,
                "gpu0_tenant": _last_gpu_tenant or "idle",
                "gpu1_tenant": "flux" if _turbo_active else ("ace" if config.LOAD_ACE else "idle"),
            }
            _gpu_cache_time = now
        except Exception as exc:
            logger.warning("nvidia-smi failed: %s", exc)
            return {"gpus": [], "turbo": _turbo_active, "error": str(exc)}
    return _gpu_cache


@app.get("/health")
async def health() -> dict:
    if _paused:
        return {
            "status": "paused",
            "ltx": "paused",
            "flux": "paused",
            "ace": "paused" if config.LOAD_ACE else "disabled",
            "chat": "ready" if chat.is_ready else "not_loaded",
            "queue": job_store.stats(),
        }
    return {
        "status": "ok",
        "ltx": "ready" if manager.is_ready else "not_loaded",
        "flux": "ready" if flux.is_ready else "not_loaded",
        "ace": "enabled" if config.LOAD_ACE else "disabled",
        "chat": "ready" if chat.is_ready else "not_loaded",
        "queue": job_store.stats(),
    }


@app.post("/v1/system/pause")
async def system_pause() -> dict:
    """Evict all models from GPU to free VRAM for training."""
    global _paused, _last_gpu_tenant
    if _paused:
        return {"status": "already_paused"}
    try:
        async with _inference_lock:
            # Exit turbo mode first if active
            if _turbo_active:
                await _exit_turbo_mode()
            # Cancel queued (not yet processing) jobs
            while not _job_queue.empty():
                try:
                    job_id = _job_queue.get_nowait()
                    job = job_store.get(job_id)
                    if job and job.status == JobStatus.QUEUED:
                        job.status = JobStatus.FAILED
                        job.error = "System paused"
                except asyncio.QueueEmpty:
                    break
            # Cancel queued batches
            while not _batch_queue.empty():
                try:
                    bid = _batch_queue.get_nowait()
                    b = batch_store.get(bid)
                    if b and b.status == BatchStatus.QUEUED:
                        b.status = BatchStatus.CANCELLED
                except asyncio.QueueEmpty:
                    break
            _paused = True
            manager.evict_all()
            flux.unload()
            if config.LOAD_JOYAI:
                try:
                    await joyai.unload()
                except Exception:
                    logger.warning("JoyAI unload during pause failed", exc_info=True)
            _last_gpu_tenant = None
        logger.info("System paused — all GPU memory freed")
        return {"status": "paused"}
    except Exception:
        logger.exception("Pause failed")
        _paused = True
        return JSONResponse(status_code=500, content={"error": "pause_failed", "status": "paused"})


@app.post("/v1/system/resume")
async def system_resume() -> dict:
    """Reload all models after training."""
    global _paused, _last_gpu_tenant
    if not _paused:
        return {"status": "already_running"}
    try:
        async with _inference_lock:
            manager.load_all()
            _last_gpu_tenant = "ltx"
            if config.LOAD_FLUX:
                flux.load()
            _paused = False
        logger.info("System resumed — all models reloaded")
        return {"status": "ready"}
    except Exception:
        logger.exception("Resume failed — system remains paused")
        return JSONResponse(status_code=500, content={"error": "resume_failed", "status": "paused"})


@app.post("/v1/flux/unload")
async def flux_unload() -> dict:
    """Unload Flux model from the Flux device to free VRAM for training / vision models."""
    global _last_gpu_tenant
    if not flux.is_ready:
        return {"status": "already_unloaded"}
    try:
        async with _inference_lock:
            flux.unload()
            _last_gpu_tenant = None
        logger.info("Flux unloaded from %s", config.FLUX_DEVICE)
        return {"status": "unloaded"}
    except Exception:
        logger.exception("Flux unload failed")
        return JSONResponse(status_code=500, content={"error": "flux_unload_failed"})


@app.post("/v1/ltx/unload")
async def ltx_unload() -> dict:
    """Unload LTX from the LTX device to free VRAM (e.g. for a training run).

    Unlike /v1/system/pause this touches ONLY LTX — the Flux pipeline stays
    available for image generation. In single-GPU swap mode (LTX_DEVICE ==
    FLUX_DEVICE), this also makes room on the GPU for a subsequent Flux
    forward pass; the next video request will auto-swap LTX back in.
    """
    global _last_gpu_tenant
    if not manager.is_ready:
        return {"status": "already_unloaded"}
    try:
        async with _inference_lock:
            manager.evict_all()
            _last_gpu_tenant = None
        logger.info("LTX unloaded from %s", config.LTX_DEVICE)
        return {"status": "unloaded"}
    except Exception:
        logger.exception("LTX unload failed")
        return JSONResponse(status_code=500, content={"error": "ltx_unload_failed"})


@app.post("/v1/ltx/reload")
async def ltx_reload() -> dict:
    """Reload LTX to the LTX device."""
    global _last_gpu_tenant
    if manager.is_ready:
        return {"status": "already_loaded"}
    try:
        async with _inference_lock:
            manager.load_all()
            _last_gpu_tenant = "ltx"
        logger.info("LTX reloaded to %s", config.LTX_DEVICE)
        return {"status": "loaded"}
    except Exception:
        logger.exception("LTX reload failed")
        return JSONResponse(status_code=500, content={"error": "ltx_reload_failed"})


@app.post("/v1/flux/reload")
async def flux_reload() -> dict:
    """Reload Flux model to the Flux device."""
    global _last_gpu_tenant
    if flux.is_ready:
        return {"status": "already_loaded"}
    try:
        async with _inference_lock:
            flux.load()
            _last_gpu_tenant = "flux"
        logger.info("Flux reloaded to %s", config.FLUX_DEVICE)
        return {"status": "loaded"}
    except Exception:
        logger.exception("Flux reload failed")
        return JSONResponse(status_code=500, content={"error": "flux_reload_failed"})


# ---------------------------------------------------------------------------
# Turbo Mode (dual-GPU) — Phase 3
# ---------------------------------------------------------------------------


def _get_gpu_free_mb(gpu_index: int) -> int:
    """Quick check of free VRAM on a GPU via torch.cuda."""
    try:
        free, total = torch.cuda.mem_get_info(gpu_index)
        return free // (1024 * 1024)
    except Exception:
        return 0


async def _ace_systemctl(action: str) -> None:
    """Start/stop ACE sidecar via systemctl --user."""
    import subprocess
    await asyncio.to_thread(
        subprocess.run,
        ["systemctl", "--user", action, "ace-step"],
        capture_output=True, text=True, timeout=15,
    )


async def _enter_turbo_mode() -> None:
    """Enable dual-GPU LTX: 2 denoiser workers, 2 video jobs at a time.

    cuda:0 gets encoder hub + denoiser worker 0.
    cuda:1 gets denoiser worker 1 (shares encoder hub on cuda:0 for text encoding).
    A second worker_loop is started so the queue can dispatch 2 jobs concurrently.
    SplitModelManager._acquire_worker() handles per-GPU serialization.

    Flux, ACE, and JoyAI are ALL unavailable during turbo — their requests get 503.
    Caller must hold _inference_lock.
    """
    global _turbo_active, _turbo_worker_task, _last_gpu_tenant

    if _turbo_active:
        return  # idempotent

    # Step 1: Evict ACE from cuda:1
    if config.LOAD_ACE:
        try:
            await _ace_systemctl("stop")
            logger.info("Turbo: ACE stopped via systemctl")
        except Exception:
            logger.exception("Turbo: ACE stop failed — aborting turbo entry")
            raise RuntimeError("turbo_entry_failed: could not stop ACE on cuda:1")

    # Step 2: Evict JoyAI from cuda:1
    if config.LOAD_JOYAI:
        try:
            await joyai.unload()
            logger.info("Turbo: JoyAI unloaded from cuda:1")
        except JoyAIError:
            logger.warning("Turbo: JoyAI unload failed — continuing (non-critical)")

    # Step 3: Evict Flux from cuda:0 (LTX needs full GPU for encoder hub + denoiser)
    flux.unload()

    # Step 4: Load dual-GPU LTX — encoder hub on cuda:0, 2 denoiser workers
    manager.evict_all()
    config.GPU_DEVICES = config.TURBO_GPU_DEVICES  # ["cuda:0", "cuda:1"]
    manager.load_all()
    _last_gpu_tenant = "ltx"

    # Step 5: Start a second worker loop so the queue can dispatch 2 jobs concurrently.
    # Both workers pull from _job_queue. In turbo mode, worker_loop skips _inference_lock
    # because SplitModelManager._acquire_worker() serializes per-GPU via worker.lock.
    _turbo_active = True
    _turbo_worker_task = asyncio.create_task(
        worker_loop(job_store, _job_queue, _inference_lock, _dispatch_job, uploads, history,
                    turbo_check=lambda: _turbo_active),
        name="turbo-worker",
    )
    logger.info("TURBO MODE ON: 2 LTX workers on %s, 2 concurrent video jobs", config.TURBO_GPU_DEVICES)


async def _exit_turbo_mode() -> None:
    """Restore single-GPU mode: 1 LTX worker on cuda:0, ACE+JoyAI back on cuda:1.

    Caller must hold _inference_lock.
    """
    global _turbo_active, _turbo_worker_task, _last_gpu_tenant

    if not _turbo_active:
        return  # idempotent

    # Step 1: Cancel the second worker loop
    if _turbo_worker_task is not None:
        _turbo_worker_task.cancel()
        try:
            await _turbo_worker_task
        except asyncio.CancelledError:
            pass
        _turbo_worker_task = None

    # Step 2: Evict dual-GPU LTX, restore single-GPU config
    manager.evict_all()
    config.GPU_DEVICES = config.NORMAL_GPU_DEVICES  # ["cuda:0"]
    manager.load_all()
    _last_gpu_tenant = "ltx"

    # Step 3: Reload ACE on cuda:1
    if config.LOAD_ACE:
        try:
            await _ace_systemctl("start")
            logger.info("Turbo exit: ACE restarted via systemctl")
        except Exception:
            logger.warning("Turbo exit: ACE restart failed — will retry on next request")

    # Step 4: Reload JoyAI on cuda:1
    if config.LOAD_JOYAI:
        try:
            await joyai.load()
            logger.info("Turbo exit: JoyAI reloaded on cuda:1")
        except JoyAIError:
            logger.warning("Turbo exit: JoyAI reload failed — non-critical")

    _turbo_active = False
    logger.info("TURBO MODE OFF: single-GPU swap restored, ACE+JoyAI reloading on cuda:1")


class TurboRequest(BaseModel):
    enable: bool


@app.post("/v1/system/turbo")
async def system_turbo(body: TurboRequest) -> JSONResponse:
    """Toggle turbo mode (claim/release cuda:1 for dual-GPU inference)."""
    if _paused:
        return JSONResponse(
            status_code=503,
            content={"error": "system_paused", "message": "Cannot toggle turbo while paused."},
            headers={"Retry-After": "300"},
        )
    if body.enable and _turbo_active:
        return JSONResponse(status_code=409, content={"error": "already_enabled", "turbo": True})
    if not body.enable and not _turbo_active:
        return JSONResponse(status_code=409, content={"error": "already_disabled", "turbo": False})

    try:
        async with _inference_lock:
            if body.enable:
                await _enter_turbo_mode()
            else:
                await _exit_turbo_mode()
        return JSONResponse(content={
            "turbo": _turbo_active,
            "flux_device": config.FLUX_DEVICE,
            "ltx_device": config.LTX_DEVICE,
            "ace_status": "unloaded" if _turbo_active else ("loaded" if config.LOAD_ACE else "disabled"),
            "joyai_status": "unloaded" if _turbo_active else ("loaded" if config.LOAD_JOYAI else "disabled"),
        })
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    except Exception:
        logger.exception("Turbo toggle failed")
        return JSONResponse(status_code=500, content={"error": "turbo_toggle_failed"})


@app.post("/v1/text-to-video")
async def text_to_video(body: TextToVideoRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    # Auto-swap handles manager.is_ready lazily inside the lock
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    try:
        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        prompt = _build_prompt(body.prompt, body.camera_motion)
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            video_bytes = await manager.generate_text_to_video(
                prompt=prompt,
                model=body.model,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=body.fps,
                seed=seed,
                generate_audio=body.generate_audio,
                lora_path=lora_path,
                lora_strength=lora_strength,
                enhance_prompt=body.enhance_prompt,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except Exception as exc:
        logger.exception("text-to-video failed")
        return _error(500, str(exc))


@app.post("/v1/image-to-video")
async def image_to_video(body: ImageToVideoRequest) -> Response:
    num_frames = _duration_to_frames(body.duration, body.fps)
    keyframe_inputs = _resolve_keyframes(body, num_frames)
    if isinstance(keyframe_inputs, JSONResponse):
        return keyframe_inputs
    if _paused:
        return _error(503, "System is paused for maintenance")
    # Auto-swap handles manager.is_ready lazily inside the lock
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    try:
        width, height = _resolution_to_dims(body.resolution)
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            video_bytes = await manager.generate_image_to_video(
                prompt=body.prompt,
                keyframes=keyframe_inputs,
                model=body.model,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=body.fps,
                seed=seed,
                generate_audio=body.generate_audio,
                lora_path=lora_path,
                lora_strength=lora_strength,
                enhance_prompt=body.enhance_prompt,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("image-to-video failed")
        return _error(500, str(exc))


@app.post("/v1/audio-to-video")
async def audio_to_video(body: AudioToVideoRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    # Auto-swap handles manager.is_ready lazily inside the lock
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    try:
        audio_path = str(uploads.resolve(body.audio_uri))
        image_path: str | None = None
        if body.image_uri:
            image_path = str(uploads.resolve(body.image_uri))

        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            video_bytes = await manager.generate_audio_to_video(
                prompt=body.prompt,
                audio_path=audio_path,
                image_path=image_path,
                model=body.model,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=body.fps,
                seed=seed,
                lora_path=lora_path,
                lora_strength=lora_strength,
                enhance_prompt=body.enhance_prompt,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("audio-to-video failed")
        return _error(500, str(exc))


@app.post("/v1/retake")
async def retake(body: RetakeRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    # Auto-swap handles manager.is_ready lazily inside the lock
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    try:
        video_path = str(uploads.resolve(body.video_uri))
        prompt = body.prompt or ""
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            video_bytes = await manager.retake(
                video_path=video_path,
                start_time=body.start_time,
                duration=body.duration,
                mode=body.mode,
                prompt=prompt,
                seed=seed,
                lora_path=lora_path,
                lora_strength=lora_strength,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("retake failed")
        return _error(422, f"Content rejected or generation failed: {exc}")


@app.post("/v1/text-to-image")
async def text_to_image(body: TextToImageRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    if not config.LOAD_FLUX:
        return _error(500, "Flux not enabled")
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    try:
        width = (body.width // 16) * 16
        height = (body.height // 16) * 16
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            image_bytes = await flux.generate_text_to_image(
                prompt=body.prompt,
                width=width,
                height=height,
                num_inference_steps=body.num_inference_steps,
                guidance_scale=body.guidance_scale,
                seed=seed,
                model=body.model,
                turbo=body.turbo,
                lora_path=lora_path,
                lora_strength=lora_strength,
            )
        return Response(content=image_bytes, media_type="image/webp")
    except FluxLoraError as exc:
        return _error(422, str(exc))
    except Exception as exc:
        logger.exception("text-to-image failed")
        return _error(500, str(exc))


@app.post("/v1/image-to-image")
async def image_to_image(body: ImageToImageRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    if not config.LOAD_FLUX:
        return _error(500, "Flux not enabled")
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    try:
        image_path = str(uploads.resolve(body.image_uri))
        width = (body.width // 16) * 16
        height = (body.height // 16) * 16
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            image_bytes = await flux.generate_image_to_image(
                prompt=body.prompt,
                image_path=image_path,
                width=width,
                height=height,
                num_inference_steps=body.num_inference_steps,
                guidance_scale=body.guidance_scale,
                seed=seed,
                model=body.model,
                turbo=body.turbo,
                lora_path=lora_path,
                lora_strength=lora_strength,
            )
        return Response(content=image_bytes, media_type="image/webp")
    except FluxLoraError as exc:
        return _error(422, str(exc))
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("image-to-image failed")
        return _error(500, str(exc))


@app.post("/v1/image-edit")
async def image_edit(body: ImageEditRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    # joyai-edit routes to the out-of-process sidecar; the flux-backed
    # models still require LOAD_FLUX.
    if body.model == "joyai-edit":
        if _turbo_active:
            return JSONResponse(
                status_code=503,
                content={"error": "turbo_mode_active: ACE/JoyAI unavailable while turbo mode is enabled. Disable turbo first."},
                headers={"Retry-After": "10"},
            )
        if len(body.image_uris) != 1:
            return _error(422, "joyai-edit requires exactly one image_uri")
        if not config.LOAD_JOYAI:
            return _error(503, "JoyAI not enabled (LOAD_JOYAI=0)")
        try:
            image_path = str(uploads.resolve(body.image_uris[0]))
            width = (body.width // 16) * 16
            height = (body.height // 16) * 16
            seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
            # JoyAI is on cuda:1 sidecar -- no inference lock needed
            image_bytes = await joyai.edit(
                prompt=body.prompt,
                image_path=image_path,
                width=width,
                height=height,
                num_inference_steps=body.num_inference_steps,
                guidance_scale=body.guidance_scale,
                seed=seed,
            )
            return Response(content=image_bytes, media_type="image/webp")
        except JoyAIError as exc:
            return _error(exc.status_code, str(exc))
        except FileNotFoundError as exc:
            return _error(404, str(exc))
        except Exception as exc:
            logger.exception("image-edit (joyai) failed")
            return _error(500, str(exc))

    if not config.LOAD_FLUX:
        return _error(500, "Flux pipeline not loaded")
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    try:
        image_paths = [str(uploads.resolve(uri)) for uri in body.image_uris]
        width = (body.width // 16) * 16
        height = (body.height // 16) * 16
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            image_bytes = await flux.generate_image_edit(
                prompt=body.prompt,
                image_paths=image_paths,
                width=width,
                height=height,
                num_inference_steps=body.num_inference_steps,
                guidance_scale=body.guidance_scale,
                seed=seed,
                model=body.model,
                lora_path=lora_path,
                lora_strength=lora_strength,
            )
        return Response(content=image_bytes, media_type="image/webp")
    except FluxLoraError as exc:
        return _error(422, str(exc))
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("image-edit failed")
        return _error(500, str(exc))


@app.post("/v1/music")
async def generate_music(body: MusicGenerationRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    if _turbo_active:
        return JSONResponse(
            status_code=503,
            content={"error": "turbo_mode_active: ACE/JoyAI unavailable while turbo mode is enabled. Disable turbo first."},
            headers={"Retry-After": "10"},
        )
    if not config.LOAD_ACE:
        return _error(503, "Music generation not enabled (LOAD_ACE=0)")
    # Validate task-type requirements
    if body.task_type in ("cover", "repaint", "extract", "lego", "complete") and not body.source_audio_uri:
        return _error(422, f"task_type '{body.task_type}' requires source_audio_uri")
    if body.task_type in ("extract", "lego", "complete") and not body.track_name:
        return _error(422, f"task_type '{body.task_type}' requires track_name")
    # Resolve URIs
    params = body.model_dump(exclude_none=True)
    if body.source_audio_uri:
        try:
            params["source_audio_path"] = str(uploads.resolve(body.source_audio_uri))
        except FileNotFoundError:
            return _error(404, "source_audio_uri not found")
    if body.reference_audio_uri:
        try:
            params["reference_audio_path"] = str(uploads.resolve(body.reference_audio_uri))
        except FileNotFoundError:
            return _error(404, "reference_audio_uri not found")
    params.pop("source_audio_uri", None)
    params.pop("reference_audio_uri", None)
    try:
        ace_params = _build_ace_params(params)
        audio_bytes = await ace.generate(params=ace_params)
        media_type = _AUDIO_MEDIA_TYPES.get(body.audio_format, "audio/mpeg")
        return Response(content=audio_bytes, media_type=media_type)
    except AceError as exc:
        return _error(exc.status_code, str(exc))
    except Exception as exc:
        logger.exception("music generation failed")
        return _error(500, str(exc))


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> JSONResponse:
    if not chat.is_ready:
        return _error(500, "Chat model not loaded")
    if not body.messages:
        return _error(422, "Messages list cannot be empty")
    try:
        messages = [m.model_dump() for m in body.messages]
        result = await chat.generate_chat_completion(
            messages=messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("chat completion failed")
        return _error(500, str(exc))


@app.post("/v1/upload")
async def upload(request: Request) -> JSONResponse:
    upload_id, storage_uri = uploads.create()

    # Build upload_url from the incoming request's host
    base = str(request.base_url).rstrip("/")
    upload_url = f"{base}/uploads/put/{upload_id}"

    return JSONResponse(
        content={
            "upload_url": upload_url,
            "storage_uri": storage_uri,
            "required_headers": {},
        }
    )


@app.put("/uploads/put/{upload_id}")
async def upload_put(upload_id: str, request: Request) -> Response:
    from upload_store import MAX_UPLOAD_BYTES
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        return _error(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit")
    data = await request.body()
    if len(data) > MAX_UPLOAD_BYTES:
        return _error(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit")
    uploads.save(upload_id, data)
    return Response(status_code=201)


# ---------------------------------------------------------------------------
# LoRA Management Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/loras")
async def list_loras() -> JSONResponse:
    loras = lora_registry.list_all()
    return JSONResponse(content={
        "loras": [
            {"id": l.id, "name": l.name, "filename": l.filename, "base_model": l.base_model,
             "size_bytes": l.size_bytes, "uploaded_at": l.uploaded_at, "description": l.description,
             "trigger_word": l.trigger_word, "strategy": l.strategy}
            for l in loras
        ],
        "count": len(loras),
    })


@app.post("/v1/loras", status_code=201)
async def upload_lora(request: Request) -> Response:
    from fastapi import UploadFile
    import io

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        return _error(400, "Expected multipart/form-data")

    form = await request.form(max_part_size=config.MAX_LORA_SIZE_BYTES)
    file = form.get("file")
    name = form.get("name")
    description = str(form.get("description", ""))
    base_model = str(form.get("base_model", "ltx-2.3"))
    trigger_word = form.get("trigger_word")
    strategy = form.get("strategy")
    trigger_word = str(trigger_word) if trigger_word else None
    strategy = str(strategy) if strategy else None

    if not file or not hasattr(file, "read"):
        return _error(400, "Missing 'file' field")
    if not name:
        return _error(422, "Missing 'name' field")

    filename = getattr(file, "filename", "unknown.safetensors") or "unknown.safetensors"
    if not filename.endswith(".safetensors"):
        return _error(400, "File must be a .safetensors file")

    data = await file.read()
    if len(data) > config.MAX_LORA_SIZE_BYTES:
        return _error(413, f"File exceeds {config.MAX_LORA_SIZE_BYTES // (1024*1024)}MB limit")

    try:
        info = lora_registry.add(name=str(name), filename=filename, data=data, description=description,
                                base_model=base_model, trigger_word=trigger_word, strategy=strategy)
    except ValueError as exc:
        return _error(400, str(exc))

    return JSONResponse(
        status_code=201,
        content={"id": info.id, "name": info.name, "filename": info.filename,
                 "base_model": info.base_model, "size_bytes": info.size_bytes,
                 "uploaded_at": info.uploaded_at, "description": info.description,
                 "trigger_word": info.trigger_word, "strategy": info.strategy},
    )


@app.delete("/v1/loras/{lora_id}")
async def delete_lora(lora_id: str) -> JSONResponse:
    if not lora_registry.delete(lora_id):
        return _error(404, f"LoRA not found: {lora_id}")
    return JSONResponse(content={"deleted": True, "id": lora_id})


# ---------------------------------------------------------------------------
# Flux LoRA endpoints (folder-drop — no upload, no delete)
# ---------------------------------------------------------------------------


@app.get("/v1/flux-loras")
async def list_flux_loras() -> JSONResponse:
    loras = flux_lora_registry.list_all()
    return JSONResponse(content={
        "loras": [
            {"id": l.id, "name": l.name, "filename": l.filename,
             "size_bytes": l.size_bytes, "model_compat": l.model_compat,
             "description": l.description, "trigger_word": l.trigger_word}
            for l in loras
        ],
        "count": len(loras),
    })


@app.post("/v1/flux-loras/rescan")
async def rescan_flux_loras() -> JSONResponse:
    count = flux_lora_registry.rescan()
    return JSONResponse(content={"rescanned": True, "count": count})


# ---------------------------------------------------------------------------
# V2 Async Job Endpoints
# ---------------------------------------------------------------------------


def _submit_job(job_type: JobType, params: dict, request: Request) -> JSONResponse:
    """Create a job, enqueue it, return 202."""
    if _paused:
        return JSONResponse(
            status_code=503,
            content={"error": "system_paused", "message": "System is paused for maintenance."},
            headers={"Retry-After": "300"},
        )
    if job_store.pending_count() >= config.MAX_QUEUE_DEPTH:
        return JSONResponse(
            status_code=429,
            content={"error": "queue_full", "message": "Job queue is full. Try again later."},
            headers={"Retry-After": "30"},
        )

    auth = request.headers.get("Authorization", "")
    api_key = auth[7:] if auth.startswith("Bearer ") else ""

    job = Job(id=make_job_id(), type=job_type, params=params, api_key=api_key)
    job_store.add(job)
    _job_queue.put_nowait(job.id)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.id,
            "status": "queued",
            "poll_url": f"/v2/jobs/{job.id}",
            "stream_url": f"/v2/jobs/{job.id}/stream",
        },
    )


@app.post("/v2/text-to-video")
async def v2_text_to_video(body: TextToVideoRequest, request: Request) -> JSONResponse:
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    width, height = _resolution_to_dims(body.resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    prompt = _build_prompt(body.prompt, body.camera_motion)
    seed = random.randint(0, 2**32 - 1)
    params = dict(prompt=prompt, model=body.model, width=width, height=height,
                  num_frames=num_frames, fps=body.fps, seed=seed, generate_audio=body.generate_audio,
                  lora_path=lora_path, lora_strength=lora_strength,
                  enhance_prompt=body.enhance_prompt)
    return _submit_job(JobType.TEXT_TO_VIDEO, params, request)


@app.post("/v2/image-to-video")
async def v2_image_to_video(body: ImageToVideoRequest, request: Request) -> JSONResponse:
    width, height = _resolution_to_dims(body.resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    keyframe_inputs = _resolve_keyframes(body, num_frames)
    if isinstance(keyframe_inputs, JSONResponse):
        return keyframe_inputs
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    seed = random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, keyframes=keyframe_inputs, model=body.model,
                  width=width, height=height, num_frames=num_frames, fps=body.fps,
                  seed=seed, generate_audio=body.generate_audio,
                  lora_path=lora_path, lora_strength=lora_strength,
                  enhance_prompt=body.enhance_prompt)
    return _submit_job(JobType.IMAGE_TO_VIDEO, params, request)


@app.post("/v2/audio-to-video")
async def v2_audio_to_video(body: AudioToVideoRequest, request: Request) -> JSONResponse:
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    audio_path = str(uploads.resolve(body.audio_uri))
    image_path: str | None = None
    if body.image_uri:
        image_path = str(uploads.resolve(body.image_uri))
    width, height = _resolution_to_dims(body.resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    seed = random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, audio_path=audio_path, image_path=image_path,
                  model=body.model, width=width, height=height, num_frames=num_frames,
                  fps=body.fps, seed=seed,
                  lora_path=lora_path, lora_strength=lora_strength,
                  enhance_prompt=body.enhance_prompt)
    return _submit_job(JobType.AUDIO_TO_VIDEO, params, request)


@app.post("/v2/retake")
async def v2_retake(body: RetakeRequest, request: Request) -> JSONResponse:
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    video_path = str(uploads.resolve(body.video_uri))
    prompt = body.prompt or ""
    seed = random.randint(0, 2**32 - 1)
    params = dict(video_path=video_path, start_time=body.start_time,
                  duration=body.duration, mode=body.mode, prompt=prompt, seed=seed,
                  lora_path=lora_path, lora_strength=lora_strength)
    return _submit_job(JobType.RETAKE, params, request)


@app.post("/v2/text-to-image")
async def v2_text_to_image(body: TextToImageRequest, request: Request) -> JSONResponse:
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    width = (body.width // 16) * 16
    height = (body.height // 16) * 16
    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, width=width, height=height,
                  num_inference_steps=body.num_inference_steps,
                  guidance_scale=body.guidance_scale, seed=seed,
                  model=body.model, turbo=body.turbo,
                  lora_path=lora_path, lora_strength=lora_strength)
    return _submit_job(JobType.TEXT_TO_IMAGE, params, request)


@app.post("/v2/image-to-image")
async def v2_image_to_image(body: ImageToImageRequest, request: Request) -> JSONResponse:
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    image_path = str(uploads.resolve(body.image_uri))
    width = (body.width // 16) * 16
    height = (body.height // 16) * 16
    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, image_path=image_path, width=width, height=height,
                  num_inference_steps=body.num_inference_steps,
                  guidance_scale=body.guidance_scale, seed=seed,
                  model=body.model, turbo=body.turbo,
                  lora_path=lora_path, lora_strength=lora_strength)
    return _submit_job(JobType.IMAGE_TO_IMAGE, params, request)


@app.post("/v2/image-edit")
async def v2_image_edit(body: ImageEditRequest, request: Request) -> JSONResponse:
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    image_paths = [str(uploads.resolve(uri)) for uri in body.image_uris]
    width = (body.width // 16) * 16
    height = (body.height // 16) * 16
    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, image_paths=image_paths, width=width, height=height,
                  num_inference_steps=body.num_inference_steps,
                  guidance_scale=body.guidance_scale, seed=seed,
                  model=body.model,
                  lora_path=lora_path, lora_strength=lora_strength)
    return _submit_job(JobType.IMAGE_EDIT, params, request)


@app.post("/v2/music")
async def v2_music(body: MusicGenerationRequest, request: Request) -> JSONResponse:
    if _paused:
        return JSONResponse(status_code=503, content={"error": "system_paused"}, headers={"Retry-After": "300"})
    if _turbo_active:
        return JSONResponse(
            status_code=503,
            content={"error": "turbo_mode_active: ACE/JoyAI unavailable while turbo mode is enabled. Disable turbo first."},
            headers={"Retry-After": "10"},
        )
    if not config.LOAD_ACE:
        return _error(503, "Music generation not enabled (LOAD_ACE=0)")
    # Validate task-type requirements
    if body.task_type in ("cover", "repaint", "extract", "lego", "complete") and not body.source_audio_uri:
        return _error(422, f"task_type '{body.task_type}' requires source_audio_uri")
    if body.task_type in ("extract", "lego", "complete") and not body.track_name:
        return _error(422, f"task_type '{body.task_type}' requires track_name")
    # Queue depth check for music
    music_pending = sum(
        1 for j in job_store._jobs.values()
        if j.type == JobType.MUSIC_GENERATION and j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
    )
    if music_pending >= config.MAX_MUSIC_PENDING:
        return JSONResponse(status_code=429, content={"error": "music_queue_full"}, headers={"Retry-After": "30"})
    # Resolve URIs
    params = body.model_dump(exclude_none=True)
    if body.source_audio_uri:
        try:
            params["source_audio_path"] = str(uploads.resolve(body.source_audio_uri))
        except FileNotFoundError:
            return _error(404, "source_audio_uri not found")
    if body.reference_audio_uri:
        try:
            params["reference_audio_path"] = str(uploads.resolve(body.reference_audio_uri))
        except FileNotFoundError:
            return _error(404, "reference_audio_uri not found")
    params.pop("source_audio_uri", None)
    params.pop("reference_audio_uri", None)
    auth = request.headers.get("Authorization", "")
    api_key = auth[7:] if auth.startswith("Bearer ") else ""
    job = Job(id=make_job_id(), type=JobType.MUSIC_GENERATION, params=params, api_key=api_key)
    job_store.add(job)
    asyncio.create_task(_run_music_job(job))
    return JSONResponse(status_code=202, content={
        "job_id": job.id, "status": "queued",
        "poll_url": f"/v2/jobs/{job.id}",
        "stream_url": f"/v2/jobs/{job.id}/stream",
    })


@app.get("/v2/jobs/{job_id}")
async def v2_job_status(job_id: str) -> JSONResponse:
    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")
    return JSONResponse(content={
        "job_id": job.id,
        "status": job.status,
        "type": job.type,
        "progress": job.progress if job.status == JobStatus.PROCESSING else (1.0 if job.status == JobStatus.COMPLETED else None),
        "phase": job.phase if job.status == JobStatus.PROCESSING else None,
        "queue_position": job_store.queue_position(job.id) if job.status == JobStatus.QUEUED else None,
        "error": {"code": job.error_code or "generation_failed", "message": job.error} if job.error else None,
        "result_url": f"/v2/jobs/{job.id}/result" if job.status == JobStatus.COMPLETED else None,
        "result_storage_uri": job.result_uri if job.status == JobStatus.COMPLETED else None,
        "result_media_type": job.result_media_type,
    })


@app.get("/v2/jobs/{job_id}/preview")
async def v2_job_preview(job_id: str) -> Response:
    """Return a low-res preview JPEG for a job in progress or completed.

    Four paths:
    1. Fast path — `job.preview_bytes` already populated (Flux step-end callback,
       or history-save backfill on the worker). Serve the cached JPEG.
    2. On-disk thumbnail path — `history.save()` has already written a thumbnail
       to `config.THUMBNAIL_DIR / thumb_{upload_id}`. Serve it as a zero-copy
       `FileResponse` (sendfile). Survives restart because it's on disk.
    3. Fallback lazy extraction — completed video job with no thumbnail on disk
       (e.g. no api_key so history.save was skipped, or the save task failed).
       Read + decode the first frame via PyAV, but offload the 100+ MB read and
       the PyAV call to a thread so the event loop isn't blocked.
    4. 204 — queued / processing / failed without result bytes. Frontends should
       keep polling; `404` shows up red in browser dev tools and confuses users
       into thinking the job is broken.
    """
    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")

    # Path 1: cached preview in RAM (Flux step-end, or worker backfill)
    if job.preview_bytes:
        return Response(content=job.preview_bytes, media_type="image/jpeg")

    # Path 2: on-disk thumbnail from history.save() — zero-copy sendfile
    if job.result_uri:
        upload_id = job.result_uri.removeprefix("storage://")
        thumb_path = config.THUMBNAIL_DIR / f"thumb_{upload_id}"
        if thumb_path.exists():
            return FileResponse(path=str(thumb_path), media_type="image/jpeg")

    # Path 3: fallback lazy extraction for completed video jobs without an
    # on-disk thumbnail. Offloaded to a thread so the 100+ MB read + PyAV
    # decode don't block the event loop.
    if (
        job.status == JobStatus.COMPLETED
        and job.result_uri
        and (job.result_media_type or "").startswith("video/")
    ):
        try:
            result_path = uploads.resolve(job.result_uri)
            if result_path.exists():
                from history_store import _first_video_frame_as_pil

                def _extract() -> bytes | None:
                    video_bytes = result_path.read_bytes()
                    frame = _first_video_frame_as_pil(video_bytes)
                    if frame is None:
                        return None
                    import io as _io
                    buf = _io.BytesIO()
                    frame.convert("RGB").save(buf, format="JPEG", quality=80)
                    return buf.getvalue()

                preview = await asyncio.to_thread(_extract)
                if preview is not None:
                    job.preview_bytes = preview  # cache for subsequent polls
                    return Response(content=preview, media_type="image/jpeg")
        except Exception:
            logger.exception("Failed to extract video preview for job %s", job_id)

    # Path 4: no preview available — 204, not 404
    return Response(status_code=204)


@app.get("/v2/jobs/{job_id}/result")
async def v2_job_result(job_id: str) -> Response:
    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")
    if job.status != JobStatus.COMPLETED or not job.result_uri:
        return _error(409, "Job result not ready")
    try:
        path = uploads.resolve(job.result_uri)
        if not path.exists():
            raise FileNotFoundError()
        return FileResponse(
            path=str(path),
            media_type=job.result_media_type or "application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )
    except FileNotFoundError:
        return _error(404, "Result file expired or not found")


@app.delete("/v2/jobs/{job_id}")
async def v2_cancel_job(job_id: str) -> JSONResponse:
    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return _error(409, "Cannot cancel a finished job")
    job.status = JobStatus.CANCELLED
    return JSONResponse(content={"job_id": job.id, "status": "cancelled"})


@app.get("/v2/jobs/{job_id}/stream")
async def v2_job_stream(
    job_id: str, request: Request, token: str | None = None,
) -> Response:
    """SSE stream for live job status + progress + phase updates.

    Eliminates client-side polling: instead of hitting `/v2/jobs/{id}` every
    500 ms for two minutes (240 GETs per video job), the client opens one
    EventSource and receives push updates whenever (status, progress, phase)
    changes. The stream closes itself with one final event on terminal state
    (completed / failed / cancelled).

    **Auth**: EventSource cannot set custom headers, so pass either a bearer
    `Authorization` header (programmatic clients) or `?token=<sse-token>`
    query param (browsers — get one via `POST /v1/sse-token`).

    **Event format**: each event is a single `data:` line containing the same
    JSON shape as `GET /v2/jobs/{id}`, so clients can reuse their existing
    polling parser. Idle-period keepalive comments (`: keepalive`) are emitted
    every 15 s to prevent intermediate proxies from closing the connection
    during long queue waits.
    """
    # Match the middleware's "no keys configured = auth disabled" mode, since
    # this endpoint bypasses the middleware to allow `?token=` query-param auth
    # (EventSource can't set headers).
    if config.API_KEYS:
        api_key = _resolve_sse_token(token) or _extract_api_key(request)
        if not api_key:
            return _error(401, "Missing API key")

    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")

    import json as _json
    import time as _time

    def _snapshot(j: Job) -> dict:
        return {
            "job_id": j.id,
            "status": j.status,
            "type": j.type,
            "progress": j.progress,
            "phase": j.phase,
            "queue_position": (
                job_store.queue_position(j.id) if j.status == JobStatus.QUEUED else None
            ),
            "error": (
                {"code": j.error_code or "generation_failed", "message": j.error}
                if j.error else None
            ),
            "result_url": f"/v2/jobs/{j.id}/result" if j.status == JobStatus.COMPLETED else None,
            "result_storage_uri": j.result_uri if j.status == JobStatus.COMPLETED else None,
            "result_media_type": j.result_media_type,
        }

    async def event_stream():
        # Dedup key — only emit when something observable changes.
        # Round progress to 3 decimals so micro-fluctuations don't flood the stream.
        last_key: tuple | None = None
        last_keepalive = _time.monotonic()
        terminal = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        try:
            while True:
                if await request.is_disconnected():
                    return
                j = job_store.get(job_id)
                if j is None:
                    # Job expired from the store (cleanup TTL). Surface it and close.
                    yield f"event: error\ndata: {_json.dumps({'error': 'job_expired'})}\n\n"
                    return
                key = (j.status, round(j.progress, 3), j.phase, j.error_code)
                if key != last_key:
                    yield f"data: {_json.dumps(_snapshot(j))}\n\n"
                    last_key = key
                    last_keepalive = _time.monotonic()
                if j.status in terminal:
                    return
                now = _time.monotonic()
                if now - last_keepalive > 15:
                    yield ": keepalive\n\n"
                    last_keepalive = now
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if ever proxied
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# SSE session tokens (short-lived, keeps API key out of URLs)
# ---------------------------------------------------------------------------

_sse_tokens: dict[str, tuple[str, float]] = {}  # token → (api_key, expires_at)


@app.post("/v1/sse-token")
async def create_sse_token(request: Request) -> JSONResponse:
    """Issue a 5-minute disposable token for SSE EventSource connections."""
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    import time as _time
    token = _secrets.token_urlsafe(32)
    _sse_tokens[token] = (api_key, _time.time() + 300)
    # Prune expired tokens
    now = _time.time()
    expired = [t for t, (_, exp) in _sse_tokens.items() if exp < now]
    for t in expired:
        del _sse_tokens[t]
    return JSONResponse(content={"token": token, "expires_in": 300})


def _resolve_sse_token(token: str | None) -> str | None:
    """Resolve an SSE token to its API key, or None if invalid/expired."""
    if not token:
        return None
    import time as _time
    entry = _sse_tokens.get(token)
    if not entry:
        return None
    api_key, expires_at = entry
    if _time.time() > expires_at:
        del _sse_tokens[token]
        return None
    return api_key


# ---------------------------------------------------------------------------
# Approved Images (noodle-i → noodle-v pipeline)
# ---------------------------------------------------------------------------


@app.post("/v1/approved-images")
async def approve_image(request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    body = await request.json()
    image_uri = body.get("image_uri")
    if not image_uri:
        return _error(400, "Missing image_uri")

    import json, hashlib, time as _time
    config.APPROVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = config.APPROVED_IMAGES_DIR / "manifest.json"
    raw = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = raw.get("images", raw) if isinstance(raw, dict) else raw

    entry = {
        "id": hashlib.sha256(f"{image_uri}{_time.time()}".encode()).hexdigest()[:16],
        "image_uri": image_uri,
        "prompt": body.get("prompt", ""),
        "model": body.get("model", ""),
        "width": body.get("width", 0),
        "height": body.get("height", 0),
        "api_key_hash": hashlib.sha256(api_key.encode()).hexdigest(),
        "created_at": _time.time(),
    }
    manifest.insert(0, entry)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return JSONResponse(content={"id": entry["id"], "status": "approved"}, status_code=201)


@app.get("/v1/approved-images")
async def list_approved_images(request: Request, limit: int = 50, offset: int = 0) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")

    import json, hashlib
    manifest_path = config.APPROVED_IMAGES_DIR / "manifest.json"
    if not manifest_path.exists():
        return JSONResponse(content=[])

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    raw = json.loads(manifest_path.read_text())
    manifest = raw.get("images", raw) if isinstance(raw, dict) else raw
    filtered = [e for e in manifest if e.get("api_key_hash") == key_hash]
    page = filtered[offset : offset + limit]

    results = []
    for e in page:
        r = {k: v for k, v in e.items() if k != "api_key_hash"}
        r["image_url"] = f"/v1/approved-images/{e['id']}/file"
        results.append(r)
    return JSONResponse(content=results)


@app.get("/v1/approved-images/events")
async def approved_images_events(request: Request, token: str | None = None) -> StreamingResponse:
    """SSE stream — emits new approved images as they arrive."""
    api_key = _resolve_sse_token(token) or _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")

    import json, hashlib

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    manifest_path = config.APPROVED_IMAGES_DIR / "manifest.json"

    async def event_stream():
        seen_ids: set[str] = set()
        last_mtime: float = 0

        while True:
            if await request.is_disconnected():
                break

            try:
                if manifest_path.exists():
                    mtime = manifest_path.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        raw = json.loads(manifest_path.read_text())
                        manifest = raw.get("images", raw) if isinstance(raw, dict) else raw
                        filtered = [e for e in manifest if e.get("api_key_hash") == key_hash]

                        for entry in filtered:
                            if entry["id"] not in seen_ids:
                                seen_ids.add(entry["id"])
                                r = {k: v for k, v in entry.items() if k != "api_key_hash"}
                                r["image_url"] = f"/v1/approved-images/{entry['id']}/file"
                                yield f"data: {json.dumps(r)}\n\n"
            except Exception:
                pass

            await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/v1/approved-images/{image_id}/file")
async def get_approved_image_file(image_id: str, request: Request) -> Response:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")

    import json, hashlib
    manifest_path = config.APPROVED_IMAGES_DIR / "manifest.json"
    if not manifest_path.exists():
        return _error(404, "Not found")

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    raw = json.loads(manifest_path.read_text())
    manifest = raw.get("images", raw) if isinstance(raw, dict) else raw
    entry = next((e for e in manifest if e["id"] == image_id and e.get("api_key_hash") == key_hash), None)
    if not entry:
        return _error(404, "Not found")

    path = uploads.resolve(entry["image_uri"])
    if not path.exists():
        return _error(404, "Image file not found")
    return FileResponse(path=str(path), media_type="image/webp")


# ---------------------------------------------------------------------------
# Char Mode — Vision Ranking
# ---------------------------------------------------------------------------

CHAR_RANKING_PROMPT = """You are a character consistency evaluator for AI image generation. You compare a REFERENCE character image against a GENERATED image to assess how well the generated image preserves the character's identity.

Focus specifically on:
- Facial structure: jaw shape, cheekbones, chin, nose bridge
- Eyes: shape, size, spacing, color, eyelid characteristics
- Head proportions: forehead height, face width-to-height ratio, head size relative to body
- Overall likeness: would someone recognize this as the same person?

Rate each criterion from 1-10 where:
- 1-3: Poor match, clearly different person
- 4-6: Some resemblance but noticeable differences
- 7-8: Good match with minor discrepancies
- 9-10: Excellent match, clearly the same character

Return ONLY valid JSON with no additional text:
{
  "score": <float, average of all criteria>,
  "analysis": {
    "face_match": <int 1-10>,
    "eyes": <int 1-10>,
    "proportions": <int 1-10>,
    "overall_likeness": <int 1-10>
  },
  "edits": {
    "add": ["specific description to append to the generation prompt"],
    "remove": ["specific words/phrases to remove from the prompt"],
    "modify": {"aspect_name": "improved description"}
  }
}

Rules for edits:
- Only suggest edits if score < 9
- Keep edits SMALL and SPECIFIC (1-3 items max)
- Focus on the biggest discrepancy between reference and generated
- Use concrete terms: "narrower jawline" not "better face"
- If score >= 9, return empty edits: {"add": [], "remove": [], "modify": {}}"""


@app.post("/v2/char/rank")
async def v2_char_rank(body: CharRankRequest, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    if not chat.is_ready:
        return _error(500, "Chat model not loaded")

    try:
        import base64

        rank_path = uploads.resolve(body.rank_image_uri)
        gen_path = uploads.resolve(body.generated_image_uri)

        rank_b64 = base64.b64encode(rank_path.read_bytes()).decode()
        gen_b64 = base64.b64encode(gen_path.read_bytes()).decode()

        messages = [
            {"role": "system", "content": CHAR_RANKING_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f'Original prompt: "{body.prompt}"\n\nFirst image is the REFERENCE character. Second image is the GENERATED result. Evaluate character consistency.'},
                {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{rank_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{gen_b64}"}},
            ]}
        ]

        result = await chat.generate_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            model=config.CHAR_VISION_MODEL,
        )

        import json as _json, re as _re
        text = result["choices"][0]["message"]["content"]
        json_match = _re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return _error(500, "Vision model did not return valid JSON")

        ranking = _json.loads(json_match.group())
        return JSONResponse(content=ranking)

    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except ValueError:
        return _error(500, "Failed to parse vision model response")
    except Exception as exc:
        logger.exception("char/rank failed")
        return _error(500, str(exc))


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@app.get("/v2/history")
async def v2_history(request: Request, limit: int = 50, offset: int = 0, type: str | None = None) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    items = history.list(api_key, limit=min(limit, 200), offset=offset, job_type=type)
    results = []
    for item in items:
        r = {
            "id": item["id"],
            "prompt": item["prompt"],
            "model": item["model"],
            "width": item["width"],
            "height": item["height"],
            "turbo": bool(item["turbo"]),
            "status": item["status"],
            "created_at": item["created_at"],
            "error": item["error"],
        }
        if item["thumbnail_uri"]:
            r["thumbnail_url"] = f"/v2/history/{item['id']}/thumbnail"
        if item["result_uri"]:
            r["image_url"] = f"/v2/history/{item['id']}/image"
        results.append(r)
    return JSONResponse(content=results)


@app.get("/v2/history/{generation_id}/image")
async def v2_history_image(generation_id: str, request: Request) -> Response:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    item = history.get(generation_id, api_key)
    if not item or not item["result_uri"]:
        return _error(404, "Not found")
    path = uploads.resolve(item["result_uri"])
    if not path.exists():
        return _error(404, "Result file not found")
    media_type = "video/mp4" if "video" in item.get("job_type", "") else "image/webp"
    return FileResponse(path=str(path), media_type=media_type)


@app.get("/v2/history/{generation_id}/thumbnail")
async def v2_history_thumbnail(generation_id: str, request: Request) -> Response:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    item = history.get(generation_id, api_key)
    if not item or not item["thumbnail_uri"]:
        return _error(404, "Not found")
    thumb_id = item["thumbnail_uri"].removeprefix("thumb://")
    path = config.THUMBNAIL_DIR / thumb_id
    if not path.exists():
        return _error(404, "Thumbnail not found")
    return FileResponse(path=str(path), media_type="image/jpeg")


@app.delete("/v2/history/{generation_id}")
async def v2_history_delete(generation_id: str, request: Request) -> JSONResponse:
    """Remove a history entry + its result/thumbnail files. Scoped to the
    caller's API key — returns 404 both when the entry doesn't exist and
    when it belongs to another key, so the ID space can't be probed."""
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    if not history.delete(generation_id, api_key):
        return _error(404, "Not found")
    return JSONResponse(content={"ok": True})


def _extract_api_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


# ---------------------------------------------------------------------------
# Batch Endpoints — Phase 3
# ---------------------------------------------------------------------------


@app.post("/v2/batch")
async def v2_batch_submit(body: BatchRequest, request: Request) -> JSONResponse:
    """Submit a batch of generation jobs. Returns 202 with batch_id."""
    if _paused:
        return JSONResponse(
            status_code=503,
            content={"error": "system_paused", "message": "System is paused for maintenance."},
            headers={"Retry-After": "300"},
        )
    if batch_store.active_count() >= config.MAX_BATCH_QUEUE_DEPTH:
        return JSONResponse(
            status_code=429,
            content={"error": "batch_queue_full", "message": "Batch queue is full. Try again later."},
            headers={"Retry-After": "30"},
        )

    # Validate all items eagerly — parse through corresponding Pydantic model
    for i, item in enumerate(body.items):
        if item.type not in _BATCH_TYPE_MAP:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid item type at index {i}: {item.type}"},
            )
        _, validator_model = _BATCH_TYPE_MAP[item.type]
        try:
            validator_model(**item.params)
        except Exception as exc:
            return JSONResponse(
                status_code=422,
                content={"error": f"Validation failed at item {i} ({item.type}): {exc}"},
            )

    auth = request.headers.get("Authorization", "")
    api_key = auth[7:] if auth.startswith("Bearer ") else ""

    # Swap optimization: sort all images before all videos, group by model within images
    image_items = [it for it in body.items if _is_image_type(it.type)]
    video_items = [it for it in body.items if not _is_image_type(it.type)]
    # Within image items, sort klein before dev to minimize Flux model reloads
    image_items.sort(key=lambda it: it.params.get("model", "flux2-klein"))
    sorted_items = image_items + video_items

    batch = BatchJob(
        id=make_batch_id(),
        items=sorted_items,
        api_key=api_key,
        total=len(sorted_items),
        turbo=_turbo_active,
        priority=body.priority,
        callback_url=body.callback_url,
    )
    batch_store.add(batch)
    _batch_queue.put_nowait(batch.id)

    return JSONResponse(
        status_code=202,
        content={
            "batch_id": batch.id,
            "status": "queued",
            "total": batch.total,
            "queue_position": max(0, batch_store.active_count() - 1),
        },
    )


@app.get("/v2/batch/{batch_id}")
async def v2_batch_status(batch_id: str) -> JSONResponse:
    """Poll batch status + partial results."""
    batch = batch_store.get(batch_id)
    if batch is None:
        return _error(404, "Batch not found")
    return JSONResponse(content={
        "batch_id": batch.id,
        "status": batch.status,
        "total": batch.total,
        "completed_count": batch.completed_count,
        "failed_count": batch.failed_count,
        "current_index": batch.current_index,
        "turbo": batch.turbo,
        "results": [
            {
                "index": r.index,
                "type": r.type,
                "status": r.status,
                "result_uri": r.result_uri,
                "media_type": r.media_type,
                "error": r.error,
                "elapsed_s": r.elapsed_s,
            }
            for r in batch.results
        ],
        "created_at": batch.created_at,
        "started_at": batch.started_at,
        "completed_at": batch.completed_at,
    })


@app.delete("/v2/batch/{batch_id}")
async def v2_batch_cancel(batch_id: str) -> JSONResponse:
    """Cancel remaining items in a batch."""
    batch = batch_store.get(batch_id)
    if batch is None:
        return _error(404, "Batch not found")
    if batch.status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED):
        return JSONResponse(status_code=409, content={
            "error": "batch_already_finished",
            "batch_id": batch.id,
            "status": batch.status,
        })
    cancelled_count = batch.total - batch.completed_count - batch.failed_count
    if batch.status == BatchStatus.PROCESSING:
        # Currently running item will finish, then batch_worker sees cancelled status
        cancelled_count = max(0, cancelled_count - 1)
    batch.status = BatchStatus.CANCELLED
    return JSONResponse(content={
        "batch_id": batch.id,
        "status": "cancelled",
        "completed_count": batch.completed_count,
        "cancelled_count": cancelled_count,
    })


# ---------------------------------------------------------------------------
# Batch worker coroutine
# ---------------------------------------------------------------------------


async def _save_batch_item_history(
    hist: "HistoryStore", job: Job, result_bytes: bytes, storage_uri: str,
) -> None:
    """Fire-and-forget history save for a single batch item."""
    try:
        p = job.params or {}
        await asyncio.to_thread(hist.save,
            job_id=job.id, api_key=job.api_key, job_type=job.type,
            prompt=p.get("prompt", ""), model=p.get("model"),
            width=p.get("width", 0), height=p.get("height", 0),
            turbo=p.get("turbo", False), status=JobStatus.COMPLETED,
            result_uri=storage_uri, result_bytes=result_bytes,
            created_at=time.time(), completed_at=time.time(), error=None,
        )
    except Exception:
        logger.warning("Failed to save batch item %s to history", job.id, exc_info=True)


async def _fire_batch_webhook(batch: BatchJob) -> None:
    """Fire-and-forget webhook with 3 retries."""
    import httpx
    payload = {
        "batch_id": batch.id,
        "status": batch.status,
        "total": batch.total,
        "completed_count": batch.completed_count,
        "failed_count": batch.failed_count,
        "results": [
            {"index": r.index, "type": r.type, "status": r.status,
             "result_uri": r.result_uri, "media_type": r.media_type,
             "error": r.error, "elapsed_s": r.elapsed_s}
            for r in batch.results
        ],
    }
    backoff = [1, 5, 15]
    for attempt, delay in enumerate(backoff):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(batch.callback_url, json=payload)
                if resp.is_success:
                    logger.info("Batch %s webhook delivered on attempt %d", batch.id, attempt + 1)
                    return
        except Exception:
            pass
        if attempt < len(backoff) - 1:
            await asyncio.sleep(delay)
    logger.warning("Batch %s webhook failed after 3 retries to %s", batch.id, batch.callback_url)


async def _process_batch_item(batch, i: int, item) -> None:
    """Process a single batch item. Used by both turbo (no lock) and normal mode.

    In turbo mode, SplitModelManager._acquire_worker() serializes per-GPU.
    In normal mode, the caller holds _inference_lock before calling this.
    """
    t0 = time.monotonic()
    try:
        job = _batch_item_to_job(item, batch.api_key)
        if not _turbo_active:
            # Single-GPU: ensure correct tenant before dispatch
            if _is_image_type(item.type):
                await _ensure_flux_ready()
            else:
                await _ensure_ltx_resident()
        result_bytes = await _dispatch_job(job)

        upload_id, storage_uri = uploads.create()
        uploads.save(upload_id, result_bytes)
        elapsed = time.monotonic() - t0

        batch.results.append(BatchItemResult(
            index=i, type=item.type, status="completed",
            result_uri=storage_uri,
            media_type=_JOB_MEDIA_TYPES.get(JobType(item.type), "application/octet-stream"),
            elapsed_s=round(elapsed, 2),
        ))
        batch.completed_count += 1

        if history and batch.api_key:
            asyncio.create_task(_save_batch_item_history(
                history, job, result_bytes, storage_uri))

    except Exception as exc:
        elapsed = time.monotonic() - t0
        batch.results.append(BatchItemResult(
            index=i, type=item.type, status="failed",
            error=str(exc)[:500], elapsed_s=round(elapsed, 2),
        ))
        batch.failed_count += 1
        logger.exception("Batch %s item %d failed", batch.id, i)


async def batch_worker() -> None:
    """Background worker that processes batches from the batch queue.

    In turbo mode: dispatches 2 items concurrently via asyncio.gather
    (one per GPU worker). SplitModelManager._acquire_worker() assigns
    each job to a different GPU.

    In normal mode: dispatches 1 item at a time, holding _inference_lock,
    with swap logic (_ensure_ltx_resident / _ensure_flux_ready).
    """
    logger.info("Batch worker started")

    while True:
        batch_id = await _batch_queue.get()
        batch = batch_store.get(batch_id)
        if batch is None or batch.status == BatchStatus.CANCELLED:
            _batch_queue.task_done()
            continue

        batch.status = BatchStatus.PROCESSING
        batch.started_at = time.monotonic()
        logger.info("Processing batch %s (%d items, turbo=%s)", batch.id, batch.total, _turbo_active)

        items = list(enumerate(batch.items))
        idx = 0
        while idx < len(items):
            if batch.status == BatchStatus.CANCELLED:
                for j in range(idx, len(items)):
                    batch.results.append(BatchItemResult(
                        index=items[j][0], type=items[j][1].type,
                        status="cancelled",
                    ))
                break

            if _turbo_active:
                # Turbo: dispatch 2 items concurrently (one per GPU worker)
                chunk = items[idx:idx+2]
                batch.current_index = chunk[0][0]
                tasks = [_process_batch_item(batch, i, item) for i, item in chunk]
                await asyncio.gather(*tasks)
                idx += len(chunk)
            else:
                # Normal: 1 at a time with inference lock
                i, item = items[idx]
                batch.current_index = i
                async with _inference_lock:
                    await _process_batch_item(batch, i, item)
                idx += 1

        # Final status
        batch.completed_at = time.monotonic()
        if batch.status != BatchStatus.CANCELLED:
            if batch.failed_count == 0:
                batch.status = BatchStatus.COMPLETED
            elif batch.completed_count == 0:
                batch.status = BatchStatus.FAILED
            else:
                batch.status = BatchStatus.PARTIAL

        total_elapsed = batch.completed_at - (batch.started_at or batch.created_at)
        logger.info("Batch %s finished: %d/%d completed in %.1fs",
                     batch.id, batch.completed_count, batch.total, total_elapsed)

        if batch.callback_url:
            asyncio.create_task(_fire_batch_webhook(batch))

        _batch_queue.task_done()


async def batch_cleanup_loop() -> None:
    """Periodically remove expired batch metadata."""
    ttl = config.BATCH_RESULT_TTL_SECONDS
    logger.info("Batch cleanup loop started (TTL=%ds)", ttl)
    while True:
        await asyncio.sleep(60)
        now = time.monotonic()
        to_remove: list[str] = []
        for batch in batch_store.all_batches():
            if batch.status in (BatchStatus.COMPLETED, BatchStatus.FAILED,
                                BatchStatus.CANCELLED, BatchStatus.PARTIAL):
                if batch.completed_at and (now - batch.completed_at) > ttl:
                    to_remove.append(batch.id)
        for bid in to_remove:
            batch_store.remove(bid)
        if to_remove:
            logger.info("Cleaned up %d expired batches", len(to_remove))


# ---------------------------------------------------------------------------
# Compositions
# ---------------------------------------------------------------------------

from composition_store import CompositionStore
compositions = CompositionStore()


@app.post("/v2/compositions")
async def v2_compositions_create(request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    body = await request.json()
    name = body.get("name", "Untitled")
    data = {"clips": body.get("clips", []), "transitions": body.get("transitions", [])}
    result = compositions.create(api_key, name, data)
    return JSONResponse(content=result, status_code=201)


@app.get("/v2/compositions")
async def v2_compositions_list(request: Request, limit: int = 50, offset: int = 0) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    items = compositions.list(api_key, limit=min(limit, 200), offset=offset)
    return JSONResponse(content=items)


@app.get("/v2/compositions/{comp_id}")
async def v2_compositions_get(comp_id: str, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    item = compositions.get(comp_id, api_key)
    if not item:
        return _error(404, "Composition not found")
    return JSONResponse(content=item)


@app.put("/v2/compositions/{comp_id}")
async def v2_compositions_update(comp_id: str, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    body = await request.json()
    name = body.get("name", "Untitled")
    data = {"clips": body.get("clips", []), "transitions": body.get("transitions", [])}
    updated = compositions.update(comp_id, api_key, name, data)
    if not updated:
        return _error(404, "Composition not found")
    return JSONResponse(content={"status": "updated"})


@app.delete("/v2/compositions/{comp_id}")
async def v2_compositions_delete(comp_id: str, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    deleted = compositions.delete(comp_id, api_key)
    if not deleted:
        return _error(404, "Composition not found")
    return JSONResponse(content={"status": "deleted"})


@app.post("/v2/compositions/{comp_id}/export")
async def v2_compositions_export(comp_id: str, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    comp = compositions.get(comp_id, api_key)
    if not comp:
        return _error(404, "Composition not found")
    params = {
        "composition_id": comp_id,
        "clips": comp["data"].get("clips", []),
        "transitions": comp["data"].get("transitions", []),
    }
    return _submit_job(JobType.EXPORT_COMPOSITION, params, request)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )
