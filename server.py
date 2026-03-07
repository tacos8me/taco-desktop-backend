"""FastAPI server implementing the LTX-compatible API for taco-desktop."""

from __future__ import annotations

import asyncio
import logging
import random
import secrets as _secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import config
from split_model_manager import SplitModelManager
from flux_manager import FluxManager
from chat_manager import ChatManager
from helpers import _duration_to_frames, _resolution_to_dims
from upload_store import UploadStore
from job_queue import (
    Job, JobStatus, JobType, JobStore, make_job_id, make_flux_callback,
    worker_loop, cleanup_loop,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

manager = SplitModelManager()
flux = FluxManager()
uploads = UploadStore(config.UPLOAD_DIR)
chat = ChatManager()

# Shared inference lock: FP8 layerwise casting in diffusers causes CUBLAS_STATUS_INTERNAL_ERROR
# when Flux and LTX run CUDA inference concurrently in the same process.
_inference_lock = asyncio.Lock()

# Job queue
job_store = JobStore()
_job_queue: asyncio.Queue[str] = asyncio.Queue()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


async def _dispatch_job(job: Job) -> bytes:
    """Route a job to the correct manager and return result bytes."""
    p = job.params

    def on_progress(progress: float) -> None:
        job.progress = progress

    match job.type:
        case JobType.TEXT_TO_VIDEO:
            return await manager.generate_text_to_video(**p, on_progress=on_progress)
        case JobType.IMAGE_TO_VIDEO:
            return await manager.generate_image_to_video(**p, on_progress=on_progress)
        case JobType.AUDIO_TO_VIDEO:
            return await manager.generate_audio_to_video(**p, on_progress=on_progress)
        case JobType.RETAKE:
            return await manager.retake(**p, on_progress=on_progress)
        case JobType.TEXT_TO_IMAGE:
            cb = make_flux_callback(job, p.get("num_inference_steps", 50))
            return await flux.generate_text_to_image(**p, callback_on_step_end=cb)
        case JobType.IMAGE_TO_IMAGE:
            cb = make_flux_callback(job, p.get("num_inference_steps", 50))
            return await flux.generate_image_to_image(**p, callback_on_step_end=cb)
        case _:
            raise ValueError(f"Unknown job type: {job.type}")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading LTX pipelines on %s ...", config.GPU_DEVICES)
    manager.load_all()
    logger.info("LTX pipelines ready.")

    logger.info("Loading Flux pipeline on %s ...", config.FLUX_DEVICE)
    flux.load()
    logger.info("Flux pipeline ready.")

    chat.load()
    logger.info("Chat proxy ready.")

    worker_task = asyncio.create_task(
        worker_loop(job_store, _job_queue, _inference_lock, _dispatch_job, uploads),
        name="queue-worker",
    )
    cleanup_task = asyncio.create_task(
        cleanup_loop(job_store, uploads),
        name="queue-cleanup",
    )
    logger.info("Job queue started.")

    yield

    worker_task.cancel()
    cleanup_task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|192\.168\.\d+\.\d+)(:\d+)?$",
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if not config.API_KEYS:
        return await call_next(request)
    if request.url.path == "/health":
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
Resolution = Literal["1920x1080", "1080x1920", "2560x1440", "1440x2560", "3840x2160", "2160x3840"]
RetakeMode = Literal["replace_audio_and_video", "replace_video", "replace_video_only", "replace_audio"]
ImageModelName = Literal["flux2-dev"]


class TextToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False
    camera_motion: str | None = Field(default=None, max_length=200)


class KeyframeInput(BaseModel):
    image_uri: str
    frame_index: int = Field(default=0, ge=0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class ImageToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str | None = None
    keyframes: list[KeyframeInput] | None = None
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False


class AudioToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    audio_uri: str
    image_uri: str | None = None
    model: ModelName
    resolution: Resolution
    duration: float = Field(default=6.0, gt=0, le=30)
    fps: float = Field(default=24.0, gt=0, le=60)


class RetakeRequest(BaseModel):
    video_uri: str
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0, le=30)
    mode: RetakeMode
    prompt: str | None = Field(default=None, max_length=10000)


class TextToImageRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    model: ImageModelName = "flux2-dev"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None


class ImageToImageRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str
    model: ImageModelName = "flux2-dev"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list  # str for text, list for multimodal


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3-12b-nvfp4"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt(prompt: str, camera_motion: str | None) -> str:
    """Append camera-motion tag to prompt if specified."""
    if camera_motion:
        return f"{prompt} [{camera_motion}]"
    return prompt


def _resolve_keyframes(body: ImageToVideoRequest) -> list[dict] | JSONResponse:
    """Resolve keyframes from an ImageToVideoRequest. Returns list of dicts or JSONResponse on error."""
    if body.keyframes and body.image_uri:
        return _error(422, "Cannot specify both image_uri and keyframes")
    if body.keyframes:
        if len(body.keyframes) == 0:
            return _error(422, "keyframes list must not be empty")
        if len(body.keyframes) > 8:
            return _error(422, "At most 8 keyframes are allowed")
        frame_indices = [kf.frame_index for kf in body.keyframes]
        if len(frame_indices) != len(set(frame_indices)):
            return _error(422, "Duplicate frame_index values are not allowed")
        if frame_indices.count(0) > 1:
            return _error(422, "At most one keyframe can have frame_index 0")
        keyframe_inputs = []
        for kf in body.keyframes:
            path = str(uploads.resolve(kf.image_uri))
            keyframe_inputs.append({"image_path": path, "frame_index": kf.frame_index, "strength": kf.strength})
        return keyframe_inputs
    elif body.image_uri:
        path = str(uploads.resolve(body.image_uri))
        return [{"image_path": path, "frame_index": 0, "strength": 1.0}]
    else:
        return _error(422, "Either image_uri or keyframes is required")


def _error(status: int, msg: str) -> JSONResponse:
    # Avoid leaking internal filesystem paths in error responses
    text = msg[:500]
    if "/mnt/" in text or "/home/" in text or "/tmp/" in text:
        text = "Internal server error"
    return JSONResponse(status_code=status, content={"error": text, "message": text, "detail": text})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "ltx": "ready" if manager.is_ready else "not_loaded",
        "flux": "ready" if flux.is_ready else "not_loaded",
        "chat": "ready" if chat.is_ready else "not_loaded",
        "queue": job_store.stats(),
    }


@app.post("/v1/text-to-video")
async def text_to_video(body: TextToVideoRequest) -> Response:
    if not manager.is_ready:
        return _error(500, "No GPU workers loaded")
    try:
        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        prompt = _build_prompt(body.prompt, body.camera_motion)
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            video_bytes = await manager.generate_text_to_video(
                prompt=prompt,
                model=body.model,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=body.fps,
                seed=seed,
                generate_audio=body.generate_audio,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except Exception as exc:
        logger.exception("text-to-video failed")
        return _error(500, str(exc))


@app.post("/v1/image-to-video")
async def image_to_video(body: ImageToVideoRequest) -> Response:
    keyframe_inputs = _resolve_keyframes(body)
    if isinstance(keyframe_inputs, JSONResponse):
        return keyframe_inputs
    if not manager.is_ready:
        return _error(500, "No GPU workers loaded")
    try:
        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
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
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("image-to-video failed")
        return _error(500, str(exc))


@app.post("/v1/audio-to-video")
async def audio_to_video(body: AudioToVideoRequest) -> Response:
    if not manager.is_ready:
        return _error(500, "No GPU workers loaded")
    try:
        audio_path = str(uploads.resolve(body.audio_uri))
        image_path: str | None = None
        if body.image_uri:
            image_path = str(uploads.resolve(body.image_uri))

        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
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
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("audio-to-video failed")
        return _error(500, str(exc))


@app.post("/v1/retake")
async def retake(body: RetakeRequest) -> Response:
    if not manager.is_ready:
        return _error(500, "No GPU workers loaded")
    try:
        video_path = str(uploads.resolve(body.video_uri))
        prompt = body.prompt or ""
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            video_bytes = await manager.retake(
                video_path=video_path,
                start_time=body.start_time,
                duration=body.duration,
                mode=body.mode,
                prompt=prompt,
                seed=seed,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("retake failed")
        return _error(422, f"Content rejected or generation failed: {exc}")


@app.post("/v1/text-to-image")
async def text_to_image(body: TextToImageRequest) -> Response:
    if not flux.is_ready:
        return _error(500, "Flux pipeline not loaded")
    try:
        width = (body.width // 16) * 16
        height = (body.height // 16) * 16
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

        async with _inference_lock:
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

        async with _inference_lock:
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
# V2 Async Job Endpoints
# ---------------------------------------------------------------------------


def _submit_job(job_type: JobType, params: dict, request: Request) -> JSONResponse:
    """Create a job, enqueue it, return 202."""
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
    width, height = _resolution_to_dims(body.resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    prompt = _build_prompt(body.prompt, body.camera_motion)
    seed = random.randint(0, 2**32 - 1)
    params = dict(prompt=prompt, model=body.model, width=width, height=height,
                  num_frames=num_frames, fps=body.fps, seed=seed, generate_audio=body.generate_audio)
    return _submit_job(JobType.TEXT_TO_VIDEO, params, request)


@app.post("/v2/image-to-video")
async def v2_image_to_video(body: ImageToVideoRequest, request: Request) -> JSONResponse:
    keyframe_inputs = _resolve_keyframes(body)
    if isinstance(keyframe_inputs, JSONResponse):
        return keyframe_inputs
    width, height = _resolution_to_dims(body.resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    seed = random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, keyframes=keyframe_inputs, model=body.model,
                  width=width, height=height, num_frames=num_frames, fps=body.fps,
                  seed=seed, generate_audio=body.generate_audio)
    return _submit_job(JobType.IMAGE_TO_VIDEO, params, request)


@app.post("/v2/audio-to-video")
async def v2_audio_to_video(body: AudioToVideoRequest, request: Request) -> JSONResponse:
    audio_path = str(uploads.resolve(body.audio_uri))
    image_path: str | None = None
    if body.image_uri:
        image_path = str(uploads.resolve(body.image_uri))
    width, height = _resolution_to_dims(body.resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    seed = random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, audio_path=audio_path, image_path=image_path,
                  model=body.model, width=width, height=height, num_frames=num_frames,
                  fps=body.fps, seed=seed)
    return _submit_job(JobType.AUDIO_TO_VIDEO, params, request)


@app.post("/v2/retake")
async def v2_retake(body: RetakeRequest, request: Request) -> JSONResponse:
    video_path = str(uploads.resolve(body.video_uri))
    prompt = body.prompt or ""
    seed = random.randint(0, 2**32 - 1)
    params = dict(video_path=video_path, start_time=body.start_time,
                  duration=body.duration, mode=body.mode, prompt=prompt, seed=seed)
    return _submit_job(JobType.RETAKE, params, request)


@app.post("/v2/text-to-image")
async def v2_text_to_image(body: TextToImageRequest, request: Request) -> JSONResponse:
    width = (body.width // 16) * 16
    height = (body.height // 16) * 16
    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, width=width, height=height,
                  num_inference_steps=body.num_inference_steps,
                  guidance_scale=body.guidance_scale, seed=seed)
    return _submit_job(JobType.TEXT_TO_IMAGE, params, request)


@app.post("/v2/image-to-image")
async def v2_image_to_image(body: ImageToImageRequest, request: Request) -> JSONResponse:
    image_path = str(uploads.resolve(body.image_uri))
    width = (body.width // 16) * 16
    height = (body.height // 16) * 16
    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, image_path=image_path, width=width, height=height,
                  num_inference_steps=body.num_inference_steps,
                  guidance_scale=body.guidance_scale, seed=seed)
    return _submit_job(JobType.IMAGE_TO_IMAGE, params, request)


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
        "queue_position": job_store.queue_position(job.id) if job.status == JobStatus.QUEUED else None,
        "error": {"code": job.error_code or "generation_failed", "message": job.error} if job.error else None,
        "result_url": f"/v2/jobs/{job.id}/result" if job.status == JobStatus.COMPLETED else None,
        "result_media_type": job.result_media_type,
    })


@app.get("/v2/jobs/{job_id}/result")
async def v2_job_result(job_id: str) -> Response:
    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")
    if job.status != JobStatus.COMPLETED or not job.result_uri:
        return _error(409, "Job result not ready")
    try:
        path = uploads.resolve(job.result_uri)
        data = path.read_bytes()
        return Response(
            content=data,
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
