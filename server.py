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

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


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
    yield


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


ModelName = Literal["ltx-2-3-fast", "ltx-2-3-pro"]
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


class ImageToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str
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
    if not manager.is_ready:
        return _error(500, "No GPU workers loaded")
    try:
        image_path = str(uploads.resolve(body.image_uri))
        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            video_bytes = await manager.generate_image_to_video(
                prompt=body.prompt,
                image_path=image_path,
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
