"""FastAPI server implementing the LTX-compatible API for taco-desktop."""

from __future__ import annotations

import logging
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import config
from pipeline_manager import PipelineManager, _duration_to_frames, _resolution_to_dims
from upload_store import UploadStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

manager = PipelineManager()
uploads = UploadStore(config.UPLOAD_DIR)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if config.GPU_DEVICES:
        logger.info("Loading pipelines on %s ...", config.GPU_DEVICES)
        manager.load_all()
        logger.info("Pipelines ready.")
    else:
        logger.warning("No GPU_DEVICES configured — running without inference.")
    yield


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


ModelName = Literal["ltx-2-3-fast", "ltx-2-3-pro"]
Resolution = Literal["1920x1080", "1080x1920", "2560x1440", "1440x2560", "3840x2160", "2160x3840"]
RetakeMode = Literal["replace_audio_and_video", "replace_video", "replace_video_only", "replace_audio"]


class TextToVideoRequest(BaseModel):
    prompt: str
    model: ModelName
    resolution: Resolution
    duration: float
    fps: float
    generate_audio: bool = False
    camera_motion: str | None = None


class ImageToVideoRequest(BaseModel):
    prompt: str
    image_uri: str
    model: ModelName
    resolution: Resolution
    duration: float
    fps: float
    generate_audio: bool = False


class AudioToVideoRequest(BaseModel):
    prompt: str
    audio_uri: str
    image_uri: str | None = None
    model: ModelName
    resolution: Resolution


class RetakeRequest(BaseModel):
    video_uri: str
    start_time: float
    duration: float
    mode: RetakeMode
    prompt: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt(prompt: str, camera_motion: str | None) -> str:
    """Append camera-motion tag to prompt if specified."""
    if camera_motion:
        return f"{prompt} [{camera_motion}]"
    return prompt


def _error(status: int, msg: str) -> JSONResponse:
    text = msg[:500]
    return JSONResponse(status_code=status, content={"error": text, "message": text, "detail": text})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/text-to-video")
async def text_to_video(body: TextToVideoRequest) -> Response:
    if not manager.workers:
        return _error(500, "No GPU workers loaded")
    try:
        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        prompt = _build_prompt(body.prompt, body.camera_motion)
        seed = random.randint(0, 2**32 - 1)

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
    if not manager.workers:
        return _error(500, "No GPU workers loaded")
    try:
        image_path = str(uploads.resolve(body.image_uri))
        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        seed = random.randint(0, 2**32 - 1)

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
    if not manager.workers:
        return _error(500, "No GPU workers loaded")
    try:
        audio_path = str(uploads.resolve(body.audio_uri))
        image_path: str | None = None
        if body.image_uri:
            image_path = str(uploads.resolve(body.image_uri))

        width, height = _resolution_to_dims(body.resolution)
        # Audio-to-video has no duration/fps in the spec; use defaults
        fps = 24.0
        duration = 6.0
        num_frames = _duration_to_frames(duration, fps)
        seed = random.randint(0, 2**32 - 1)

        video_bytes = await manager.generate_audio_to_video(
            prompt=body.prompt,
            audio_path=audio_path,
            image_path=image_path,
            model=body.model,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
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
    if not manager.workers:
        return _error(500, "No GPU workers loaded")
    try:
        video_path = str(uploads.resolve(body.video_uri))
        prompt = body.prompt or ""
        seed = random.randint(0, 2**32 - 1)

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
    data = await request.body()
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
