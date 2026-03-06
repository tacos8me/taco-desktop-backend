"""Dual-GPU pipeline manager for LTX-2 video generation.

Loads DistilledPipeline, TI2VidTwoStagesPipeline, RetakePipeline, and
A2VidPipelineTwoStage onto each GPU at startup.  Uses asyncio.Lock per worker
to dispatch requests to whichever GPU is free, and runs blocking pipeline
calls in a thread pool.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.retake import RetakePipeline
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snap_to_multiple(value: int, divisor: int) -> int:
    """Round up to nearest multiple of divisor."""
    return ((value + divisor - 1) // divisor) * divisor


def _resolution_to_dims(resolution: str) -> tuple[int, int]:
    """Parse '1920x1080' into (width, height), snapped to multiples of 64."""
    w, h = resolution.split("x")
    return _snap_to_multiple(int(w), 64), _snap_to_multiple(int(h), 64)


def _duration_to_frames(duration: float, fps: float) -> int:
    """Convert duration (seconds) to frame count, snapped to 8k+1."""
    raw = int(duration * fps)
    # Snap to nearest 8k+1: num_frames = 8*k + 1
    k = max(round((raw - 1) / 8), 1)
    return 8 * k + 1


def _video_to_bytes(video, fps: float, audio, num_frames: int, *, include_audio: bool = True) -> bytes:
    """Encode decoded video + audio to MP4 bytes via a temp file."""
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        encode_video(
            video=video,
            fps=int(fps),
            audio=audio if include_audio else None,
            output_path=tmp_path,
            video_chunks_number=video_chunks_number,
        )
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# GPU Worker
# ---------------------------------------------------------------------------


@dataclass
class GPUWorker:
    device: torch.device
    distilled: DistilledPipeline
    ti2vid: TI2VidTwoStagesPipeline
    retake: RetakePipeline
    a2vid: A2VidPipelineTwoStage
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------------------------------------------------------------------
# Private inference functions (blocking, run in thread pool)
# ---------------------------------------------------------------------------


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
    generate_audio: bool,
) -> bytes:
    if model == "ltx-2-3-fast":
        video, audio = worker.distilled(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            images=[],
            tiling_config=TilingConfig.default(),
        )
    elif model == "ltx-2-3-pro":
        params = detect_params(config.DEV_CHECKPOINT)
        video, audio = worker.ti2vid(
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=params.num_inference_steps,
            video_guider_params=params.video_guider_params,
            audio_guider_params=params.audio_guider_params,
            images=[],
            tiling_config=TilingConfig.default(),
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    return _video_to_bytes(video, fps, audio, num_frames, include_audio=generate_audio)


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
    generate_audio: bool,
) -> bytes:
    images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]

    if model == "ltx-2-3-fast":
        video, audio = worker.distilled(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            images=images,
            tiling_config=TilingConfig.default(),
        )
    elif model == "ltx-2-3-pro":
        params = detect_params(config.DEV_CHECKPOINT)
        video, audio = worker.ti2vid(
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=params.num_inference_steps,
            video_guider_params=params.video_guider_params,
            audio_guider_params=params.audio_guider_params,
            images=images,
            tiling_config=TilingConfig.default(),
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    return _video_to_bytes(video, fps, audio, num_frames, include_audio=generate_audio)


@torch.inference_mode()
def _run_a2v(
    worker: GPUWorker,
    prompt: str,
    audio_path: str,
    image_path: str | None,
    model: str,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    seed: int,
) -> bytes:
    images: list[ImageConditioningInput] = []
    if image_path is not None:
        images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]

    params = detect_params(config.DEV_CHECKPOINT)
    video, audio = worker.a2vid(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=params.num_inference_steps,
        video_guider_params=params.video_guider_params,
        images=images,
        audio_path=audio_path,
        audio_max_duration=num_frames / fps,
        tiling_config=TilingConfig.default(),
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
    end_time = start_time + duration
    regenerate_video = mode in ("replace_audio_and_video", "replace_video", "replace_video_only")
    regenerate_audio = mode in ("replace_audio_and_video", "replace_audio")

    params = detect_params(config.DEV_CHECKPOINT)
    video, audio = worker.retake(
        video_path=video_path,
        prompt=prompt,
        start_time=start_time,
        end_time=end_time,
        seed=seed,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps=params.num_inference_steps,
        video_guider_params=params.video_guider_params,
        audio_guider_params=params.audio_guider_params,
        regenerate_video=regenerate_video,
        regenerate_audio=regenerate_audio,
        tiling_config=TilingConfig.default(),
    )

    fps, num_frames, _, _ = get_videostream_metadata(video_path)
    return _video_to_bytes(video, fps, audio, num_frames)


# ---------------------------------------------------------------------------
# Pipeline Manager
# ---------------------------------------------------------------------------


class PipelineManager:
    """Loads LTX pipelines onto multiple GPUs and dispatches requests."""

    def __init__(self) -> None:
        self.workers: list[GPUWorker] = []

    def load_all(self) -> None:
        """Load all four pipeline types onto each configured GPU."""
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                path=config.DISTILLED_LORA,
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

        for device_str in config.GPU_DEVICES:
            device = torch.device(device_str)
            logger.info("Loading pipelines onto %s ...", device_str)

            distilled = DistilledPipeline(
                distilled_checkpoint_path=config.DISTILLED_CHECKPOINT,
                gemma_root=config.GEMMA_ROOT,
                spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
                loras=[],
                device=device,
            )

            ti2vid = TI2VidTwoStagesPipeline(
                checkpoint_path=config.DEV_CHECKPOINT,
                distilled_lora=distilled_lora,
                spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
                gemma_root=config.GEMMA_ROOT,
                loras=[],
                device=device,
            )

            retake = RetakePipeline(
                checkpoint_path=config.DEV_CHECKPOINT,
                gemma_root=config.GEMMA_ROOT,
                loras=[],
                device=device,
            )

            a2vid = A2VidPipelineTwoStage(
                checkpoint_path=config.DEV_CHECKPOINT,
                distilled_lora=distilled_lora,
                spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
                gemma_root=config.GEMMA_ROOT,
                loras=[],
                device=device,
            )

            worker = GPUWorker(
                device=device,
                distilled=distilled,
                ti2vid=ti2vid,
                retake=retake,
                a2vid=a2vid,
            )
            self.workers.append(worker)
            logger.info("Pipelines loaded on %s", device_str)

    async def _acquire_worker(self) -> GPUWorker:
        """Wait for and return the first unlocked worker."""
        while True:
            for worker in self.workers:
                if not worker.lock.locked():
                    await worker.lock.acquire()
                    return worker
            # All workers busy -- wait briefly then retry
            await asyncio.sleep(0.05)

    async def generate_text_to_video(
        self,
        prompt: str,
        model: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        generate_audio: bool = True,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                _run_t2v,
                worker,
                prompt,
                model,
                width,
                height,
                num_frames,
                fps,
                seed,
                generate_audio,
            )
        finally:
            worker.lock.release()

    async def generate_image_to_video(
        self,
        prompt: str,
        image_path: str,
        model: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        generate_audio: bool = True,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
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
                generate_audio,
            )
        finally:
            worker.lock.release()

    async def generate_audio_to_video(
        self,
        prompt: str,
        audio_path: str,
        image_path: str | None,
        model: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                _run_a2v,
                worker,
                prompt,
                audio_path,
                image_path,
                model,
                width,
                height,
                num_frames,
                fps,
                seed,
            )
        finally:
            worker.lock.release()

    async def retake(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        mode: str,
        prompt: str,
        seed: int,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                _run_retake,
                worker,
                video_path,
                start_time,
                duration,
                mode,
                prompt,
                seed,
            )
        finally:
            worker.lock.release()
