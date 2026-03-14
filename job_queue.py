"""Async job queue for generation requests.

Submit-poll-fetch pattern to work around Cloudflare's 100s timeout.
Single asyncio.Queue + one background worker (serialized by _inference_lock).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

import config
from upload_store import UploadStore

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(StrEnum):
    TEXT_TO_VIDEO = "text-to-video"
    IMAGE_TO_VIDEO = "image-to-video"
    AUDIO_TO_VIDEO = "audio-to-video"
    RETAKE = "retake"
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


_MEDIA_TYPES: dict[JobType, str] = {
    JobType.TEXT_TO_VIDEO: "video/mp4",
    JobType.IMAGE_TO_VIDEO: "video/mp4",
    JobType.AUDIO_TO_VIDEO: "video/mp4",
    JobType.RETAKE: "video/mp4",
    JobType.TEXT_TO_IMAGE: "image/webp",
    JobType.IMAGE_TO_IMAGE: "image/webp",
}


@dataclass
class Job:
    id: str
    type: JobType
    status: JobStatus = JobStatus.QUEUED
    params: dict[str, Any] = field(default_factory=dict)
    api_key: str = ""
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    result_uri: str | None = None
    result_media_type: str | None = None
    error: str | None = None
    error_code: str | None = None


class JobStore:
    """In-memory job storage keyed by job ID."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def add(self, job: Job) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def count_by_status(self, status: JobStatus) -> int:
        return sum(1 for j in self._jobs.values() if j.status == status)

    def queue_position(self, job_id: str) -> int | None:
        """Return 0-based queue position, or None if not queued."""
        pos = 0
        for j in self._jobs.values():
            if j.status == JobStatus.QUEUED:
                if j.id == job_id:
                    return pos
                pos += 1
        return None

    def pending_count(self) -> int:
        return self.count_by_status(JobStatus.QUEUED)

    def stats(self) -> dict[str, int]:
        s: dict[str, int] = {}
        for j in self._jobs.values():
            s[j.status] = s.get(j.status, 0) + 1
        return s


def make_job_id() -> str:
    """Generate an unguessable job ID (128-bit entropy)."""
    return secrets.token_urlsafe(16)


def update_progress(job: Job, step: int, total_steps: int) -> None:
    """Update job progress from a denoising step callback."""
    job.current_step = step
    job.total_steps = total_steps
    job.progress = min(step / max(total_steps, 1), 0.99)


def make_progress_callback(job: Job, total_steps: int, stage_offset: float = 0.0, stage_scale: float = 1.0) -> Callable:
    """Create a step-counting wrapper for LTX denoise functions.

    Returns a function that wraps a denoise_fn call and updates job progress.
    For multi-stage pipelines, use stage_offset/stage_scale to map to 0-1 range.
    """
    call_count = 0

    def wrapper(original_fn: Callable) -> Callable:
        def counting_fn(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            result = original_fn(*args, **kwargs)
            call_count += 1
            raw_progress = min(call_count / max(total_steps, 1), 1.0)
            job.progress = min(stage_offset + raw_progress * stage_scale, 0.99)
            job.current_step = call_count
            job.total_steps = total_steps
            return result
        return counting_fn
    return wrapper


def make_flux_callback(job: Job, total_steps: int) -> Callable:
    """Create a diffusers callback_on_step_end for Flux progress tracking."""
    def callback(pipe: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
        job.progress = min(step / max(total_steps, 1), 0.99)
        job.current_step = step
        job.total_steps = total_steps
        return callback_kwargs
    return callback


async def worker_loop(
    job_store: JobStore,
    queue: asyncio.Queue[str],
    inference_lock: asyncio.Lock,
    dispatch_fn: Callable,
    uploads: UploadStore,
) -> None:
    """Background worker that processes jobs from the queue."""
    logger.info("Queue worker started")
    while True:
        job_id = await queue.get()
        job = job_store.get(job_id)
        if job is None or job.status == JobStatus.CANCELLED:
            queue.task_done()
            continue

        job.status = JobStatus.PROCESSING
        job.started_at = time.monotonic()
        logger.info("Processing job %s (%s)", job.id, job.type)

        try:
            async with inference_lock:
                result_bytes = await dispatch_fn(job)

            upload_id, storage_uri = uploads.create()
            uploads.save(upload_id, result_bytes)

            job.result_uri = storage_uri
            job.result_media_type = _MEDIA_TYPES.get(job.type, "application/octet-stream")
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            elapsed = time.monotonic() - (job.started_at or job.created_at)
            logger.info("Job %s completed in %.1fs", job.id, elapsed)

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)[:500]
            job.error_code = "cuda_oom" if "out of memory" in str(exc).lower() else "generation_failed"
            logger.exception("Job %s failed", job.id)

        finally:
            job.completed_at = time.monotonic()
            queue.task_done()


async def cleanup_loop(job_store: JobStore, uploads: UploadStore) -> None:
    """Periodically remove expired jobs and their result files."""
    ttl = config.JOB_RESULT_TTL_SECONDS
    logger.info("Cleanup loop started (TTL=%ds)", ttl)
    while True:
        await asyncio.sleep(60)
        now = time.monotonic()
        to_remove: list[str] = []
        for job in list(job_store._jobs.values()):
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at and (now - job.completed_at) > ttl:
                    to_remove.append(job.id)

        for job_id in to_remove:
            job = job_store.get(job_id)
            if job and job.result_uri:
                upload_id = job.result_uri.removeprefix("storage://")
                path = uploads.base_dir / upload_id
                path.unlink(missing_ok=True)
            job_store.remove(job_id)

        if to_remove:
            logger.info("Cleaned up %d expired jobs", len(to_remove))
