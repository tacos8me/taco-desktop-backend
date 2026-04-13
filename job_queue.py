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


# ---------------------------------------------------------------------------
# Batch models
# ---------------------------------------------------------------------------


class BatchStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"        # some items succeeded, some failed
    FAILED = "failed"          # all items failed
    CANCELLED = "cancelled"


@dataclass
class BatchItemResult:
    index: int
    type: str
    status: str  # "completed" | "failed" | "cancelled"
    result_uri: str | None = None
    media_type: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass
class BatchJob:
    id: str                          # batch_xxx (different prefix from job IDs)
    items: list[Any]                 # list[BatchItem] — avoid circular import
    status: BatchStatus = BatchStatus.QUEUED
    api_key: str = ""
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None
    total: int = 0                   # len(items)
    completed_count: int = 0
    failed_count: int = 0
    current_index: int = 0           # which item is currently running
    results: list[BatchItemResult] = field(default_factory=list)
    turbo: bool = False              # was turbo mode active when batch started?
    priority: str = "normal"
    callback_url: str | None = None


class BatchStore:
    """In-memory batch storage keyed by batch ID."""

    def __init__(self) -> None:
        self._batches: dict[str, BatchJob] = {}

    def add(self, batch: BatchJob) -> None:
        self._batches[batch.id] = batch

    def get(self, batch_id: str) -> BatchJob | None:
        return self._batches.get(batch_id)

    def remove(self, batch_id: str) -> None:
        self._batches.pop(batch_id, None)

    def active_count(self) -> int:
        """Number of batches that are queued or processing."""
        return sum(
            1 for b in self._batches.values()
            if b.status in (BatchStatus.QUEUED, BatchStatus.PROCESSING)
        )

    def all_batches(self) -> list[BatchJob]:
        return list(self._batches.values())


def make_batch_id() -> str:
    """Generate an unguessable batch ID with batch_ prefix."""
    return "batch_" + secrets.token_urlsafe(16)


# ---------------------------------------------------------------------------
# Job models
# ---------------------------------------------------------------------------


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
    IMAGE_EDIT = "image-edit"
    EXPORT_COMPOSITION = "export-composition"
    MUSIC_GENERATION = "music-generation"


_MEDIA_TYPES: dict[JobType, str] = {
    JobType.TEXT_TO_VIDEO: "video/mp4",
    JobType.IMAGE_TO_VIDEO: "video/mp4",
    JobType.AUDIO_TO_VIDEO: "video/mp4",
    JobType.RETAKE: "video/mp4",
    JobType.TEXT_TO_IMAGE: "image/webp",
    JobType.IMAGE_TO_IMAGE: "image/webp",
    JobType.IMAGE_EDIT: "image/webp",
    JobType.EXPORT_COMPOSITION: "video/mp4",
    JobType.MUSIC_GENERATION: "audio/mpeg",
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
    # Coarse post-denoise phase: "denoising" | "decoding" | "encoding" | "saving" | None.
    # Progress alone cannot convey what's happening in the invisible 0.90→1.0 tail
    # (VAE decode + ffmpeg encode + thumbnail write), so clients can render a phase
    # label instead of a frozen percentage.
    phase: str | None = None
    current_step: int = 0
    total_steps: int = 0
    result_uri: str | None = None
    result_media_type: str | None = None
    error: str | None = None
    error_code: str | None = None
    preview_bytes: bytes | None = None


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


def make_flux_callback(job: Job, total_steps: int) -> Callable:
    """Create a diffusers callback_on_step_end for Flux progress tracking.

    Denoising is capped at 0.90 — the remaining 0.10 is reserved for post-denoise
    phases (encoding, saving) that are emitted by the dispatch layer.
    """
    def callback(pipe: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
        if job.status == JobStatus.CANCELLED:
            return callback_kwargs
        job.progress = min((step / max(total_steps, 1)) * 0.9, 0.90)
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
    history: "HistoryStore | None" = None,
    turbo_check: Callable[[Job], bool] | Callable[[], bool] | None = None,
) -> None:
    """Background worker that processes jobs from the queue.

    In turbo mode (turbo_check returns True), the inference lock is SKIPPED
    because SplitModelManager._acquire_worker() handles per-GPU serialization
    via worker.lock. This allows 2 worker_loop instances to dispatch 2 video
    jobs concurrently on 2 GPUs.

    turbo_check receives the Job being dispatched so it can decide per-job
    whether the lock is needed (e.g. Flux image jobs still need the lock
    even when turbo is active for video jobs).
    """
    logger.info("Queue worker started")
    while True:
        job_id = await queue.get()
        job = job_store.get(job_id)
        if job is None or job.status == JobStatus.CANCELLED:
            queue.task_done()
            continue

        job.status = JobStatus.PROCESSING
        job.started_at = time.monotonic()
        job.phase = "denoising"
        logger.info("Processing job %s (%s)", job.id, job.type)

        result_bytes: bytes | None = None
        try:
            if turbo_check and turbo_check(job):
                # Turbo: skip inference lock — 2 LTX workers handle per-GPU
                # serialization via SplitModelManager._acquire_worker(). Both
                # worker_loop instances can dispatch concurrently.
                result_bytes = await dispatch_fn(job)
            else:
                async with inference_lock:
                    result_bytes = await dispatch_fn(job)

            # Dispatch returned bytes; the final step is moving them into the
            # upload store and flipping status. Surface this as "saving" so
            # the progress bar keeps advancing while the disk write runs.
            job.phase = "saving"
            job.progress = 0.99
            upload_id, storage_uri = uploads.create()
            uploads.save(upload_id, result_bytes)

            job.result_uri = storage_uri
            job.result_media_type = _MEDIA_TYPES.get(job.type, "application/octet-stream")
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.phase = None
            elapsed = time.monotonic() - (job.started_at or job.created_at)
            logger.info("Job %s completed in %.1fs", job.id, elapsed)

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.phase = None
            job.error = str(exc)[:500]
            job.error_code = "cuda_oom" if "out of memory" in str(exc).lower() else "generation_failed"
            logger.exception("Job %s failed", job.id)

        finally:
            job.completed_at = time.monotonic()
            # Fire-and-forget history save: PyAV first-frame decode + JPEG encode
            # + SQLite commit together take 130–430 ms for a 1080p video, which
            # would otherwise block the worker from dequeuing the next job. The
            # task also backfills job.preview_bytes from the on-disk thumbnail so
            # /v2/jobs/{id}/preview hits the fast path on the next poll.
            if history and job.api_key and result_bytes is not None:
                _params = job.params or {}
                _captured = dict(
                    job_id=job.id,
                    api_key=job.api_key,
                    job_type=job.type,
                    prompt=_params.get("prompt", ""),
                    model=_params.get("model"),
                    width=_params.get("width", 0),
                    height=_params.get("height", 0),
                    turbo=_params.get("turbo", False),
                    status=job.status,
                    result_uri=job.result_uri,
                    result_bytes=result_bytes,
                    created_at=time.time(),
                    completed_at=time.time(),
                    error=job.error,
                )
                _job_id_for_log = job.id
                _result_uri = job.result_uri
                _job_ref = job

                async def _save_and_populate(
                    captured: dict = _captured,
                    jid: str = _job_id_for_log,
                    result_uri: str | None = _result_uri,
                    j: Job = _job_ref,
                ) -> None:
                    try:
                        t0 = time.perf_counter()
                        await asyncio.to_thread(history.save, **captured)
                        logger.info("history.save: %.2fs (job %s)", time.perf_counter() - t0, jid)
                        # Backfill preview_bytes from the thumbnail we just wrote,
                        # so the /preview endpoint's fast path hits on next poll.
                        if result_uri and not j.preview_bytes:
                            upload_id = result_uri.removeprefix("storage://")
                            thumb_path = config.THUMBNAIL_DIR / f"thumb_{upload_id}"
                            if thumb_path.exists():
                                j.preview_bytes = thumb_path.read_bytes()
                    except Exception:
                        logger.warning("Failed to save job %s to history", jid, exc_info=True)

                asyncio.create_task(_save_and_populate())
            queue.task_done()


async def cleanup_loop(job_store: JobStore, uploads: UploadStore) -> None:
    """Periodically remove expired job metadata. Result files for completed jobs are kept (managed by history store)."""
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
            if job and job.result_uri and job.status != JobStatus.COMPLETED:
                # Only delete result files for failed/cancelled jobs — completed results are managed by history
                upload_id = job.result_uri.removeprefix("storage://")
                path = uploads.base_dir / upload_id
                path.unlink(missing_ok=True)
            job_store.remove(job_id)

        if to_remove:
            logger.info("Cleaned up %d expired jobs", len(to_remove))
