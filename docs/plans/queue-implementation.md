# Queue Implementation Plan

Internal design for the async job queue in taco-backend.

## 1. Job Data Model

```python
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"

class JobType(str, Enum):
    text_to_video = "text-to-video"
    image_to_video = "image-to-video"
    audio_to_video = "audio-to-video"
    retake = "retake"
    text_to_image = "text-to-image"
    image_to_image = "image-to-image"

class Job(BaseModel):
    id: str                           # UUID hex (matches upload_store format)
    type: JobType
    status: JobStatus = JobStatus.queued
    params: dict                      # Validated request body as dict
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0            # 0.0 to 1.0
    current_step: int = 0
    total_steps: int = 0
    result_uri: str | None = None    # storage:// URI when completed
    result_media_type: str | None = None  # "video/mp4" or "image/png"
    error: str | None = None         # Error message when failed
```

### Field rationale

- `id`: UUID hex, same format as upload_store IDs. Generated at submission time.
- `params`: Store the validated Pydantic model as a dict. This avoids a union type per request model and keeps the Job model generic. The worker dispatches based on `type`.
- `progress` / `current_step` / `total_steps`: Separate fields for step-level granularity. `progress` is the derived 0-1 float for simple client polling. Both are needed because different models have wildly different step counts (fast=8 stage1 + 2 stage2, pro=30+2, flux=50).
- `result_uri`: A `storage://` URI pointing to the output file in upload_store. Null until completed.
- `result_media_type`: Needed so the GET result endpoint knows what Content-Type to return.

## 2. Job Store

In-memory dict keyed by job ID. No persistence across restarts — this is a single-user local server.

```python
class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}   # id -> Job

    def add(self, job: Job) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> Job:
        job = self._jobs[job_id]
        for k, v in fields.items():
            setattr(job, k, v)
        return job

    def list_recent(self, limit: int = 50) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
```

### Why not SQLite / Redis?

- Single-user server, jobs complete in 15-60s, no persistence requirement.
- Adding a database creates deployment complexity for zero benefit.
- If persistence becomes needed later, the `JobStore` interface can swap to SQLite without changing callers.

## 3. Queue Mechanism

`asyncio.Queue` feeding a single background worker task.

```python
import asyncio

_job_queue: asyncio.Queue[str] = asyncio.Queue()  # holds job IDs, not full Job objects
```

### Submit flow

1. Endpoint validates request, creates `Job` with `status=queued`, adds to `JobStore`.
2. Puts `job.id` onto `_job_queue`.
3. Returns `{"job_id": job.id}` immediately (202 Accepted).

### Why Queue[str] not Queue[Job]?

Job objects live in the store and are mutated in-place (progress updates). The queue only needs the ID to look up the current state.

### Queue size

Unbounded (`maxsize=0`). With a single user, the queue will rarely have more than 1-2 items. If bounded behavior is ever needed, the submit endpoint can check `_job_queue.qsize()` and reject with 429.

## 4. Background Worker

A single `asyncio.Task` started in the FastAPI lifespan. It pulls job IDs from the queue and dispatches to the correct generation function.

```python
async def _worker_loop(
    job_store: JobStore,
    queue: asyncio.Queue[str],
    inference_lock: asyncio.Lock,
    managers: dict,        # {"ltx": SplitModelManager, "flux": FluxManager}
    upload_store: UploadStore,
) -> None:
    while True:
        job_id = await queue.get()
        job = job_store.get(job_id)
        if job is None:
            queue.task_done()
            continue

        job_store.update(job_id, status=JobStatus.running, started_at=datetime.utcnow())
        try:
            async with inference_lock:
                result_bytes = await _dispatch(job, managers)

            # Store result via upload_store
            _, storage_uri = upload_store.create()
            upload_id = storage_uri.removeprefix("storage://")
            upload_store.save(upload_id, result_bytes)

            job_store.update(
                job_id,
                status=JobStatus.completed,
                completed_at=datetime.utcnow(),
                progress=1.0,
                result_uri=storage_uri,
                result_media_type=_media_type_for(job.type),
            )
        except Exception as exc:
            job_store.update(
                job_id,
                status=JobStatus.failed,
                completed_at=datetime.utcnow(),
                error=str(exc)[:500],
            )
        finally:
            queue.task_done()
```

### Dispatch routing

```python
async def _dispatch(job: Job, managers: dict) -> bytes:
    p = job.params
    match job.type:
        case JobType.text_to_video:
            return await managers["ltx"].generate_text_to_video(**p)
        case JobType.image_to_video:
            return await managers["ltx"].generate_image_to_video(**p)
        case JobType.audio_to_video:
            return await managers["ltx"].generate_audio_to_video(**p)
        case JobType.retake:
            return await managers["ltx"].retake(**p)
        case JobType.text_to_image:
            return await managers["flux"].generate_text_to_image(**p)
        case JobType.image_to_image:
            return await managers["flux"].generate_image_to_image(**p)
```

The `params` dict is constructed at submission time to match the exact kwargs of the manager's generate method. This keeps the dispatch dead simple — it just unpacks the dict.

### Why a single worker, not a worker pool?

- The `_inference_lock` already serializes all GPU work. A pool of workers would just contend on the same lock.
- Both GPUs live in one process. Concurrent GPU inference causes CUBLAS crashes (see MEMORY.md). One job at a time is correct.
- `SplitModelManager._acquire_worker()` handles dual-GPU selection internally; the queue worker doesn't need to know about GPU topology.

## 5. Progress Tracking

### How LTX denoising works

LTX uses `euler_denoising_loop(sigmas, ...)` where `len(sigmas) - 1` = number of steps. The loop calls `denoise_fn` once per step. There is no built-in callback mechanism in ltx-pipelines' euler loop.

### How Flux denoising works

Flux uses diffusers' `__call__` pipeline which supports `callback_on_step_end(pipe, step, timestep, callback_kwargs)`.

### Proposed approach: step-counting wrapper

Rather than modifying ltx-pipelines internals, wrap the denoising function to count steps:

```python
def _make_counting_denoise_fn(original_fn, job_store, job_id, stage_offset, total_steps):
    """Wrap a denoise_fn to update job progress on each call."""
    def counting_fn(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        step = stage_offset + counting_fn._call_count
        counting_fn._call_count += 1
        progress = min(step / total_steps, 0.99)  # cap at 0.99, 1.0 = done
        job_store.update(job_id, current_step=step, progress=progress)
        return result
    counting_fn._call_count = 0
    return counting_fn
```

This wraps `simple_denoising_func(...)` or `multi_modal_guider_factory_denoising_func(...)` before passing to `euler_denoising_loop`. Since the denoise function runs in a thread executor, updates are written to the `Job` object in memory. The polling endpoint reads from the main thread — no thread-safety issue because Python's GIL protects dict/attribute writes.

For Flux, use diffusers' native `callback_on_step_end`:

```python
def _flux_progress_callback(job_store, job_id, total_steps):
    def callback(pipe, step, timestep, callback_kwargs):
        progress = min(step / total_steps, 0.99)
        job_store.update(job_id, current_step=step, total_steps=total_steps, progress=progress)
        return callback_kwargs
    return callback
```

### Step count estimates per pipeline

| Pipeline | Stage 1 steps | Stage 2 steps | Total |
|----------|--------------|---------------|-------|
| t2v-fast (distilled) | 8 | 2 | 10 |
| t2v-pro (dev) | 30 | 2 | 32 |
| i2v-fast | 8 | 2 | 10 |
| i2v-pro | 30 | 2 | 32 |
| a2v | 30 | 2 | 32 |
| retake | 30 | 0 | 30 |
| flux-t2i | 50 | 0 | 50 |
| flux-i2i | 50 | 0 | 50 |

These are the denoising steps only. Encoding, decoding, and upsampling are not tracked (they're fast relative to denoising). Progress jumps from 0.99 to 1.0 when the full pipeline completes.

### Implementation change required

The progress-counting wrapper needs to be plumbed into the `_run_t2v`, `_run_i2v`, `_run_a2v`, and `_run_retake` methods. Two approaches:

**Option A (minimal change):** Add optional `job_store` and `job_id` parameters to each `_run_*` method. When provided, wrap the denoise_fn. When None (legacy sync path), skip wrapping. This is the recommended approach — it keeps the existing sync API working and adds progress as an opt-in layer.

**Option B (refactor):** Create a `ProgressTracker` protocol and inject it. Overkill for single-user.

## 6. Worker Lifecycle

### Startup

In the FastAPI lifespan, after loading models:

```python
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ... load models ...

    worker_task = asyncio.create_task(
        _worker_loop(job_store, _job_queue, _inference_lock, managers, uploads),
        name="queue-worker",
    )

    yield

    # Graceful shutdown
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
```

### Graceful shutdown

When the server receives SIGTERM/SIGINT:
1. Uvicorn triggers lifespan shutdown (the `yield` returns).
2. We cancel the worker task.
3. If a job is in-progress, `CancelledError` propagates. The `finally` block in `_worker_loop` calls `queue.task_done()`.
4. The in-progress job stays in `running` state — since there's no persistence, this is fine (server restart = clean slate).

### What about in-flight HTTP requests?

The async job endpoints return immediately (202), so there are no long-lived HTTP connections to worry about. The only concern is the blocking `run_in_executor` call inside the manager. When cancelled, the executor thread continues to completion (Python can't interrupt threads), but the result is discarded. This is acceptable — the GPU work finishes but we don't store the result.

## 7. Result Storage

Results are stored via `upload_store` — the same mechanism used for user uploads.

### Flow

1. Worker calls the generate function, gets `bytes` back.
2. Worker calls `upload_store.create()` to get a new `(upload_id, storage_uri)`.
3. Worker calls `upload_store.save(upload_id, result_bytes)` to write the file.
4. Worker sets `job.result_uri = storage_uri`.
5. Client polls job status, sees `completed`, gets `result_uri`.
6. Client fetches result via a new `GET /v1/jobs/{job_id}/result` endpoint that resolves the URI and streams the file.

### Why reuse upload_store?

- Same file format and URI scheme (`storage://`).
- Already handles UUID validation, directory creation.
- Result files are indistinguishable from uploads — both are blobs on disk.
- No new storage abstraction needed.

### Alternative considered: inline base64

Rejected. Video files are 5-50MB. Base64 adds 33% overhead and forces the entire payload into the JSON status response.

## 8. Memory and Cleanup Strategy

### Result file lifecycle

Result files persist on disk until explicitly cleaned up. Two strategies:

**Strategy A (recommended): TTL-based cleanup**

A periodic cleanup task runs every N minutes and deletes result files older than a configurable TTL (default: 1 hour). This is simple and predictable.

```python
async def _cleanup_loop(job_store: JobStore, upload_store: UploadStore, ttl_seconds: int = 3600):
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        cutoff = datetime.utcnow() - timedelta(seconds=ttl_seconds)
        for job in list(job_store._jobs.values()):
            if job.status in (JobStatus.completed, JobStatus.failed) and job.completed_at and job.completed_at < cutoff:
                if job.result_uri:
                    upload_id = job.result_uri.removeprefix("storage://")
                    path = upload_store.base_dir / upload_id
                    path.unlink(missing_ok=True)
                job_store.remove(job.id)
```

**Strategy B: Client-acknowledged cleanup**

Client sends `DELETE /v1/jobs/{job_id}` after downloading the result. Riskier — if the client crashes, files leak forever.

**Recommendation:** Use both. TTL cleanup as the safety net, with an optional DELETE endpoint for eager cleanup.

### Job dict cleanup

Job objects are removed from the in-memory dict when the TTL cleanup runs. With ~1KB per Job object and jobs completing every 15-60s, memory usage is negligible even without cleanup. The cleanup is primarily for disk space (result files).

### GPU memory

No change needed. The existing `cleanup_memory()` calls in `_run_t2v` etc. handle GPU memory between stages. The queue worker doesn't hold any GPU resources itself.

## 9. Interaction with `_inference_lock`

### Current state

`_inference_lock` is an `asyncio.Lock` in `server.py` that wraps every generation call. It exists because FP8 layerwise casting in diffusers causes CUBLAS crashes when Flux and LTX run GPU inference concurrently.

### With the queue: worker uses the lock, does not replace it

The background worker acquires `_inference_lock` before calling any generate function. This is necessary because:

1. **The lock must remain.** The CUBLAS bug is a hardware/driver issue, not an application logic issue. Removing the lock re-exposes the crash.
2. **Legacy sync endpoints may coexist during migration.** If we keep the old sync endpoints temporarily (for backward compatibility during frontend migration), both the queue worker and sync endpoints need the same lock.
3. **The worker is the only consumer post-migration.** Once the frontend is fully migrated to async endpoints, the sync endpoints are removed and the worker is the only thing acquiring the lock. At that point, the lock is still correct — it serializes jobs — but it could technically be removed since the single worker already serializes naturally. Keep it anyway as a safety net.

### Lock acquisition pattern

```
Client POST /v1/jobs → 202 (no lock needed)
Client GET /v1/jobs/{id} → 200 (no lock needed)
Worker: await queue.get() → got job_id
Worker: async with _inference_lock: → blocks until GPU free
Worker:     result = await manager.generate_*(**params)
Worker: upload_store.save(result)
Worker: job.status = completed
```

The lock is acquired OUTSIDE the try/except so that a CancelledError during lock acquisition doesn't mark the job as failed.

### SplitModelManager's internal worker lock

`SplitModelManager._acquire_worker()` has its own per-GPU `asyncio.Lock`. With a single queue worker, this lock is never contended — the worker always gets the first (and only) available GPU worker. The internal lock is harmless and should be left in place in case multi-GPU denoising is re-enabled later.

### FluxManager's internal lock

`FluxManager._lock` is similarly never contended with a single queue worker. Leave it in place.

## 10. Summary of New/Modified Files

| File | Change |
|------|--------|
| `job_queue.py` (new) | `Job`, `JobStatus`, `JobType`, `JobStore`, `_worker_loop`, `_dispatch`, progress helpers |
| `server.py` | Add async job endpoints, start worker in lifespan, cleanup task |
| `split_model_manager.py` | Add optional `job_store`/`job_id` params to `_run_*` methods for progress tracking |
| `flux_manager.py` | Add optional `callback_on_step_end` pass-through for progress tracking |
| `upload_store.py` | No changes needed |
| `config.py` | Add `JOB_RESULT_TTL_SECONDS = 3600` |
