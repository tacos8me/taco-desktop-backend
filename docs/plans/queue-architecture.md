# Async Job Queue Architecture

## Problem

Cloudflare Tunnel enforces a 100-second HTTP response timeout. Some generation
requests (high-res video, pro model, audio+video) can exceed this. The server
must accept work immediately and let clients poll for results.

## Design Principles

- Single-process, in-memory. No Redis, no Celery, no external broker.
- Reuse existing `UploadStore` for result file storage.
- Preserve the `_inference_lock` — it exists to prevent concurrent CUDA
  inference (CUBLAS crash with FP8 layerwise casting). The queue worker must
  hold this lock during GPU work, exactly like today's endpoints.
- Chat endpoint stays synchronous (fast, no GPU contention).

---

## 1. Job Lifecycle

```
submit ──► queued ──► processing ──► completed
                                 └──► failed
```

| State        | Meaning                                        |
|--------------|------------------------------------------------|
| `queued`     | Job accepted, waiting in the FIFO queue.       |
| `processing` | Worker has dequeued the job and is running it.  |
| `completed`  | Result file written, ready for download.        |
| `failed`     | Generation raised an exception. Error stored.   |

A job is immutable once `completed` or `failed`. Clients poll status until
terminal.

## 2. Data Flow (text description)

1. **Client POST** `/v1/text-to-video` (or any generation endpoint).
2. Endpoint validates the request body (Pydantic). On validation error, return
   422 immediately — no job created.
3. Endpoint creates a `Job` object (unique ID, status `queued`, serialized
   request params, creation timestamp).
4. Endpoint puts the job ID into an `asyncio.Queue`.
5. Endpoint returns **202 Accepted** with JSON:
   `{"job_id": "<id>", "status": "queued", "poll_url": "/v1/jobs/<id>"}`.
6. **Background worker** (a single `asyncio.Task` started at lifespan) loops:
   `job_id = await queue.get()`, acquires `_inference_lock`, runs generation,
   writes result bytes to `UploadStore`, updates job status.
7. **Client polls** `GET /v1/jobs/<id>`. Returns current status. When
   `completed`, response includes `"result_url": "/v1/jobs/<id>/result"`.
8. **Client fetches** `GET /v1/jobs/<id>/result` — returns raw binary
   (video/mp4 or image/png) with appropriate Content-Type.

## 3. Job Storage

In-memory `dict[str, Job]` keyed by job ID. No persistence across restarts
(acceptable for 1-3 LAN users; jobs are short-lived).

```python
@dataclass
class Job:
    id: str                          # uuid4().hex
    status: Literal["queued", "processing", "completed", "failed"]
    endpoint: str                    # e.g. "text-to-video", "text-to-image"
    params: dict                     # validated request params (serializable)
    created_at: float                # time.monotonic()
    started_at: float | None = None
    finished_at: float | None = None
    result_uri: str | None = None    # storage://<id> when completed
    result_media_type: str | None = None  # "video/mp4" or "image/png"
    error: str | None = None         # error message when failed
```

**Capacity**: Keep at most ~100 recent jobs in the dict. Evict completed/failed
jobs older than 1 hour on each new submission (simple sweep, not a background
timer).

## 4. Queue Strategy

```python
_job_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=20)
_jobs: dict[str, Job] = {}
```

- **Single FIFO queue** — both LTX and Flux jobs go in the same queue because
  they share a single `_inference_lock` anyway (concurrent GPU inference is
  forbidden).
- **maxsize=20** — if the queue is full, the submit endpoint returns 503
  ("Server busy, try again later"). This prevents unbounded memory growth.
- **Single worker** — one `asyncio.Task` consumes from the queue. This is the
  simplest correct design given the serialized lock.

### Why not separate queues per GPU?

The `_inference_lock` already serializes ALL inference across both GPUs. Even
with two queues and two workers, only one would run at a time. A single queue
is simpler and equivalent in throughput. If the CUBLAS concurrency bug is ever
fixed, the architecture can be split into per-GPU queues later.

## 5. Result Storage and Retrieval

Results are stored via `UploadStore` — the same mechanism used for input file
uploads. This reuses the existing UUID-based file management and avoids a
second storage abstraction.

- On completion, the worker calls `uploads.create()` to get a fresh
  `(upload_id, storage_uri)`, then `uploads.save(upload_id, result_bytes)`.
- `job.result_uri` is set to the `storage_uri`.
- The `/v1/jobs/<id>/result` endpoint resolves the storage URI and streams the
  file back with the correct Content-Type.

**Cleanup**: Result files follow the same lifecycle as upload files. A periodic
cleanup (or the eviction sweep from section 3) can delete result files for
expired jobs.

## 6. How `_inference_lock` Interacts with Queue Worker

Today, each HTTP handler acquires `_inference_lock` directly:

```python
# Current pattern (server.py)
async with _inference_lock:
    video_bytes = await manager.generate_text_to_video(...)
```

With the queue, the **worker task** is the only code that acquires the lock:

```python
# Worker loop (simplified)
async def _worker():
    while True:
        job_id = await _job_queue.get()
        job = _jobs[job_id]
        job.status = "processing"
        job.started_at = time.monotonic()
        try:
            async with _inference_lock:
                result_bytes, media_type = await _dispatch(job)
            upload_id, storage_uri = uploads.create()
            uploads.save(upload_id, result_bytes)
            job.result_uri = storage_uri
            job.result_media_type = media_type
            job.status = "completed"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)[:500]
        finally:
            job.finished_at = time.monotonic()
            _job_queue.task_done()
```

The HTTP endpoints no longer touch `_inference_lock` at all — they just enqueue
and return 202. This cleanly separates request handling from GPU work.

**Important**: The lock is still necessary even with a single worker, because
chat or future endpoints might also need GPU access. The lock remains the
single source of truth for "is the GPU busy."

## 7. Endpoint Summary

### New endpoints

| Method | Path                    | Purpose                              |
|--------|-------------------------|--------------------------------------|
| GET    | `/v1/jobs/<id>`         | Poll job status                      |
| GET    | `/v1/jobs/<id>/result`  | Download result (completed jobs only)|

### Modified endpoints

All generation endpoints (`text-to-video`, `image-to-video`, `audio-to-video`,
`retake`, `text-to-image`, `image-to-image`) change from:

- Synchronous: validate → lock → generate → return binary

To:

- Async submit: validate → create job → enqueue → return 202 JSON

### Unchanged endpoints

- `GET /health` — add queue depth info (`"queue_depth": N`)
- `POST /v1/upload`, `PUT /uploads/put/<id>` — unchanged
- `POST /v1/chat/completions` — unchanged (fast, no GPU contention)

## 8. Error Handling

- **Validation errors**: Return 422 immediately, no job created.
- **Queue full**: Return 503 with `Retry-After` header.
- **Generation failure**: Job moves to `failed`, error message stored. Client
  sees `{"status": "failed", "error": "..."}` on poll.
- **Server restart**: All in-memory jobs are lost. Clients polling old job IDs
  get 404. This is acceptable for the LAN use case.
- **File not found** (upload URIs in request): Detected at enqueue time if
  possible (validate that storage URIs resolve before accepting the job). If
  the file disappears between enqueue and processing, the job fails normally.

## 9. Job Cancellation (optional, low priority)

A `DELETE /v1/jobs/<id>` endpoint could remove a queued job. For `processing`
jobs, cancellation is harder (would require cooperative cancellation in the
generation code). Defer this — the 1-3 user scenario makes it low priority.

## 10. Migration Path

The refactor is backward-incompatible for generation endpoints (202 + poll
instead of blocking response). The frontend must be updated simultaneously.
However, the change is contained:

1. Add `Job` dataclass and `_job_queue` to server.py (or a new `job_store.py`).
2. Add worker task to lifespan.
3. Rewrite each generation endpoint to enqueue instead of blocking.
4. Add `/v1/jobs/<id>` and `/v1/jobs/<id>/result` endpoints.
5. Update frontend to submit + poll + download.

No changes needed to `split_model_manager.py`, `flux_manager.py`,
`upload_store.py`, `config.py`, or `chat_manager.py`.
