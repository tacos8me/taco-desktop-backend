# Phase 3: Batch Scheduler + Turbo Mode

**Status:** Spec  
**Depends on:** Phase 2 (ACE integration) must be merged first — server.py, config.py, job_queue.py all change.

---

## Overview

Batch mode lets a client submit N image/video jobs as a single request. The server processes them sequentially under one inference lock acquisition, avoiding per-job swap overhead. Turbo mode is a GPU topology change: temporarily claim cuda:1 from ACE/JoyAI for parallel Flux + LTX inference (dual-GPU, zero swap). Both features compose: a batch can run in turbo mode for maximum throughput.

---

## 1. Pydantic Models

### `BatchItem`

```python
class BatchItem(BaseModel):
    """One generation request inside a batch."""
    type: Literal["text-to-image", "image-to-image", "image-edit",
                   "text-to-video", "image-to-video"]
    params: dict[str, Any]
    # params is the same body you'd POST to the corresponding v1/v2 endpoint,
    # minus the auth header. Examples:
    #   text-to-image: {prompt, model, width, height, lora, turbo, ...}
    #   text-to-video: {prompt, model, resolution, duration, ...}
```

### `BatchRequest`

```python
class BatchRequest(BaseModel):
    """Submit a batch of generation jobs."""
    items: list[BatchItem] = Field(..., min_length=1, max_length=50)
    # Sequential by default. If turbo mode is active, image and video items
    # can run on separate GPUs concurrently.
    priority: Literal["normal", "high"] = "normal"
    # high = jump ahead of single-job queue entries (not ahead of other batches)
    callback_url: str | None = None
    # Optional webhook: POST {batch_id, status, results} on completion
```

### `BatchJob`

```python
@dataclass
class BatchJob:
    id: str                          # batch_xxx (different prefix from job IDs)
    items: list[BatchItem]
    status: BatchStatus              # queued | processing | completed | partial | failed | cancelled
    api_key: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    total: int = 0                   # len(items)
    completed_count: int = 0
    failed_count: int = 0
    current_index: int = 0           # which item is currently running
    results: list[BatchItemResult] = field(default_factory=list)
    turbo: bool = False              # was turbo mode active when batch started?

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
    status: Literal["completed", "failed", "cancelled"]
    result_uri: str | None = None
    media_type: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0
```

---

## 2. `POST /v2/batch` — Submit batch

### Request

```
POST /v2/batch
Authorization: Bearer <key>
Content-Type: application/json

{
  "items": [
    {"type": "text-to-image", "params": {"prompt": "...", "model": "flux2-klein", "width": 1024, "height": 1024}},
    {"type": "text-to-image", "params": {"prompt": "...", "model": "flux2-klein", "width": 1024, "height": 1024}},
    {"type": "text-to-video", "params": {"prompt": "...", "model": "ltx-2-3-fast", "resolution": "1920x1080", "duration": 5}}
  ],
  "priority": "normal"
}
```

### Response — `202 Accepted`

```json
{
  "batch_id": "batch_aB3x...",
  "status": "queued",
  "total": 3,
  "queue_position": 0
}
```

### Error responses

| Status | Condition |
|--------|-----------|
| 400 | Empty items list, invalid item type |
| 422 | Invalid params on any item (validated eagerly before queuing) |
| 429 | Batch queue full (`MAX_BATCH_QUEUE_DEPTH` exceeded). `Retry-After: 30` |
| 503 | System paused. `Retry-After: 300` |

### Validation strategy

Validate all items eagerly on submission (parse each `params` dict through the corresponding Pydantic model: `TextToImageRequest`, `ImageToVideoRequest`, etc.). If any item fails validation, reject the entire batch with the index + error. This prevents queuing a batch that will partially fail on item 47 of 50.

---

## 3. `GET /v2/batch/{batch_id}` — Poll status

### Response

```json
{
  "batch_id": "batch_aB3x...",
  "status": "processing",
  "total": 3,
  "completed_count": 1,
  "failed_count": 0,
  "current_index": 1,
  "turbo": false,
  "results": [
    {
      "index": 0,
      "type": "text-to-image",
      "status": "completed",
      "result_uri": "storage://abc123",
      "media_type": "image/webp",
      "elapsed_s": 12.3
    }
  ],
  "created_at": 1712880000.0,
  "started_at": 1712880005.0,
  "completed_at": null
}
```

Results array grows as items complete — clients can fetch partial results before the batch finishes.

### `DELETE /v2/batch/{batch_id}` — Cancel batch

Cancels remaining items. Already-completed items are preserved. Returns:

```json
{
  "batch_id": "batch_aB3x...",
  "status": "cancelled",
  "completed_count": 1,
  "cancelled_count": 2
}
```

---

## 4. `_enter_turbo_mode()` — Claim dual GPU

This is the sequence of operations to enter turbo mode (claim cuda:1 for taco-backend inference alongside cuda:0):

```python
async def _enter_turbo_mode() -> None:
    """Claim cuda:1 for inference. Caller must hold _inference_lock."""
    global _turbo_active

    if _turbo_active:
        return  # idempotent

    # Step 1: Unload ACE from cuda:1
    # ACE runs as an external sidecar (like JoyAI). Ask it to release GPU memory.
    try:
        await ace_client.unload()  # POST /unload to ACE sidecar
        logger.info("Turbo: ACE unloaded from cuda:1")
    except Exception:
        logger.exception("Turbo: ACE unload failed — aborting turbo entry")
        raise RuntimeError("turbo_entry_failed: could not unload ACE from cuda:1")

    # Step 2: Unload JoyAI from cuda:1
    try:
        await joyai.unload()
        logger.info("Turbo: JoyAI unloaded from cuda:1")
    except JoyAIError:
        logger.warning("Turbo: JoyAI unload failed — continuing (non-critical)")

    # Step 3: Verify cuda:1 is free
    # Quick nvidia-smi check: cuda:1 should be < 1 GB used
    free_mb = _get_gpu_free_mb(1)
    if free_mb < 90_000:  # 90 GB threshold on a 96 GB card
        logger.warning("Turbo: cuda:1 has only %d MB free after unloads", free_mb)

    # Step 4: Update device config for dual-GPU
    # LTX stays on cuda:0, Flux moves to cuda:1 (or vice versa — see topology note)
    config.FLUX_DEVICE = "cuda:1"
    flux.unload()  # Unload from cuda:0
    flux.load(device="cuda:1")  # Reload on cuda:1

    # Step 5: Disable mutual exclusion — both can run concurrently now
    _turbo_active = True
    logger.info("Turbo mode ACTIVE: Flux on cuda:1, LTX on cuda:0")
```

### Why Flux moves (not LTX)

LTX's `SplitModelManager` is deeply wired to `GPU_DEVICES[0]` and reloading it is expensive (25-30s cold). Flux's `enable_model_cpu_offload(device=...)` accepts a device argument, making it trivial to redirect. Moving Flux to cuda:1 costs ~15-60s for the first forward pass (offload hook setup), but subsequent requests are fast.

### Concurrent dispatch in turbo mode

When `_turbo_active`, the dispatcher needs two independent locks:

```python
_flux_lock = asyncio.Lock()   # guards cuda:1 during turbo
_ltx_lock  = asyncio.Lock()   # guards cuda:0 during turbo
```

The existing `_inference_lock` is still used for non-turbo mode and for turbo entry/exit (mode transitions). In turbo mode, the worker loop routes:
- `IMAGE_*` jobs -> acquire `_flux_lock`, run on cuda:1
- `VIDEO_*` jobs -> acquire `_ltx_lock`, run on cuda:0

This allows one image and one video job to run concurrently.

---

## 5. `_exit_turbo_mode()` — Release cuda:1

```python
async def _exit_turbo_mode() -> None:
    """Release cuda:1 back to ACE/JoyAI. Caller must hold _inference_lock."""
    global _turbo_active

    if not _turbo_active:
        return  # idempotent

    # Step 1: Wait for any in-flight turbo jobs to finish
    # (caller should ensure both _flux_lock and _ltx_lock are free)

    # Step 2: Unload Flux from cuda:1
    flux.unload()

    # Step 3: Restore single-GPU config
    config.FLUX_DEVICE = "cuda:0"
    # Do NOT eagerly reload Flux on cuda:0 — lazy load on next image request

    # Step 4: Reload ACE on cuda:1
    try:
        await ace_client.load()
        logger.info("Turbo exit: ACE reloaded on cuda:1")
    except Exception:
        logger.warning("Turbo exit: ACE reload failed — will retry on next request")

    # Step 5: Reload JoyAI on cuda:1
    if config.LOAD_JOYAI:
        try:
            await joyai.load()
            logger.info("Turbo exit: JoyAI reloaded on cuda:1")
        except JoyAIError:
            logger.warning("Turbo exit: JoyAI reload failed — non-critical")

    _turbo_active = False
    logger.info("Turbo mode DEACTIVATED: single-GPU swap restored")
```

---

## 6. `batch_worker()` coroutine

The batch worker runs alongside the existing `worker_loop`. It processes items from a separate `_batch_queue`.

```python
async def batch_worker(
    batch_store: BatchStore,
    queue: asyncio.Queue[str],
    inference_lock: asyncio.Lock,
    dispatch_fn: Callable,
    uploads: UploadStore,
    history: HistoryStore | None = None,
) -> None:
    """Background worker that processes batches from the batch queue."""
    logger.info("Batch worker started")
    while True:
        batch_id = await queue.get()
        batch = batch_store.get(batch_id)
        if batch is None or batch.status == BatchStatus.CANCELLED:
            queue.task_done()
            continue

        batch.status = BatchStatus.PROCESSING
        batch.started_at = time.monotonic()
        logger.info("Processing batch %s (%d items)", batch.id, batch.total)

        for i, item in enumerate(batch.items):
            # Check cancellation between items
            if batch.status == BatchStatus.CANCELLED:
                # Mark remaining as cancelled
                for j in range(i, len(batch.items)):
                    batch.results.append(BatchItemResult(
                        index=j, type=batch.items[j].type,
                        status="cancelled",
                    ))
                break

            batch.current_index = i
            t0 = time.monotonic()

            try:
                # Build a synthetic Job for the dispatch function
                job = _batch_item_to_job(item, batch.api_key)

                if batch.turbo and _turbo_active:
                    # Turbo: pick the right per-device lock
                    lock = _flux_lock if _is_image_type(item.type) else _ltx_lock
                else:
                    lock = inference_lock

                async with lock:
                    if not batch.turbo:
                        # Single-GPU: ensure correct tenant
                        if _is_image_type(item.type):
                            await _ensure_flux_ready()
                        else:
                            await _ensure_ltx_resident()
                    result_bytes = await dispatch_fn(job)

                # Save result
                upload_id, storage_uri = uploads.create()
                uploads.save(upload_id, result_bytes)
                elapsed = time.monotonic() - t0

                batch.results.append(BatchItemResult(
                    index=i, type=item.type, status="completed",
                    result_uri=storage_uri,
                    media_type=_MEDIA_TYPES.get(JobType(item.type), "application/octet-stream"),
                    elapsed_s=round(elapsed, 2),
                ))
                batch.completed_count += 1

                # Fire-and-forget history save
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

        # Webhook callback
        if batch.callback_url:
            asyncio.create_task(_fire_batch_webhook(batch))

        queue.task_done()
```

### Swap optimization

When processing a batch with mixed item types (images + videos), sort items by type before processing to minimize GPU swaps:

```python
# In /v2/batch handler, before queuing:
image_items = [i for i in items if _is_image_type(i.type)]
video_items = [i for i in items if not _is_image_type(i.type)]
batch.items = image_items + video_items  # all images first, then all videos
```

This guarantees at most 1 swap per batch (image tenant -> video tenant) instead of up to N swaps for interleaved items.

---

## 7. `POST /v1/system/turbo` — Toggle turbo mode

### Request

```json
{"enable": true}
```

### Response — `200 OK`

```json
{
  "turbo": true,
  "flux_device": "cuda:1",
  "ltx_device": "cuda:0",
  "ace_status": "unloaded",
  "joyai_status": "unloaded"
}
```

### Error responses

| Status | Condition |
|--------|-----------|
| 409 | Already in the requested state |
| 500 | Turbo entry failed (ACE unload failed, cuda:1 not free) |
| 503 | System paused |

### Interaction with system pause

`/v1/system/pause` automatically exits turbo mode first (releases cuda:1 back to ACE/JoyAI, then evicts everything from cuda:0). `/v1/system/resume` does NOT re-enter turbo — it restores normal single-GPU mode.

---

## 8. `GET /v1/system/gpu` — GPU telemetry

No auth required (like `/health`). Returns nvidia-smi data for the dashboard.

### Implementation

```python
import subprocess, json

async def _query_gpu_info() -> list[dict]:
    """Run nvidia-smi and parse GPU info."""
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
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_used_mb": int(parts[2]),
            "memory_total_mb": int(parts[3]),
            "temperature_c": int(parts[4]) if parts[4] != "[N/A]" else None,
            "utilization_pct": int(parts[5]) if parts[5] != "[N/A]" else None,
            "power_draw_w": float(parts[6]) if parts[6] != "[N/A]" else None,
        })
    return gpus
```

### Response

```json
{
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA RTX PRO 6000",
      "memory_used_mb": 68432,
      "memory_total_mb": 98304,
      "temperature_c": 62,
      "utilization_pct": 95,
      "power_draw_w": 285.3
    },
    {
      "index": 1,
      "name": "NVIDIA RTX PRO 6000",
      "memory_used_mb": 24100,
      "memory_total_mb": 98304,
      "temperature_c": 48,
      "utilization_pct": 0,
      "power_draw_w": 35.2
    }
  ],
  "turbo": false,
  "gpu0_tenant": "ltx",
  "gpu1_tenant": "ace"
}
```

The `gpu0_tenant` / `gpu1_tenant` fields mirror the `_last_gpu_tenant` state variable so the dashboard doesn't have to guess.

### Cache

Cache the nvidia-smi result for 2 seconds (the dashboard polls every 5s). nvidia-smi itself takes ~50ms but spawning a subprocess on every poll is wasteful.

```python
_gpu_cache: dict | None = None
_gpu_cache_time: float = 0.0

@app.get("/v1/system/gpu")
async def system_gpu() -> dict:
    global _gpu_cache, _gpu_cache_time
    now = time.monotonic()
    if _gpu_cache is None or (now - _gpu_cache_time) > 2.0:
        gpus = await _query_gpu_info()
        _gpu_cache = {
            "gpus": gpus,
            "turbo": _turbo_active,
            "gpu0_tenant": _last_gpu_tenant or "idle",
            "gpu1_tenant": "idle" if _turbo_active else "ace",
        }
        _gpu_cache_time = now
    return _gpu_cache
```

---

## 9. Config Constants

Add to `config.py`:

```python
# Turbo mode — dual-GPU inference
TURBO_GPU_DEVICES = ["cuda:0", "cuda:1"]   # devices available in turbo mode
NORMAL_GPU_DEVICES = ["cuda:0"]             # devices in normal single-GPU mode

# Batch queue
MAX_BATCH_QUEUE_DEPTH = 5                   # max concurrent batch submissions
MAX_BATCH_ITEMS = 50                        # max items per batch
BATCH_RESULT_TTL_SECONDS = 1800             # 30 min (batches are larger, keep longer)
```

---

## 10. Edge Cases

### 10.1 Batch cancelled mid-run

- The `batch_worker` checks `batch.status == CANCELLED` between items (not mid-inference).
- If item N is currently running when cancel arrives, it finishes. Items N+1..M are marked cancelled.
- Completed items' `result_uri` files are preserved (history manages lifecycle).
- Cancel request returns immediately (does not wait for current item to finish).

### 10.2 Turbo entry fails

Possible failure modes and handling:

| Failure | Recovery |
|---------|----------|
| ACE `POST /unload` returns 500 | Abort turbo entry. Return 500 to client. Log error. Do NOT partially claim cuda:1. |
| ACE sidecar unreachable (network timeout) | Same as above — treat as failure, do not enter turbo. |
| JoyAI unload fails | Continue — JoyAI is non-critical. Log warning. cuda:1 memory may be partially claimed. |
| cuda:1 still has >6 GB used after unloads | Log warning but proceed. Flux's `enable_model_cpu_offload` will page in/out, so partial occupancy is survivable (just slower). |
| Flux fails to load on cuda:1 | Abort turbo. Restore ACE on cuda:1. Return 500. |

Turbo entry is atomic from the client's perspective: either fully succeeds (200) or fully fails (500) with rollback.

### 10.3 ACE/JoyAI request during turbo mode

When turbo is active, ACE and JoyAI are unloaded. Requests for `model: "joyai-edit"` or music generation should return:

```json
{"error": "turbo_mode_active: ACE/JoyAI unavailable while turbo mode is enabled. Disable turbo first.", "status": 503}
```

with `Retry-After: 10` header.

### 10.4 System pause during active batch

- `_enter_paused` should:
  1. Set `_paused = True`
  2. Cancel all queued batches (not the currently-processing one)
  3. Wait for the current batch item to finish (it holds the inference lock)
  4. Exit turbo mode if active
  5. Evict all models
- The currently-processing batch transitions to `PARTIAL` or `CANCELLED` depending on how many items completed.

### 10.5 Batch with mixed model types

A batch containing both `flux2-dev` and `flux2-klein` items triggers a full Flux pipeline reload (~30-60s) on each model switch. The swap optimizer groups by model within the image items:

```
Sort order: flux2-klein items -> flux2-dev items -> video items
```

This minimizes reloads to at most 1 Klein->Dev transition + 1 Flux->LTX swap.

### 10.6 OOM during batch item

If a CUDA OOM occurs on item N:
1. Item N is marked `failed` with `error_code: "cuda_oom"`.
2. `cleanup_memory()` is called (gc.collect + empty_cache + synchronize).
3. Processing continues with item N+1 — do NOT abort the entire batch.
4. If 3 consecutive items OOM, abort the batch with `status: "failed"` and `error: "repeated_oom"`.

### 10.7 Batch queue interaction with single-job queue

Batches and single jobs share the same GPU. Priority:
- Single jobs go into `_job_queue` (existing).
- Batches go into `_batch_queue` (new).
- The `batch_worker` runs alongside `worker_loop`. Both compete for `_inference_lock`.
- A batch holds the inference lock for one item at a time, releasing between items. This lets single jobs interleave between batch items.
- `priority: "high"` batches: items are submitted at the front of the internal item queue, but this is within the batch — it doesn't preempt single jobs.

### 10.8 Webhook delivery

The `callback_url` webhook is fire-and-forget with 3 retries (1s, 5s, 15s backoff). Payload:

```json
{
  "batch_id": "batch_aB3x...",
  "status": "completed",
  "total": 3,
  "completed_count": 3,
  "failed_count": 0,
  "results": [...]
}
```

If all 3 retries fail, log a warning and move on. No dead-letter queue.

---

## Implementation Order

1. **`/v1/system/gpu` endpoint** — standalone, no dependencies. Enables dashboard GPU cards immediately.
2. **Batch models + store** — `BatchJob`, `BatchStore`, `BatchItemResult` in `job_queue.py`.
3. **`/v2/batch` + `/v2/batch/{id}` endpoints** — in `server.py`. Batch worker coroutine.
4. **Turbo mode** — `_enter_turbo_mode`, `_exit_turbo_mode`, `/v1/system/turbo` endpoint. Dual-lock dispatch.
5. **Dashboard turbo buttons** — enable the disabled buttons in `dashboard.html`, wire to `/v1/system/turbo`.
6. **Batch swap optimizer** — sort items by type+model before processing.
7. **Webhook delivery** — optional, can be deferred.

Each step is independently shippable. Steps 1-3 can be done without turbo mode (batches run on single GPU with normal swap).
