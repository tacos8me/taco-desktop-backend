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
    items: list[BatchItem]           # list[Any] in code to avoid circular import
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
```

---

## 2. `POST /v2/batch` — Submit batch

### Supported item types

| `type` | Pydantic validator | LoRA registry | Notes |
|--------|--------------------|---------------|-------|
| `text-to-image` | `TextToImageRequest` | `flux_lora_registry` | `width`/`height` snapped to multiples of 16 |
| `image-to-image` | `ImageToImageRequest` | `flux_lora_registry` | `image_uri` resolved to filesystem path |
| `image-edit` | `ImageEditRequest` | `flux_lora_registry` | `image_uris` resolved to filesystem paths |
| `text-to-video` | `TextToVideoRequest` | `lora_registry` | `resolution` + `duration` converted to `width`/`height`/`num_frames` |
| `image-to-video` | `ImageToVideoRequest` | `lora_registry` | `resolution` + `duration` converted; `image_uri` resolved |

### Param preprocessing (`_batch_item_to_job`)

The batch endpoint applies the same preprocessing as the individual v2 handlers before creating a `Job` for dispatch:

1. **LoRA resolution**: `lora: {id, strength}` is resolved to `lora_path` (filesystem path) + `lora_strength` (float) via the appropriate registry (Flux LoRA for image types, LTX LoRA for video types). Unknown LoRA IDs raise `ValueError`, failing the item.
2. **Video params**: `resolution` (e.g. `"1920x1080"`) is converted to `width`/`height`. `duration` (seconds) is converted to `num_frames` via `_duration_to_frames(duration, fps)`. `camera_motion` is appended to the prompt as `[camera_motion]`. `seed` is auto-generated if missing. `generate_audio` defaults to `false`.
3. **Image params**: `width`/`height` are snapped to multiples of 16. `seed` is auto-generated if missing.
4. **URI resolution**: `image_uri` (image-to-image, image-to-video) and `image_uris` (image-edit) are resolved from `storage://` URIs to filesystem paths.

### Request

```
POST /v2/batch
Authorization: Bearer <key>
Content-Type: application/json

{
  "items": [
    {"type": "text-to-image", "params": {"prompt": "...", "model": "flux2-klein", "width": 1024, "height": 1024}},
    {"type": "text-to-image", "params": {"prompt": "...", "model": "flux2-klein", "width": 1024, "height": 1024}},
    {"type": "text-to-video", "params": {"prompt": "...", "model": "ltx-2-3-fast", "resolution": "1920x1080", "duration": 5, "fps": 24}}
  ],
  "priority": "normal"
}
```

### Example: batch with LoRA (video)

```json
{
  "items": [
    {
      "type": "text-to-video",
      "params": {
        "prompt": "A person walking through a forest",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 5,
        "fps": 24,
        "lora": {"id": "my-style-lora", "strength": 0.8}
      }
    },
    {
      "type": "text-to-video",
      "params": {
        "prompt": "A bird flying over mountains",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 3,
        "fps": 24,
        "lora": {"id": "my-style-lora", "strength": 0.8}
      }
    }
  ]
}
```

The `lora.id` must match an ID from `GET /v1/loras` (for video types) or `GET /v1/flux-loras` (for image types). The server resolves IDs to filesystem paths internally via `_batch_item_to_job`. LoRA strength changes between items are free for Flux (adapter mode), but cause a full transformer reload for LTX (fusion mode).

### Example: batch with LoRA (image)

```json
{
  "items": [
    {
      "type": "text-to-image",
      "params": {
        "prompt": "A portrait in oil painting style",
        "model": "flux2-dev",
        "width": 1024,
        "height": 1024,
        "turbo": true,
        "lora": {"id": "oil-painting", "strength": 0.7}
      }
    },
    {
      "type": "image-edit",
      "params": {
        "prompt": "Make the background a sunset",
        "model": "flux2-klein",
        "image_uris": ["storage://abc123"],
        "width": 1024,
        "height": 1024
      }
    }
  ]
}
```

### Example: mixed batch (images + videos)

```json
{
  "items": [
    {
      "type": "text-to-image",
      "params": {"prompt": "A cat on a roof", "model": "flux2-klein", "width": 1024, "height": 1024}
    },
    {
      "type": "image-to-video",
      "params": {
        "prompt": "The cat jumps down from the roof",
        "image_uri": "storage://def456",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 4,
        "fps": 24
      }
    }
  ]
}
```

Note: the server sorts items by type before processing (all images first, then all videos) to minimize GPU swaps. Within image items, Klein items are sorted before Dev to minimize Flux model reloads.

### Per-type required params

**text-to-image**: `prompt`, `model` (`flux2-dev` | `flux2-klein`), `width`, `height`. Optional: `num_inference_steps`, `guidance_scale`, `seed`, `turbo`, `lora`.

**image-to-image**: `prompt`, `image_uri`, `model`, `width`, `height`. Optional: `num_inference_steps`, `guidance_scale`, `seed`, `turbo`, `lora`.

**image-edit**: `prompt`, `image_uris` (list of `storage://` URIs), `model`. Optional: `width`, `height`, `num_inference_steps`, `guidance_scale`, `seed`, `lora`.

**text-to-video**: `prompt`, `model` (`ltx-2-3-fast` | `ltx-2-3-pro` | `ltx-2-3-hq`), `resolution`, `duration`, `fps`. Optional: `generate_audio`, `camera_motion`, `lora`, `enhance_prompt`.

**image-to-video**: `prompt`, `model`, `resolution`, `duration`, `fps`. Either `image_uri` or `keyframes` required. Optional: `image_strength`, `generate_audio`, `lora`, `enhance_prompt`.

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
| 400 | Invalid item type |
| 422 | Invalid params on any item (validated eagerly before queuing) |
| 429 | Batch queue full (`MAX_BATCH_QUEUE_DEPTH` exceeded). `Retry-After: 30` |
| 503 | System paused. `Retry-After: 300` |

### Validation strategy

Validate all items eagerly on submission (parse each `params` dict through the corresponding Pydantic model: `TextToImageRequest`, `ImageToVideoRequest`, etc.). If any item fails validation, reject the entire batch with the index + error. This prevents queuing a batch that will partially fail on item 47 of 50.

Note: validation catches schema errors (missing required fields, wrong types, out-of-range values) but does NOT validate LoRA IDs or `storage://` URIs at submission time. These are resolved at dispatch time by `_batch_item_to_job` and will fail the individual item (not the whole batch) if invalid.

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

Turbo mode uses an **LTX sidecar** (a separate process managed via systemctl) to run video jobs on cuda:1, while Flux/LTX image jobs continue on cuda:0 in-process. This avoids moving Flux between GPUs and the complexity of dual in-process managers.

```python
async def _enter_turbo_mode() -> None:
    """Enable turbo: tell the LTX sidecar to load its pipeline on cuda:1.
    Caller must hold _inference_lock."""
    global _turbo_active, _turbo_worker_task, _last_gpu_tenant

    if _turbo_active:
        return  # idempotent

    # Step 1: Stop ACE on cuda:1 (via systemctl, not HTTP)
    if config.LOAD_ACE:
        await _ace_systemctl("stop")

    # Step 2: Unload JoyAI from cuda:1
    if config.LOAD_JOYAI:
        await joyai.unload()  # non-critical, logged warning on failure

    # Step 3: Evict Flux from cuda:0 (LTX needs full GPU)
    flux.unload()

    # Step 4: Tell the LTX sidecar to load its pipeline on cuda:1
    await ltx_sidecar.load()

    # Step 5: Start a second worker_loop dispatching to the sidecar
    _turbo_active = True
    _turbo_worker_task = asyncio.create_task(
        worker_loop(job_store, _job_queue, _inference_lock, _dispatch_job_turbo, ...),
        name="turbo-worker",
    )
```

### Why the sidecar approach (not in-process)

LTX's `SplitModelManager` is deeply wired to `GPU_DEVICES[0]`. Running a second in-process LTX manager on cuda:1 would require duplicating the entire encoder hub + transformer state. Instead, the LTX sidecar is a separate lightweight process that runs the same LTX pipeline independently on cuda:1. Communication is via HTTP (`ltx_sidecar_client.py`). Flux is evicted during turbo because the sidecar + in-process LTX both need GPU memory; Flux lazy-reloads on the first post-turbo image request.

### Concurrent dispatch in turbo mode

In turbo mode, two workers pull from the same `_job_queue`:
- The primary `worker_loop` dispatches via `_dispatch_job` (in-process, cuda:0)
- The turbo `worker_loop` dispatches via `_dispatch_job_turbo` (HTTP to sidecar, cuda:1)

For batch processing specifically, the `batch_worker` dispatches 2 items concurrently via `asyncio.gather`: one to `_dispatch_job` (cuda:0) and one to `_dispatch_job_turbo` (cuda:1).

---

## 5. `_exit_turbo_mode()` — Release cuda:1

```python
async def _exit_turbo_mode() -> None:
    """Unload the LTX sidecar pipeline, restore cuda:1 for ACE+JoyAI.
    Caller must hold _inference_lock."""
    global _turbo_active, _turbo_worker_task

    if not _turbo_active:
        return  # idempotent

    # Step 1: Cancel the second worker loop
    if _turbo_worker_task is not None:
        _turbo_worker_task.cancel()
        await _turbo_worker_task  # swallow CancelledError

    # Step 2: Tell the sidecar to free cuda:1 GPU memory
    await ltx_sidecar.unload()

    # Step 3: Restart ACE on cuda:1 (via systemctl)
    if config.LOAD_ACE:
        await _ace_systemctl("start")

    # Step 4: Reload JoyAI on cuda:1
    if config.LOAD_JOYAI:
        await joyai.load()

    _turbo_active = False
```

---

## 6. `batch_worker()` coroutine

The batch worker runs alongside the existing `worker_loop`. It processes items from a separate `_batch_queue`. Individual items are processed by `_process_batch_item()`, which handles job creation, dispatch, result storage, and error handling.

```python
async def batch_worker() -> None:
    """Background worker that processes batches from the batch queue.

    In turbo mode: dispatches 2 items concurrently via asyncio.gather
    (one to _dispatch_job on cuda:0, one to _dispatch_job_turbo on cuda:1).

    In normal mode: dispatches 1 item at a time, holding _inference_lock,
    with swap logic (_ensure_ltx_resident / _ensure_flux_ready).
    """
    while True:
        batch_id = await _batch_queue.get()
        batch = batch_store.get(batch_id)
        if batch is None or batch.status == BatchStatus.CANCELLED:
            _batch_queue.task_done()
            continue

        batch.status = BatchStatus.PROCESSING
        batch.started_at = time.monotonic()

        # Auto-turbo: if cuda:1 idle long enough, engage dual-GPU
        idle_min = _cuda1_idle_seconds() / 60
        if (not _turbo_active
            and idle_min >= config.AUTO_TURBO_IDLE_MINUTES
            and len(batch.items) >= 2):
            try:
                async with _inference_lock:
                    await _enter_turbo_mode()
            except Exception:
                pass  # proceed in single-GPU mode

        items = list(enumerate(batch.items))
        idx = 0
        while idx < len(items):
            if batch.status == BatchStatus.CANCELLED:
                for j in range(idx, len(items)):
                    batch.results.append(BatchItemResult(
                        index=items[j][0], type=items[j][1].type,
                        status="cancelled"))
                break

            if _turbo_active:
                # Turbo: 2 items concurrently — cuda:0 + cuda:1 (sidecar)
                chunk = items[idx:idx+2]
                dispatchers = [_dispatch_job, _dispatch_job_turbo]
                tasks = [
                    _process_batch_item(batch, i, item, dispatch_fn=dispatchers[ci])
                    for ci, (i, item) in enumerate(chunk)
                ]
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

        if batch.callback_url:
            asyncio.create_task(_fire_batch_webhook(batch))
        _batch_queue.task_done()
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
| ACE `systemctl stop` fails | Abort turbo entry. Return 500 to client. Log error. Do NOT partially claim cuda:1. |
| JoyAI unload fails | Continue — JoyAI is non-critical. Log warning. |
| LTX sidecar `load()` fails | Abort turbo. Restart ACE on cuda:1. Return 500. |
| LTX sidecar unreachable | Same as above — treat as failure, do not enter turbo. |

Turbo entry is atomic from the client's perspective: either fully succeeds (200) or fully fails (500) with rollback. For auto-turbo (batch_worker), a failed entry logs a warning and the batch proceeds in single-GPU mode.

### 10.3 ACE/JoyAI request during turbo mode

When turbo is active, ACE and JoyAI are unloaded. JoyAI-edit and music requests call `_auto_exit_turbo_if_active(reason)`, which gracefully exits turbo mode (~15s) then proceeds with the request normally. This resets `_last_cuda1_activity`, so the auto-turbo idle timer restarts from zero.

The auto-exit is transparent to the client -- they don't see a 503 or need to retry.

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

---

## LTX Model Parameters Reference

### Transformer states

The `DenoiserWorker.ensure_transformer(state, user_lora)` method in `split_model_manager.py` manages six named transformer states. Each state specifies a base checkpoint and zero or more preset LoRAs fused at load time.

| State | Checkpoint | LoRA | Strength | Used By |
|-------|-----------|------|----------|---------|
| `dev` | `ltx-2.3-22b-dev.safetensors` | none | -- | pro stage 1 (t2v, i2v, a2v), retake |
| `distilled` | `ltx-2.3-22b-distilled.safetensors` | none | -- | fast i2v stage 1, fast a2v stage 1, fast t2v pass 2 |
| `dev_lora` | `ltx-2.3-22b-dev.safetensors` | `distilled-lora-384` | 1.0 | pro/i2v/a2v stage 2 |
| `dev_lora_025` | `ltx-2.3-22b-dev.safetensors` | `distilled-lora-384` | 0.25 | hq stage 1 |
| `dev_lora_050` | `ltx-2.3-22b-dev.safetensors` | `distilled-lora-384` | 0.50 | hq stage 2 |
| `dev_lora_020` | `ltx-2.3-22b-dev.safetensors` | `distilled-lora-384` | 0.20 | fast t2v pass 1 |

All states use BF16 precision. LoRAs are fused via `LoraPathStrengthAndSDOps` with `LTXV_LORA_COMFY_RENAMING_MAP` at load time (permanent fusion -- no unfuse).

### Model pipeline details

**`ltx-2-3-fast` (text-to-video)** -- 2-pass split-schedule, 8 total steps, no CFG:

1. Pass 1: `dev_lora_020` (dev + distilled LoRA @ 0.2), `simple_denoising_func`, first 4 steps of `DISTILLED_SIGMA_VALUES`
2. Evict transformer, swap to `distilled`
3. Pass 2: `distilled`, `simple_denoising_func`, last 4 steps of `DISTILLED_SIGMA_VALUES`
4. Upsample 2x via spatial upscaler
5. Stage 2: `distilled` (no swap -- already loaded), 5 distilled steps (`_STAGE_2_SIGMAS`), `simple_denoising_func`

**`ltx-2-3-fast` (image-to-video / audio-to-video)** -- single-pass, 8 steps, no CFG:

1. Stage 1: `distilled`, `simple_denoising_func`, 8 steps (`DISTILLED_SIGMA_VALUES`)
2. Upsample 2x via spatial upscaler
3. Stage 2: `distilled` (no swap), 5 distilled steps (`_STAGE_2_SIGMAS`), `simple_denoising_func`

Note: i2v/a2v fast does NOT use the 2-pass split with `dev_lora_020`. Only t2v fast uses the split schedule.

**`ltx-2-3-pro`** -- 2-stage, 30 euler steps + CFG + STG:

1. Stage 1: `dev` (30 euler steps, `LTX2Scheduler` token-adapted sigmas, `multi_modal_guider_factory_denoising_func` with CFG + STG)
2. Upsample 2x via spatial upscaler
3. Stage 2: `dev_lora` (dev + distilled LoRA @ 1.0), 5 distilled steps (`_STAGE_2_SIGMAS`), `simple_denoising_func`

**`ltx-2-3-hq`** -- 2-stage, res2s sampler, 15 steps + CFG + STG:

1. Stage 1: `dev_lora_025` (dev + distilled LoRA @ 0.25), `Res2sDiffusionStep`, 15 res2s steps via `LTX2Scheduler` (NFE = 2 * 15 + 1 = 31), `multi_modal_guider_denoising_func` with CFG + STG
2. Upsample 2x via spatial upscaler
3. Stage 2: `dev_lora_050` (dev + distilled LoRA @ 0.50), 5 distilled steps (`_STAGE_2_SIGMAS`), res2s loop, `simple_denoising_func`

**Retake** -- single-stage, 30 euler steps, dev transformer:

1. Evict transformer to make room for VAE encode of input video
2. `dev` transformer, 30 euler steps, `multi_modal_guider_denoising_func` with CFG + STG, temporal region masking

### Sigma schedules

```
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]  # 8 steps
_STAGE_2_SIGMAS        = [0.909375, 0.727, 0.546, 0.364, 0.182, 0.0]                               # 5 steps
```

Stage 2 uses 5 steps instead of the upstream default of 3 (`STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]`). More steps resolve fast motion better during super-resolution.

### User LoRA integration

User-supplied LoRAs are appended to whatever preset LoRA tuple the state already defines. In `ensure_transformer`:

```python
if user_lora:
    path, strength = user_lora
    loras = loras + (LoraPathStrengthAndSDOps(
        path=path, strength=strength,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    ),)
```

The combined `loras` tuple is passed to `ModelLedger(... loras=loras)`, which fuses all adapters into the transformer at load time. Fusion is permanent -- there is no `unfuse` operation. Changing the LoRA file or strength requires a full transformer reload.

Cache key is `(transformer_state, user_lora_tuple)` where `user_lora_tuple` is `(path, strength)` or `None`. A cache hit (same state + same user LoRA) skips the reload entirely.

Request field: `lora: {id, strength}` on `TextToVideoRequest`, `ImageToVideoRequest`, `AudioToVideoRequest`, `RetakeRequest`. The `LoRAInput` Pydantic model defines `id: str` (resolved from `/v1/loras` registry) and `strength: float` (default 1.0, range 0.0--2.0).

### Checkpoint file sizes

All files in `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/`:

| File | Size |
|------|------|
| `ltx-2.3-22b-dev.safetensors` | 43 GB |
| `ltx-2.3-22b-distilled.safetensors` | 43 GB |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.1 GB |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950 MB |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | 950 MB |
| `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` | 1.1 GB |
| `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | 250 MB |

Config references (`config.py`):
- `DEV_CHECKPOINT` = `ltx-2.3-22b-dev.safetensors`
- `DISTILLED_CHECKPOINT` = `ltx-2.3-22b-distilled.safetensors`
- `DISTILLED_LORA` = `ltx-2.3-22b-distilled-lora-384.safetensors`
- `SPATIAL_UPSAMPLER` = `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`

### Auto-turbo

The `AUTO_TURBO_IDLE_MINUTES` config (default: 30, env-overridable) controls automatic turbo mode engagement for batch processing.

**Idle timer**: `_last_cuda1_activity` tracks the monotonic timestamp of the last JoyAI edit or ACE music completion. `_cuda1_idle_seconds()` returns the elapsed time since that event. Initialized to `time.monotonic()` at import so the 30-minute timer does not fire immediately at startup.

**Auto-engage**: When `batch_worker` picks up a batch, it checks three conditions:
1. Turbo is not already active
2. cuda:1 idle time >= `AUTO_TURBO_IDLE_MINUTES`
3. Batch has >= 2 items

If all pass, it acquires `_inference_lock` and calls `_enter_turbo_mode()`. If turbo entry fails (ACE stop fails, etc.), the batch proceeds in single-GPU mode with a warning log.

**Auto-exit**: When a JoyAI-edit or ACE music request arrives while turbo is active, `_auto_exit_turbo_if_active(reason)` gracefully exits turbo mode (~15s), then the request proceeds normally. This resets `_last_cuda1_activity`, so the idle timer restarts from zero.

**Flow**: idle 30 min -> batch arrives -> auto-engage turbo -> batch runs on dual GPU -> JoyAI/ACE request arrives -> auto-exit turbo -> cuda:1 returns to ACE/JoyAI -> idle timer restarts.
