# Async Queue API Specification

> Design doc for taco-backend async job endpoints.
> Date: 2026-03-07

## Overview

Replace synchronous generation endpoints (which block HTTP connections for 16-63+ seconds) with an async submit-poll-fetch pattern. The client submits a job, receives a job ID immediately, and polls or subscribes via SSE for completion.

---

## 1. Endpoint Summary

### Async (new submit/poll/fetch pattern)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v2/text-to-video` | Submit t2v job |
| POST | `/v2/image-to-video` | Submit i2v job |
| POST | `/v2/audio-to-video` | Submit a2v job |
| POST | `/v2/retake` | Submit retake job |
| POST | `/v2/text-to-image` | Submit t2i job |
| POST | `/v2/image-to-image` | Submit i2i job |
| GET | `/v2/jobs/{job_id}` | Poll job status |
| GET | `/v2/jobs/{job_id}/result` | Fetch completed result (binary) |
| GET | `/v2/jobs/{job_id}/stream` | SSE stream for job progress |
| DELETE | `/v2/jobs/{job_id}` | Cancel a queued/running job |
| GET | `/v2/jobs` | List recent jobs (optional) |

### Stay synchronous (no queue needed)

| Method | Path | Reason |
|--------|------|--------|
| GET | `/health` | Instant, no GPU |
| POST | `/v1/chat/completions` | Proxied to external server, fast (~1s) |
| POST | `/v1/upload` | Metadata only, instant |
| PUT | `/uploads/put/{id}` | File upload, no GPU |

### Backward compatibility (kept, deprecated)

| Method | Path | Behavior |
|--------|------|----------|
| POST | `/v1/text-to-video` | Sync, returns binary (deprecated) |
| POST | `/v1/image-to-video` | Sync, returns binary (deprecated) |
| POST | `/v1/audio-to-video` | Sync, returns binary (deprecated) |
| POST | `/v1/retake` | Sync, returns binary (deprecated) |
| POST | `/v1/text-to-image` | Sync, returns binary (deprecated) |
| POST | `/v1/image-to-image` | Sync, returns binary (deprecated) |

The v1 sync endpoints remain available but are marked deprecated via response header `Deprecation: true` and `Link: </v2/...>; rel="successor-version"`. They will be removed in a future release.

---

## 2. Request Schemas

The v2 submit endpoints reuse the **exact same Pydantic request models** as v1:

- `TextToVideoRequest` -- POST `/v2/text-to-video`
- `ImageToVideoRequest` -- POST `/v2/image-to-video`
- `AudioToVideoRequest` -- POST `/v2/audio-to-video`
- `RetakeRequest` -- POST `/v2/retake`
- `TextToImageRequest` -- POST `/v2/text-to-image`
- `ImageToImageRequest` -- POST `/v2/image-to-image`

No schema changes. The only difference is the response format.

Optional addition to all request models (can be added later):

```python
class AsyncOptions(BaseModel):
    """Optional fields appended to any generation request."""
    priority: int = Field(default=0, ge=0, le=10)  # Higher = higher priority
    webhook_url: str | None = None                   # POST result notification
```

Priority and webhook are deferred -- not required for initial implementation.

---

## 3. Response Schemas

### 3a. Job Submission Response (202 Accepted)

Returned by all POST `/v2/*` endpoints.

```python
class JobSubmissionResponse(BaseModel):
    job_id: str          # UUID hex, e.g. "a1b2c3d4..."
    status: JobStatus    # "queued"
    created_at: str      # ISO 8601 timestamp
    poll_url: str        # "/v2/jobs/{job_id}"
    stream_url: str      # "/v2/jobs/{job_id}/stream"
```

Example:
```json
{
  "job_id": "f47ac10b58cc4372a5670e02b2c3d479",
  "status": "queued",
  "created_at": "2026-03-07T14:30:00Z",
  "poll_url": "/v2/jobs/f47ac10b58cc4372a5670e02b2c3d479",
  "stream_url": "/v2/jobs/f47ac10b58cc4372a5670e02b2c3d479/stream"
}
```

### 3b. Job Status Response (GET `/v2/jobs/{job_id}`)

```python
class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    type: str               # "text-to-video", "image-to-image", etc.
    created_at: str
    started_at: str | None   # When processing began
    completed_at: str | None # When finished (success or failure)
    progress: float | None   # 0.0-1.0, null if not yet started
    queue_position: int | None  # Position in queue, null if processing/done
    error: JobError | None   # Populated only when status == "failed"
    result_url: str | None   # "/v2/jobs/{job_id}/result", only when completed
    result_media_type: str | None  # "video/mp4" or "image/png", only when completed
```

### 3c. Job Result (GET `/v2/jobs/{job_id}/result`)

- Returns raw binary with appropriate `Content-Type` (`video/mp4` or `image/png`)
- Returns `404` if job not found
- Returns `409 Conflict` with error body if job is not yet completed
- Result is available for a retention period (e.g. 1 hour) after completion, then cleaned up

### 3d. Job List Response (GET `/v2/jobs`)

```python
class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int
```

Query params: `?status=queued&limit=20&offset=0`

---

## 4. Job Status Enum

```python
from enum import StrEnum

class JobStatus(StrEnum):
    QUEUED     = "queued"      # Accepted, waiting in queue
    PROCESSING = "processing"  # GPU inference in progress
    COMPLETED  = "completed"   # Result ready for retrieval
    FAILED     = "failed"      # Generation failed
    CANCELLED  = "cancelled"   # Cancelled by client
```

State transitions:
```
queued -> processing -> completed
                     -> failed
queued -> cancelled
processing -> cancelled (best-effort)
```

---

## 5. Progress Reporting

### Via polling (GET `/v2/jobs/{job_id}`)

The `progress` field is a float from 0.0 to 1.0:

- `null` -- job is queued, not started
- `0.0` -- processing just started (pipeline setup)
- `0.01-0.99` -- denoising step progress (step N / total steps)
- `1.0` -- generation complete, encoding result

For multi-stage pipelines (e.g. ltx-2-3-pro with stage 1 + stage 2):

- Stage 1 maps to 0.0-0.7
- Stage 2 maps to 0.7-0.95
- Encoding/muxing maps to 0.95-1.0

### Via SSE (GET `/v2/jobs/{job_id}/stream`)

Server-Sent Events stream with the following event types:

```
event: status
data: {"status": "queued", "queue_position": 2}

event: status
data: {"status": "processing"}

event: progress
data: {"progress": 0.15, "stage": "denoising", "step": 3, "total_steps": 20}

event: progress
data: {"progress": 0.45, "stage": "denoising", "step": 9, "total_steps": 20}

event: progress
data: {"progress": 0.95, "stage": "encoding"}

event: completed
data: {"status": "completed", "result_url": "/v2/jobs/abc123/result", "result_media_type": "video/mp4"}

event: error
data: {"status": "failed", "error": {"code": "generation_failed", "message": "CUDA out of memory"}}
```

SSE connection behavior:
- Client connects at any time (before, during, or after processing)
- If job is already completed/failed, server sends the final event and closes
- Server sends a heartbeat comment (`: heartbeat`) every 15 seconds to keep the connection alive
- Client can reconnect using `Last-Event-ID` header (events are numbered)

---

## 6. Error Format

### 6a. Job Failure (in JobStatusResponse and SSE)

```python
class JobError(BaseModel):
    code: str       # Machine-readable error code
    message: str    # Human-readable description
```

Error codes:

| Code | Meaning |
|------|---------|
| `generation_failed` | Pipeline threw an exception during inference |
| `cuda_oom` | CUDA out of memory |
| `invalid_input` | Input validation failed after submission (e.g. corrupt image) |
| `upload_not_found` | Referenced storage:// URI no longer exists |
| `timeout` | Job exceeded maximum processing time |
| `cancelled` | Job was cancelled by client |
| `internal_error` | Unexpected server error |

### 6b. HTTP Error Responses (non-job errors)

Standard error format (unchanged from v1):

```json
{
  "error": "description",
  "message": "description",
  "detail": "description"
}
```

Relevant HTTP status codes for v2 endpoints:

| Status | When |
|--------|------|
| 202 | Job accepted and queued |
| 200 | Job status / result retrieved |
| 400 | Malformed request body |
| 401 | Missing/invalid API key |
| 404 | Job not found |
| 409 | Result requested but job not completed |
| 422 | Validation error (Pydantic) |
| 429 | Queue is full (max pending jobs exceeded) |
| 500 | Server error |
| 503 | Pipeline not loaded / server not ready |

---

## 7. Job Cancellation

### DELETE `/v2/jobs/{job_id}`

- If **queued**: removes from queue, sets status to `cancelled`. Returns 200.
- If **processing**: sets a cancellation flag. The pipeline checks this flag between denoising steps and aborts if set. Best-effort -- the current step completes before checking. Returns 202 (accepted, cancellation pending).
- If **completed/failed/cancelled**: returns 409 (cannot cancel a finished job).
- If **not found**: returns 404.

Response body:
```json
{
  "job_id": "abc123",
  "status": "cancelled",
  "message": "Job cancelled"
}
```

---

## 8. Queue Management

### Queue full (429)

When the queue exceeds a configurable maximum (e.g. 10 pending jobs), new submissions return:

```json
HTTP/1.1 429 Too Many Requests
Retry-After: 30

{
  "error": "queue_full",
  "message": "Job queue is full. Try again later.",
  "detail": "Maximum 10 pending jobs allowed."
}
```

### Job retention

- **Completed results**: retained for 1 hour, then cleaned up
- **Job metadata**: retained for 24 hours
- **Failed jobs**: metadata retained for 24 hours, no result to clean up

---

## 9. Backward Compatibility Strategy

### Phase 1: Add v2, keep v1 (this release)

- All v2 async endpoints added alongside existing v1 sync endpoints
- v1 endpoints unchanged in behavior -- same request/response format
- v1 responses gain deprecation headers:
  ```
  Deprecation: true
  Link: </v2/text-to-video>; rel="successor-version"
  ```
- `/health` response adds queue status:
  ```json
  {
    "status": "ok",
    "ltx": "ready",
    "flux": "ready",
    "chat": "ready",
    "queue": {
      "pending": 3,
      "processing": 1
    }
  }
  ```

### Phase 2: v1 sync endpoints run through queue (future)

- v1 endpoints internally submit a job to the same queue, then block until completion before returning the binary result
- This unifies the code path -- all generation goes through the queue
- v1 callers see no behavior change (same request/response), but benefit from queue ordering

### Phase 3: Remove v1 (future)

- After taco-desktop has migrated to v2, remove v1 sync endpoints
- Non-breaking for any client already on v2

### Client migration path

The frontend needs to change from:
```
POST /v1/text-to-video → wait 60s → receive video/mp4 body
```
To:
```
POST /v2/text-to-video → receive 202 with job_id
GET /v2/jobs/{id}/stream → SSE progress events
GET /v2/jobs/{id}/result → receive video/mp4 body
```

Or the simpler polling approach:
```
POST /v2/text-to-video → receive 202 with job_id
loop: GET /v2/jobs/{id} → check status, read progress
GET /v2/jobs/{id}/result → receive video/mp4 body
```

---

## 10. Auth

All v2 endpoints require the same `Authorization: Bearer <key>` header as v1. The existing auth middleware applies to all paths except `/health`.

Job isolation: a job is accessible to any authenticated client (single-user system). No per-user job scoping needed.

---

## 11. CORS

Same CORS policy as v1 (localhost + LAN IPs). The v2 endpoints are added under the same middleware. SSE streams need `text/event-stream` in allowed response types -- no CORS change needed since it's a GET with standard headers.

---

## 12. OpenAPI / Typing Summary

New Pydantic models to add:

```python
class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobError(BaseModel):
    code: str
    message: str

class JobSubmissionResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    poll_url: str
    stream_url: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    type: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    progress: float | None = None
    queue_position: int | None = None
    error: JobError | None = None
    result_url: str | None = None
    result_media_type: str | None = None

class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int
```

Existing models reused without changes:
- `TextToVideoRequest`, `ImageToVideoRequest`, `AudioToVideoRequest`
- `RetakeRequest`, `TextToImageRequest`, `ImageToImageRequest`
- `ModelName`, `Resolution`, `RetakeMode`, `ImageModelName`
