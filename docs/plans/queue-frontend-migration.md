# Frontend Migration Guide: Async Job Queue

> **Audience**: taco-desktop frontend team
> **Date**: 2026-03-07
> **Status**: Planning

---

## 1. Why This Change

The taco-backend is moving all generation endpoints (video and image) from synchronous to asynchronous. The reason: **Cloudflare Tunnel enforces a 100-second HTTP response timeout**. Some generation requests exceed this -- pro-model text-to-video takes ~63s, and higher resolutions or audio+video can push past 100s. When that happens, Cloudflare kills the connection and the user gets nothing.

The fix is a **submit-poll-fetch** pattern:

1. **Submit** a generation request -- get a job ID back immediately (< 1s)
2. **Poll** for status and progress -- show a progress bar to the user
3. **Fetch** the result once complete -- download the video/image binary

No individual HTTP request blocks for more than a few seconds, so Cloudflare never times out.

---

## 2. Endpoint Changes

### Generation endpoints: v1 (old) vs v2 (new)

All generation endpoints move from `/v1/` to `/v2/`. The **request body is identical** -- same fields, same validation. Only the response changes.

| Old (synchronous) | New (async) | Request body |
|---|---|---|
| `POST /v1/text-to-video` | `POST /v2/text-to-video` | `TextToVideoRequest` (unchanged) |
| `POST /v1/image-to-video` | `POST /v2/image-to-video` | `ImageToVideoRequest` (unchanged) |
| `POST /v1/audio-to-video` | `POST /v2/audio-to-video` | `AudioToVideoRequest` (unchanged) |
| `POST /v1/retake` | `POST /v2/retake` | `RetakeRequest` (unchanged) |
| `POST /v1/text-to-image` | `POST /v2/text-to-image` | `TextToImageRequest` (unchanged) |
| `POST /v1/image-to-image` | `POST /v2/image-to-image` | `ImageToImageRequest` (unchanged) |

**Old behavior** (v1): POST returns raw binary (`video/mp4` or `image/png`) after 16-63+ seconds.

**New behavior** (v2): POST returns `202 Accepted` with a JSON body containing the job ID. You then poll or stream for progress, and fetch the binary when done.

### New endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v2/jobs/{job_id}` | Poll job status and progress |
| `GET` | `/v2/jobs/{job_id}/result` | Download the completed result (binary) |
| `GET` | `/v2/jobs/{job_id}/stream` | SSE stream for real-time progress |
| `DELETE` | `/v2/jobs/{job_id}` | Cancel a queued or running job |
| `GET` | `/v2/jobs` | List recent jobs (optional) |

### Endpoints that stay the same (no changes needed)

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Still instant. Now includes `queue` field with depth info. |
| `POST` | `/v1/chat/completions` | Still synchronous. Fast (~1s), no GPU contention. |
| `POST` | `/v1/upload` | Still synchronous. Metadata only, instant. |
| `PUT` | `/uploads/put/{id}` | Still synchronous. File upload, no GPU. |

### Backward compatibility

The old v1 sync endpoints will remain available during migration (marked deprecated with response headers). They will be removed in a future release. Do not write new code against v1 generation endpoints.

---

## 3. New Workflow: Submit, Poll, Fetch

### Sequence diagram

```
Frontend                          Backend
   |                                 |
   |  POST /v2/text-to-video        |
   |  { prompt, model, ... }        |
   |-------------------------------->|
   |                                 |  validate request
   |                                 |  create job (status: queued)
   |  202 Accepted                   |  enqueue job
   |  { job_id, poll_url, ... }      |
   |<--------------------------------|
   |                                 |
   |  GET /v2/jobs/{job_id}          |
   |-------------------------------->|
   |  { status: "queued",            |
   |    queue_position: 0 }          |
   |<--------------------------------|
   |                                 |  worker picks up job
   |  GET /v2/jobs/{job_id}          |  (status: processing)
   |-------------------------------->|
   |  { status: "processing",       |
   |    progress: 0.35 }             |
   |<--------------------------------|
   |                                 |
   |  GET /v2/jobs/{job_id}          |
   |-------------------------------->|
   |  { status: "processing",       |
   |    progress: 0.72 }             |
   |<--------------------------------|
   |                                 |  generation complete
   |  GET /v2/jobs/{job_id}          |
   |-------------------------------->|
   |  { status: "completed",        |
   |    result_url: "/v2/jobs/.../   |
   |      result" }                  |
   |<--------------------------------|
   |                                 |
   |  GET /v2/jobs/{job_id}/result   |
   |-------------------------------->|
   |  <binary: video/mp4>            |
   |<--------------------------------|
```

### Alternative: SSE stream (instead of polling)

Instead of repeated GET requests, you can open an SSE connection for real-time updates:

```
Frontend                          Backend
   |                                 |
   |  POST /v2/text-to-video        |
   |-------------------------------->|
   |  202 { job_id, stream_url }     |
   |<--------------------------------|
   |                                 |
   |  GET /v2/jobs/{job_id}/stream   |
   |  (EventSource / SSE)            |
   |================================>|  (long-lived connection)
   |                                 |
   |  event: status                  |
   |  data: { status: "processing" } |
   |<--------------------------------|
   |                                 |
   |  event: progress                |
   |  data: { progress: 0.35,       |
   |    step: 7, total_steps: 20 }   |
   |<--------------------------------|
   |                                 |
   |  event: progress                |
   |  data: { progress: 0.72, ... }  |
   |<--------------------------------|
   |                                 |
   |  event: completed               |
   |  data: { result_url: "..." }    |
   |<--------------------------------|
   |                                 |
   |  GET /v2/jobs/{job_id}/result   |
   |-------------------------------->|
   |  <binary: video/mp4>            |
   |<--------------------------------|
```

**Recommendation**: Use SSE for the primary UX (smoother progress bar). Fall back to polling if EventSource is unavailable or if the SSE connection drops.

---

## 4. Polling Strategy

If using polling instead of SSE:

| Phase | Interval | Reason |
|---|---|---|
| `queued` | Every 2 seconds | Check if processing has started |
| `processing` (0-50%) | Every 1 second | Show fast initial progress |
| `processing` (50-100%) | Every 1 second | Keep updating smoothly |
| After 60 seconds | Every 3 seconds | Reduce load on long jobs |
| After 120 seconds | Every 5 seconds | Further backoff for very long jobs |

### Backoff rules

- Start polling immediately after receiving the 202 response.
- Use the intervals above. Do not poll faster than 1 second.
- Stop polling when `status` is `completed`, `failed`, or `cancelled`.
- If a poll request fails (network error), retry after 3 seconds up to 5 times before showing an error.

---

## 5. Progress Display

The `progress` field in the job status response is a float from `0.0` to `1.0`:

| Value | Meaning |
|---|---|
| `null` | Job is queued, not started yet |
| `0.0` | Processing just started (pipeline setup) |
| `0.01` - `0.99` | Denoising steps in progress |
| `1.0` | Generation complete, result ready |

### Mapping progress to UI

```
progress == null  -->  "Waiting in queue..." (show spinner)
progress == 0.0   -->  "Starting..." (show 0%)
0.0 < progress < 1.0 --> show progress bar at Math.round(progress * 100) + "%"
progress == 1.0   -->  "Complete!" (fetch result)
```

### Multi-stage pipelines

For pro-model video (which has 2 denoising stages), the backend maps progress across both stages:

- Stage 1 (main denoising): progress `0.0` - `0.7`
- Stage 2 (refinement): progress `0.7` - `0.95`
- Encoding/muxing: progress `0.95` - `1.0`

You do not need to handle this -- just display the `progress` value directly. The backend normalizes it for you.

### Queue position

When `status` is `queued`, the response includes `queue_position` (0-based). Display this as "Position N in queue" or "Your job is next" when position is 0.

---

## 6. Error Handling

### Job failure

When `status` is `"failed"`, the response includes an `error` object:

```json
{
  "job_id": "abc123...",
  "status": "failed",
  "error": {
    "code": "generation_failed",
    "message": "CUDA out of memory"
  }
}
```

Error codes and suggested user messages:

| Code | User-facing message |
|---|---|
| `generation_failed` | "Generation failed. Try again or use a lower resolution." |
| `cuda_oom` | "Not enough GPU memory. Try a lower resolution or shorter duration." |
| `invalid_input` | "The input file could not be processed. Please re-upload." |
| `upload_not_found` | "The uploaded file has expired. Please re-upload and try again." |
| `timeout` | "Generation took too long. Try a shorter duration or lower resolution." |
| `cancelled` | "Job was cancelled." |
| `internal_error` | "Something went wrong on the server. Please try again." |

### HTTP error responses (non-job errors)

These are returned immediately at submit time -- no job is created.

| Status | When | Action |
|---|---|---|
| `422` | Validation error (bad params) | Show validation error to user. Fix form inputs. |
| `429` | Queue is full | Show "Server is busy, please wait." Retry after `Retry-After` header (seconds). |
| `503` | Server not ready (models loading) | Show "Server is starting up. Please wait." Retry in 10s. |
| `401` | Bad API key | Show auth error. |
| `404` | Job not found (expired or invalid ID) | Job expired or server restarted. Re-submit. |
| `409` | Result requested but job not done | Keep polling -- result is not ready yet. |

### Network errors during polling

- If a poll request fails, retry up to 5 times with 3-second intervals.
- If all retries fail, show "Connection lost. Retrying..." and continue retrying every 10 seconds.
- Do NOT cancel the job on network errors -- the job continues running on the server. Just resume polling when connectivity returns.
- If the page is refreshed or the app is reopened, you can resume polling if you saved the `job_id` (e.g., in localStorage).

### Server restart

If the server restarts while a job is running, all in-memory jobs are lost. Polling will return `404`. Treat this as a terminal error and ask the user to re-submit.

---

## 7. TypeScript Code Examples

### Types

```typescript
// Job status returned by the API
type JobStatus = "queued" | "processing" | "completed" | "failed" | "cancelled";

interface JobError {
  code: string;
  message: string;
}

interface JobSubmissionResponse {
  job_id: string;
  status: "queued";
  created_at: string;
  poll_url: string;
  stream_url: string;
}

interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  type: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress: number | null;
  queue_position: number | null;
  error: JobError | null;
  result_url: string | null;
  result_media_type: string | null;
}
```

### Submit a job

```typescript
const API_BASE = "http://192.168.1.100:8090";
const API_KEY = "your-api-key";

async function submitJob(
  endpoint: string,
  body: Record<string, unknown>
): Promise<JobSubmissionResponse> {
  const res = await fetch(`${API_BASE}/v2/${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json();
    if (res.status === 429) {
      const retryAfter = res.headers.get("Retry-After") ?? "30";
      throw new Error(`Queue full. Retry in ${retryAfter}s`);
    }
    throw new Error(err.message ?? `HTTP ${res.status}`);
  }

  return res.json();
}
```

### Poll for status (simple approach)

```typescript
async function pollUntilDone(
  jobId: string,
  onProgress?: (progress: number | null, status: JobStatus) => void
): Promise<JobStatusResponse> {
  const pollInterval = (elapsed: number): number => {
    if (elapsed > 120_000) return 5000;
    if (elapsed > 60_000) return 3000;
    return 1000;
  };

  const startTime = Date.now();
  let retries = 0;

  while (true) {
    const elapsed = Date.now() - startTime;
    await sleep(pollInterval(elapsed));

    let status: JobStatusResponse;
    try {
      const res = await fetch(`${API_BASE}/v2/jobs/${jobId}`, {
        headers: { Authorization: `Bearer ${API_KEY}` },
      });

      if (res.status === 404) {
        throw new Error("Job not found. It may have expired.");
      }
      if (!res.ok) {
        throw new Error(`Poll failed: HTTP ${res.status}`);
      }

      status = await res.json();
      retries = 0; // reset on success
    } catch (err) {
      retries++;
      if (retries >= 5) throw err;
      continue;
    }

    onProgress?.(status.progress, status.status);

    if (status.status === "completed") return status;
    if (status.status === "failed") {
      throw new Error(status.error?.message ?? "Job failed");
    }
    if (status.status === "cancelled") {
      throw new Error("Job was cancelled");
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
```

### Stream progress via SSE (recommended)

```typescript
function streamJobProgress(
  jobId: string,
  callbacks: {
    onProgress?: (progress: number, step: number, totalSteps: number) => void;
    onCompleted?: (resultUrl: string, mediaType: string) => void;
    onFailed?: (error: JobError) => void;
    onStatusChange?: (status: JobStatus) => void;
  }
): EventSource {
  const url = `${API_BASE}/v2/jobs/${jobId}/stream`;
  const source = new EventSource(url);

  source.addEventListener("status", (e: MessageEvent) => {
    const data = JSON.parse(e.data);
    callbacks.onStatusChange?.(data.status);
  });

  source.addEventListener("progress", (e: MessageEvent) => {
    const data = JSON.parse(e.data);
    callbacks.onProgress?.(data.progress, data.step, data.total_steps);
  });

  source.addEventListener("completed", (e: MessageEvent) => {
    const data = JSON.parse(e.data);
    callbacks.onCompleted?.(data.result_url, data.result_media_type);
    source.close();
  });

  source.addEventListener("error", (e: MessageEvent) => {
    // SSE "error" event with data = job failure
    if (e.data) {
      const data = JSON.parse(e.data);
      callbacks.onFailed?.(data.error);
      source.close();
    }
    // Otherwise it's a connection error -- EventSource auto-reconnects
  });

  return source;
}
```

**Note on SSE auth**: `EventSource` does not support custom headers. If your backend requires `Authorization` for SSE, pass the API key as a query parameter: `/v2/jobs/{id}/stream?token=<key>`. Alternatively, use `fetch` with `ReadableStream` for SSE with headers (see below).

### SSE with fetch (supports auth headers)

```typescript
async function streamJobWithFetch(
  jobId: string,
  onEvent: (eventType: string, data: Record<string, unknown>) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/v2/jobs/${jobId}/stream`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
  });

  if (!res.ok || !res.body) {
    throw new Error(`Stream failed: HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    let currentEvent = "message";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7);
      } else if (line.startsWith("data: ")) {
        const data = JSON.parse(line.slice(6));
        onEvent(currentEvent, data);
      }
      // Ignore empty lines and comments (`: heartbeat`)
    }
  }
}
```

### Fetch the result

```typescript
async function fetchResult(jobId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/v2/jobs/${jobId}/result`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
  });

  if (res.status === 409) {
    throw new Error("Result not ready yet");
  }
  if (!res.ok) {
    throw new Error(`Fetch failed: HTTP ${res.status}`);
  }

  return res.blob();
}
```

### Cancel a job

```typescript
async function cancelJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/v2/jobs/${jobId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${API_KEY}` },
  });

  if (res.status === 409) {
    // Job already finished, nothing to cancel
    return;
  }
  if (!res.ok) {
    throw new Error(`Cancel failed: HTTP ${res.status}`);
  }
}
```

### Full example: text-to-video with progress bar

```typescript
async function generateVideo(prompt: string): Promise<Blob> {
  // 1. Submit
  const submission = await submitJob("text-to-video", {
    prompt,
    model: "ltx-2-3-fast",
    resolution: "1920x1080",
    duration: 5,
    fps: 24,
    generate_audio: false,
  });

  console.log("Job submitted:", submission.job_id);

  // 2. Poll for progress
  const completed = await pollUntilDone(
    submission.job_id,
    (progress, status) => {
      if (progress === null) {
        updateUI("Waiting in queue...");
      } else {
        updateUI(`Generating: ${Math.round(progress * 100)}%`);
      }
    }
  );

  // 3. Fetch result
  const blob = await fetchResult(completed.job_id);
  console.log("Got result:", completed.result_media_type, blob.size, "bytes");
  return blob;
}

// Alternative using SSE:
async function generateVideoSSE(prompt: string): Promise<Blob> {
  // 1. Submit
  const submission = await submitJob("text-to-video", {
    prompt,
    model: "ltx-2-3-fast",
    resolution: "1920x1080",
    duration: 5,
    fps: 24,
    generate_audio: false,
  });

  // 2. Stream progress
  return new Promise((resolve, reject) => {
    streamJobProgress(submission.job_id, {
      onStatusChange: (status) => {
        if (status === "queued") updateUI("Waiting in queue...");
        if (status === "processing") updateUI("Generating...");
      },
      onProgress: (progress) => {
        updateUI(`Generating: ${Math.round(progress * 100)}%`);
      },
      onCompleted: async (resultUrl) => {
        try {
          const blob = await fetchResult(submission.job_id);
          resolve(blob);
        } catch (err) {
          reject(err);
        }
      },
      onFailed: (error) => {
        reject(new Error(error.message));
      },
    });
  });
}
```

---

## 8. Response Format Reference

### Submit response (202 Accepted)

```json
{
  "job_id": "f47ac10b58cc4372a5670e02b2c3d479",
  "status": "queued",
  "created_at": "2026-03-07T14:30:00Z",
  "poll_url": "/v2/jobs/f47ac10b58cc4372a5670e02b2c3d479",
  "stream_url": "/v2/jobs/f47ac10b58cc4372a5670e02b2c3d479/stream"
}
```

### Poll response (200 OK) -- queued

```json
{
  "job_id": "f47ac10b58cc4372a5670e02b2c3d479",
  "status": "queued",
  "type": "text-to-video",
  "created_at": "2026-03-07T14:30:00Z",
  "started_at": null,
  "completed_at": null,
  "progress": null,
  "queue_position": 0,
  "error": null,
  "result_url": null,
  "result_media_type": null
}
```

### Poll response (200 OK) -- processing

```json
{
  "job_id": "f47ac10b58cc4372a5670e02b2c3d479",
  "status": "processing",
  "type": "text-to-video",
  "created_at": "2026-03-07T14:30:00Z",
  "started_at": "2026-03-07T14:30:01Z",
  "completed_at": null,
  "progress": 0.45,
  "queue_position": null,
  "error": null,
  "result_url": null,
  "result_media_type": null
}
```

### Poll response (200 OK) -- completed

```json
{
  "job_id": "f47ac10b58cc4372a5670e02b2c3d479",
  "status": "completed",
  "type": "text-to-video",
  "created_at": "2026-03-07T14:30:00Z",
  "started_at": "2026-03-07T14:30:01Z",
  "completed_at": "2026-03-07T14:30:17Z",
  "progress": 1.0,
  "queue_position": null,
  "error": null,
  "result_url": "/v2/jobs/f47ac10b58cc4372a5670e02b2c3d479/result",
  "result_media_type": "video/mp4"
}
```

### Poll response (200 OK) -- failed

```json
{
  "job_id": "f47ac10b58cc4372a5670e02b2c3d479",
  "status": "failed",
  "type": "text-to-video",
  "created_at": "2026-03-07T14:30:00Z",
  "started_at": "2026-03-07T14:30:01Z",
  "completed_at": "2026-03-07T14:30:05Z",
  "progress": 0.15,
  "queue_position": null,
  "error": {
    "code": "cuda_oom",
    "message": "CUDA out of memory"
  },
  "result_url": null,
  "result_media_type": null
}
```

### SSE events

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

The server sends `: heartbeat` comments every 15 seconds to keep the connection alive.

### Result endpoint

`GET /v2/jobs/{job_id}/result` returns raw binary with the appropriate `Content-Type` header (`video/mp4` or `image/png`). This is the same format as the old v1 sync response body.

---

## 9. Migration Checklist

### Before starting

- [ ] Confirm v2 endpoints are deployed and accessible
- [ ] Verify `/health` response includes `queue` field
- [ ] Test submit + poll + fetch flow manually with curl

### Code changes

- [ ] Define TypeScript types for `JobSubmissionResponse`, `JobStatusResponse`, `JobError`
- [ ] Create `submitJob()` function (POST to `/v2/...`, expect 202)
- [ ] Create `pollUntilDone()` or `streamJobProgress()` function
- [ ] Create `fetchResult()` function (GET `/v2/jobs/{id}/result`, returns Blob)
- [ ] Create `cancelJob()` function (DELETE `/v2/jobs/{id}`)
- [ ] Replace every call site that POSTs to `/v1/text-to-video` (and other gen endpoints):
  - [ ] `POST /v1/text-to-video` --> submit + poll + fetch
  - [ ] `POST /v1/image-to-video` --> submit + poll + fetch
  - [ ] `POST /v1/audio-to-video` --> submit + poll + fetch
  - [ ] `POST /v1/retake` --> submit + poll + fetch
  - [ ] `POST /v1/text-to-image` --> submit + poll + fetch
  - [ ] `POST /v1/image-to-image` --> submit + poll + fetch
- [ ] Do NOT change these endpoints (they remain on v1, same behavior):
  - [ ] `POST /v1/chat/completions`
  - [ ] `POST /v1/upload` + `PUT /uploads/put/{id}`
  - [ ] `GET /health`

### UX changes

- [ ] Add progress bar or percentage display during generation
- [ ] Show "Waiting in queue..." when `status == "queued"` with queue position
- [ ] Show "Generating: N%" when `status == "processing"`
- [ ] Handle `429` (queue full) -- show "Server is busy" with retry
- [ ] Handle `failed` status -- show user-friendly error based on error code
- [ ] Handle network errors during polling -- retry with backoff, don't cancel
- [ ] Add cancel button (calls `DELETE /v2/jobs/{id}`)
- [ ] Optionally: persist `job_id` in localStorage to survive page refreshes

### Testing

- [ ] Test text-to-video (fast model) -- should complete in ~16s
- [ ] Test text-to-video (pro model) -- should complete in ~63s
- [ ] Test text-to-image (flux) -- should complete in ~51s
- [ ] Test with Cloudflare Tunnel -- verify no 100s timeout
- [ ] Test queue full (429) -- submit many jobs rapidly
- [ ] Test job failure -- use invalid parameters that pass validation but fail at generation
- [ ] Test cancel -- submit a job and cancel while queued / while processing
- [ ] Test network interruption -- disconnect during polling, reconnect, verify resume
- [ ] Test server restart during job -- verify 404 handling
