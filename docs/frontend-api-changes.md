# Frontend API Changes

> **Audience**: taco-desktop frontend team
> **Date**: 2026-03-07 (original) · 2026-04-09 (Flux 2 LoRA added)
> **Server**: `http://<host>:8090`

This document covers all recent backend changes that affect the frontend. Read it top to bottom before starting migration work.

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Async Job Queue (v2 Endpoints)](#2-async-job-queue-v2-endpoints)
3. [New Model: ltx-2-3-hq](#3-new-model-ltx-2-3-hq)
4. [Multi-Keyframe Image-to-Video](#4-multi-keyframe-image-to-video)
5. [Temporal Retake (Now Functional)](#5-temporal-retake-now-functional)
6. [TypeScript Definitions](#6-typescript-definitions)
7. [Migration Checklist](#7-migration-checklist)
8. [Flux 2 Image LoRAs (v1.1, 2026-04-09)](#8-flux-2-image-loras-v11-2026-04-09)

---

## 1. Authentication

All endpoints except `GET /health` now require a Bearer token.

```
Authorization: Bearer <api-key>
```

If missing or invalid, the server returns `401`:

```json
{ "error": "Invalid or missing API key", "message": "Invalid or missing API key" }
```

**Endpoints that do NOT require auth:**
- `GET /health`

**Endpoints that DO require auth:**
- Everything else (all v1, v2, upload, chat endpoints)

**SSE caveat**: `EventSource` does not support custom headers. If you use native `EventSource` for the SSE stream endpoint, pass the token as a query parameter: `/v2/jobs/{id}/stream?token=<key>`. Alternatively, use `fetch()` with `ReadableStream` to send the `Authorization` header (see examples below).

---

## 2. Async Job Queue (v2 Endpoints)

### Why

Cloudflare Tunnel enforces a 100-second HTTP timeout. Some generation requests (pro model, high-res, audio+video) exceed this. The v2 endpoints use a **submit-poll-fetch** pattern where no single HTTP request blocks for more than a few seconds.

### v1 vs v2

The request bodies are **identical** between v1 and v2. Only the response changes.

| v1 (synchronous, deprecated) | v2 (async) | Request body |
|---|---|---|
| `POST /v1/text-to-video` | `POST /v2/text-to-video` | `TextToVideoRequest` |
| `POST /v1/image-to-video` | `POST /v2/image-to-video` | `ImageToVideoRequest` |
| `POST /v1/audio-to-video` | `POST /v2/audio-to-video` | `AudioToVideoRequest` |
| `POST /v1/retake` | `POST /v2/retake` | `RetakeRequest` |
| `POST /v1/text-to-image` | `POST /v2/text-to-image` | `TextToImageRequest` |
| `POST /v1/image-to-image` | `POST /v2/image-to-image` | `ImageToImageRequest` |

**v1 behavior**: Returns raw binary (`video/mp4` or `image/png`) after 16-63+ seconds.
**v2 behavior**: Returns `202 Accepted` with a JSON body containing a job ID. You then poll for progress and fetch the binary when done.

### Unchanged endpoints (no migration needed)

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | No auth. Now includes `queue` stats. |
| `POST` | `/v1/chat/completions` | Synchronous. Fast (~1s). |
| `POST` | `/v1/upload` | Synchronous. Returns `upload_url` + `storage_uri`. |
| `PUT` | `/uploads/put/{id}` | Synchronous. File upload. |

### New job management endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v2/jobs/{job_id}` | Poll job status and progress |
| `GET` | `/v2/jobs/{job_id}/result` | Download completed result (binary) |
| `DELETE` | `/v2/jobs/{job_id}` | Cancel a queued or running job |

> **Note**: The SSE streaming endpoint (`GET /v2/jobs/{job_id}/stream`) is referenced in submit responses but is **not yet implemented**. Use polling for now. The `stream_url` field in the submit response is reserved for future use.

### Submit response (202 Accepted)

```json
{
  "job_id": "f47ac10b58cc4372a5670e02b2c3d479",
  "status": "queued",
  "poll_url": "/v2/jobs/f47ac10b58cc4372a5670e02b2c3d479",
  "stream_url": "/v2/jobs/f47ac10b58cc4372a5670e02b2c3d479/stream"
}
```

### Poll response (GET /v2/jobs/{job_id})

```json
{
  "job_id": "f47ac10b58cc4372a5670e02b2c3d479",
  "status": "queued | processing | completed | failed | cancelled",
  "type": "text-to-video",
  "progress": 0.45,
  "queue_position": 0,
  "error": null,
  "result_url": null,
  "result_media_type": null
}
```

Field behavior by status:

| Status | `progress` | `queue_position` | `error` | `result_url` |
|---|---|---|---|---|
| `queued` | `null` | 0-based int | `null` | `null` |
| `processing` | `0.0` - `0.99` | `null` | `null` | `null` |
| `completed` | `1.0` | `null` | `null` | `"/v2/jobs/{id}/result"` |
| `failed` | last value | `null` | `{ code, message }` | `null` |
| `cancelled` | last value | `null` | `null` | `null` |

### Result endpoint (GET /v2/jobs/{job_id}/result)

Returns raw binary with `Content-Type: video/mp4` or `Content-Type: image/png`.

Returns `409` if the job is not yet completed. Returns `404` if the result has expired (results are cleaned up after 10 minutes).

### Cancel endpoint (DELETE /v2/jobs/{job_id})

Returns `200` with `{ "job_id": "...", "status": "cancelled" }`.

Returns `409` if the job is already finished (completed, failed, or cancelled).

### Error responses at submit time

These are returned immediately -- no job is created.

| Status | Meaning | Frontend action |
|---|---|---|
| `401` | Invalid or missing API key | Show auth error |
| `422` | Validation error (bad params) | Show validation error, fix form inputs |
| `429` | Queue full (max 10 pending jobs) | Show "Server is busy". Retry after `Retry-After` header (30s). |
| `500` | Pipeline not loaded | Show "Server is starting up" |

### Error responses during polling

| Status | Meaning | Frontend action |
|---|---|---|
| `404` | Job not found (expired or server restarted) | Re-submit the job |
| `409` | Result not ready (on `/result` endpoint) | Keep polling |

### Job failure error codes

When `status` is `"failed"`, the `error` object contains:

| `error.code` | Meaning | Suggested user message |
|---|---|---|
| `generation_failed` | Generic generation error | "Generation failed. Try again or use a lower resolution." |
| `cuda_oom` | GPU out of memory | "Not enough GPU memory. Try a lower resolution or shorter duration." |

### Progress display

The `progress` field is a float from `0.0` to `1.0`:

```
progress == null      -> "Waiting in queue..." (show spinner + queue position)
progress == 0.0       -> "Starting..." (show 0%)
0.0 < progress < 1.0  -> show Math.round(progress * 100) + "%"
progress == 1.0       -> "Complete!" (fetch result)
```

For multi-stage pipelines (pro and hq models), the backend normalizes progress across all stages. You do not need to handle this -- just display `progress` directly.

### Polling strategy

| Phase | Interval |
|---|---|
| `queued` | Every 2 seconds |
| `processing` (first 60s) | Every 1 second |
| `processing` (60-120s) | Every 3 seconds |
| `processing` (120s+) | Every 5 seconds |

Stop polling when `status` is `completed`, `failed`, or `cancelled`.

On network errors during polling, retry up to 5 times at 3-second intervals. Do NOT cancel the job on network errors -- it continues running on the server. Resume polling when connectivity returns.

### Queue limits

- Maximum 10 pending jobs in the queue. Exceeding this returns `429`.
- Results expire after 10 minutes. Fetch them promptly.
- Jobs are processed one at a time (serialized). Queue position tells you where you are.
- All jobs are in-memory. A server restart loses all jobs.

### TypeScript: submit, poll, fetch flow

```typescript
const API_BASE = "http://192.168.1.100:8090";
const API_KEY = "your-api-key";

const headers = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${API_KEY}`,
};

// --- Submit ---

async function submitJob(
  endpoint: string,
  body: Record<string, unknown>
): Promise<JobSubmitResponse> {
  const res = await fetch(`${API_BASE}/v2/${endpoint}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (res.status === 429) {
    const retryAfter = res.headers.get("Retry-After") ?? "30";
    throw new Error(`Queue full. Retry in ${retryAfter}s`);
  }
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message ?? `HTTP ${res.status}`);
  }

  return res.json();
}

// --- Poll ---

async function pollUntilDone(
  jobId: string,
  onProgress?: (progress: number | null, queuePosition: number | null) => void
): Promise<JobStatusResponse> {
  const startTime = Date.now();
  let retries = 0;

  while (true) {
    const elapsed = Date.now() - startTime;
    const interval =
      elapsed > 120_000 ? 5000 : elapsed > 60_000 ? 3000 : elapsed > 0 ? 1000 : 2000;
    await new Promise((r) => setTimeout(r, interval));

    let status: JobStatusResponse;
    try {
      const res = await fetch(`${API_BASE}/v2/jobs/${jobId}`, {
        headers: { Authorization: `Bearer ${API_KEY}` },
      });
      if (res.status === 404) throw new Error("Job not found (expired or server restarted)");
      if (!res.ok) throw new Error(`Poll failed: HTTP ${res.status}`);
      status = await res.json();
      retries = 0;
    } catch (err) {
      if (++retries >= 5) throw err;
      continue;
    }

    onProgress?.(status.progress, status.queue_position);

    if (status.status === "completed") return status;
    if (status.status === "failed") throw new Error(status.error?.message ?? "Job failed");
    if (status.status === "cancelled") throw new Error("Job was cancelled");
  }
}

// --- Fetch result ---

async function fetchResult(jobId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/v2/jobs/${jobId}/result`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
  });
  if (res.status === 409) throw new Error("Result not ready yet");
  if (res.status === 404) throw new Error("Result expired or not found");
  if (!res.ok) throw new Error(`Fetch failed: HTTP ${res.status}`);
  return res.blob();
}

// --- Cancel ---

async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${API_BASE}/v2/jobs/${jobId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${API_KEY}` },
  });
}
```

### Full example: generate a video

```typescript
async function generateVideo(prompt: string): Promise<Blob> {
  // 1. Submit
  const { job_id } = await submitJob("text-to-video", {
    prompt,
    model: "ltx-2-3-fast",
    resolution: "1920x1080",
    duration: 5,
    fps: 24,
    generate_audio: false,
  });

  // 2. Poll
  const completed = await pollUntilDone(job_id, (progress, queuePos) => {
    if (progress === null) {
      showStatus(`Waiting in queue (position ${(queuePos ?? 0) + 1})...`);
    } else {
      showStatus(`Generating: ${Math.round(progress * 100)}%`);
    }
  });

  // 3. Fetch
  return fetchResult(completed.job_id);
}
```

---

## 3. New Model: ltx-2-3-hq

A new model option `"ltx-2-3-hq"` is available alongside the existing models.

### Model comparison

| Model | Quality | Speed | Use case |
|---|---|---|---|
| `ltx-2-3-fast` | Good | ~16s | Previews, rapid iteration |
| `ltx-2-3-pro` | High | ~63s | Final renders |
| `ltx-2-3-hq` | Highest | ~73s (~15% slower than pro) | Maximum quality, best temporal coherence |

### Frontend change

Just pass `"ltx-2-3-hq"` in the `model` field. No other changes needed.

```typescript
await submitJob("text-to-video", {
  prompt: "A cat walking on a beach at sunset",
  model: "ltx-2-3-hq",  // <-- new option
  resolution: "1920x1080",
  duration: 5,
  fps: 24,
  generate_audio: false,
});
```

The model is available for all LTX video endpoints: text-to-video, image-to-video, audio-to-video, and retake.

### Updated `ModelName` type

```typescript
type ModelName = "ltx-2-3-fast" | "ltx-2-3-pro" | "ltx-2-3-hq";
```

---

## 4. Multi-Keyframe Image-to-Video

The image-to-video endpoint now supports multiple conditioning keyframes at specific frame positions, enabling start/end bookending and mid-video scene transitions.

### Before (still works)

```json
{
  "prompt": "A dog running",
  "image_uri": "storage://abc123",
  "model": "ltx-2-3-fast",
  "resolution": "1920x1080",
  "duration": 3,
  "fps": 24
}
```

### After (new `keyframes` field)

```json
{
  "prompt": "A smooth transition between two scenes",
  "keyframes": [
    { "image_uri": "storage://first-image-id", "frame_index": 0, "strength": 1.0 },
    { "image_uri": "storage://last-image-id", "frame_index": 72, "strength": 0.8 }
  ],
  "model": "ltx-2-3-fast",
  "resolution": "1920x1080",
  "duration": 3,
  "fps": 24
}
```

### Backward compatibility

- `image_uri` alone still works exactly as before (treated as a single keyframe at frame 0, strength 1.0).
- `keyframes` is the new way. You must provide one or the other.
- You **cannot** set both `image_uri` and `keyframes` (returns 422).

### Keyframe fields

| Field | Type | Default | Description |
|---|---|---|---|
| `image_uri` | `string` | required | `storage://` URI of the uploaded image |
| `frame_index` | `int` | `0` | Target frame in the output video (0-indexed) |
| `strength` | `float` | `1.0` | Conditioning strength (0.0 = no influence, 1.0 = maximum) |

### How frame_index maps to the video

The total frame count is computed from `duration` and `fps`, snapped to `8k+1`:

```
frames = 8 * round((duration * fps - 1) / 8) + 1
```

Examples:
- 3s at 24fps = 73 frames (indices 0-72)
- 5s at 24fps = 121 frames (indices 0-120)

**frame_index = 0** gives the strongest control (replaces latent tokens at the first frame). **frame_index > 0** provides softer guidance via cross-attention keyframe tokens.

### Validation rules

All of these return `422` if violated:

1. Either `image_uri` or `keyframes` must be provided (not both, not neither)
2. `keyframes` list must not be empty
3. Maximum 8 keyframes
4. No duplicate `frame_index` values
5. At most one keyframe can have `frame_index = 0`
6. `strength` must be between 0.0 and 1.0
7. `frame_index` must be >= 0

### TypeScript example: two-keyframe bookend

```typescript
// Upload two images first
const firstImageUri = await uploadFile(firstImageBlob);
const lastImageUri = await uploadFile(lastImageBlob);

// Calculate last frame index: 3s at 24fps = 73 frames, last = 72
const fps = 24;
const duration = 3;
const rawFrames = Math.floor(duration * fps);
const lastFrame = 8 * Math.round((rawFrames - 1) / 8); // = 72

const { job_id } = await submitJob("image-to-video", {
  prompt: "Smooth cinematic transition",
  keyframes: [
    { image_uri: firstImageUri, frame_index: 0, strength: 1.0 },
    { image_uri: lastImageUri, frame_index: lastFrame, strength: 0.8 },
  ],
  model: "ltx-2-3-fast",
  resolution: "1920x1080",
  duration,
  fps,
});
```

---

## 5. Temporal Retake (Now Functional)

The retake endpoint previously ignored `start_time`, `duration`, and `mode` -- it always regenerated the entire video from scratch. **This is now fixed.** The backend uses temporal masking to regenerate only the specified time range and only the specified modalities.

### Request body (unchanged)

```json
{
  "video_uri": "storage://source-video-id",
  "start_time": 1.0,
  "duration": 2.0,
  "mode": "replace_video_only",
  "prompt": "A different scene in this section"
}
```

### RetakeMode values and what they do

| Mode | Video | Audio | Effect |
|---|---|---|---|
| `replace_audio_and_video` | Regenerated | Regenerated | Both modalities regenerated in `[start_time, start_time + duration]` |
| `replace_video` | Regenerated | Regenerated | Same as `replace_audio_and_video` |
| `replace_video_only` | Regenerated | Preserved | Only video is regenerated; audio passes through unchanged |
| `replace_audio` | Preserved | Regenerated | Only audio is regenerated; video frames are identical to source |

### How it works now

- Frames/audio **outside** the `[start_time, start_time + duration]` window are preserved pixel-for-pixel from the source video.
- Frames/audio **inside** the window are regenerated based on the prompt.
- The transition at window boundaries is handled at the latent patch level. There is no explicit crossfade -- the model naturally blends at patch boundaries.

### Edge cases

| Scenario | Behavior |
|---|---|
| `start_time = 0`, `duration >= video_length` | Full regeneration (same as before) |
| `start_time + duration > video_length` | Regenerates to the end of the video |
| Very short duration (< 0.33s) | At least one latent patch will be regenerated |
| Source video has no audio track | Audio is generated from scratch for the retake region |

### TypeScript example: retake a section

```typescript
const { job_id } = await submitJob("retake", {
  video_uri: "storage://original-video-id",
  start_time: 2.0,
  duration: 1.5,
  mode: "replace_video_only",
  prompt: "An explosion in this section",
});

const completed = await pollUntilDone(job_id, (progress) => {
  showStatus(`Retaking: ${Math.round((progress ?? 0) * 100)}%`);
});

const resultBlob = await fetchResult(completed.job_id);
```

---

## 6. TypeScript Definitions

Copy these into your project. They match the actual server response shapes.

```typescript
// --- Model and enum types ---

type ModelName = "ltx-2-3-fast" | "ltx-2-3-pro" | "ltx-2-3-hq";
type ImageModelName = "flux2-dev";
type Resolution =
  | "1920x1080"
  | "1080x1920"
  | "2560x1440"
  | "1440x2560"
  | "3840x2160"
  | "2160x3840";
type RetakeMode =
  | "replace_audio_and_video"
  | "replace_video"
  | "replace_video_only"
  | "replace_audio";
type JobStatus = "queued" | "processing" | "completed" | "failed" | "cancelled";
type JobType =
  | "text-to-video"
  | "image-to-video"
  | "audio-to-video"
  | "retake"
  | "text-to-image"
  | "image-to-image";

// --- Request types ---

interface TextToVideoRequest {
  prompt: string;
  model: ModelName;
  resolution: Resolution;
  duration: number; // seconds, > 0, <= 30
  fps: number; // > 0, <= 60
  generate_audio?: boolean; // default false
  camera_motion?: string | null; // max 200 chars
}

interface KeyframeInput {
  image_uri: string; // storage:// URI
  frame_index?: number; // default 0, >= 0
  strength?: number; // default 1.0, 0.0-1.0
}

interface ImageToVideoRequest {
  prompt: string;
  image_uri?: string | null; // deprecated, use keyframes
  keyframes?: KeyframeInput[] | null; // new: 1-8 keyframes
  model: ModelName;
  resolution: Resolution;
  duration: number;
  fps: number;
  generate_audio?: boolean;
}

interface AudioToVideoRequest {
  prompt: string;
  audio_uri: string; // storage:// URI
  image_uri?: string | null;
  model: ModelName;
  resolution: Resolution;
  duration?: number; // default 6.0
  fps?: number; // default 24.0
}

interface RetakeRequest {
  video_uri: string; // storage:// URI
  start_time: number; // seconds, >= 0
  duration: number; // seconds, > 0, <= 30
  mode: RetakeMode;
  prompt?: string | null; // max 10000 chars
}

interface TextToImageRequest {
  prompt: string;
  model?: ImageModelName; // default "flux2-dev"
  width?: number; // default 1024, 64-4096, snapped to multiple of 16
  height?: number; // default 1024, 64-4096, snapped to multiple of 16
  num_inference_steps?: number; // default 50, 1-100
  guidance_scale?: number; // default 4.0, 0-20
  seed?: number | null;
}

interface ImageToImageRequest {
  prompt: string;
  image_uri: string; // storage:// URI
  model?: ImageModelName;
  width?: number;
  height?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  seed?: number | null;
}

interface ChatMessage {
  role: string;
  content: string | unknown[]; // string for text, array for multimodal
}

interface ChatCompletionRequest {
  model?: string; // default "gemma-3-12b-nvfp4"
  messages: ChatMessage[];
  temperature?: number; // default 0.7, 0-2.0
  max_tokens?: number; // default 512, 1-8192
}

// --- Response types ---

interface JobSubmitResponse {
  job_id: string;
  status: "queued";
  poll_url: string;
  stream_url: string;
}

interface JobError {
  code: string; // "generation_failed" | "cuda_oom"
  message: string;
}

interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  type: JobType;
  progress: number | null;
  queue_position: number | null;
  error: JobError | null;
  result_url: string | null;
  result_media_type: string | null; // "video/mp4" | "image/png"
}

interface UploadResponse {
  upload_url: string; // PUT URL for uploading file bytes
  storage_uri: string; // storage:// URI to reference in requests
  required_headers: Record<string, never>; // always empty
}

interface HealthResponse {
  status: "ok";
  ltx: "ready" | "not_loaded";
  flux: "ready" | "not_loaded";
  chat: "ready" | "not_loaded";
  queue: Record<string, number>; // e.g. { "queued": 2, "processing": 1 }
}
```

---

## 7. Migration Checklist

### Auth

- [ ] Add `Authorization: Bearer <key>` header to all API calls (except `/health`)
- [ ] Handle `401` responses globally

### v2 queue migration

- [ ] Replace all `POST /v1/<endpoint>` generation calls with `POST /v2/<endpoint>`
- [ ] Implement `submitJob()` helper (POST, expect 202, parse `job_id`)
- [ ] Implement `pollUntilDone()` helper with backoff
- [ ] Implement `fetchResult()` helper (GET `/v2/jobs/{id}/result`, returns Blob)
- [ ] Implement `cancelJob()` helper (DELETE `/v2/jobs/{id}`)
- [ ] Add progress bar UI (use `progress` field, 0.0-1.0)
- [ ] Add queue position display when `status == "queued"`
- [ ] Handle `429` (queue full) with retry after delay
- [ ] Handle `404` on poll (server restart) -- re-submit
- [ ] Handle network errors during polling -- retry, do not cancel
- [ ] Add cancel button wired to `DELETE /v2/jobs/{id}`
- [ ] Optionally persist `job_id` in localStorage to survive page refresh
- [ ] Do NOT change: `/v1/upload`, `/uploads/put/{id}`, `/v1/chat/completions`, `/health`

### New model

- [ ] Add `"ltx-2-3-hq"` to model selector UI
- [ ] Update `ModelName` type to include `"ltx-2-3-hq"`

### Multi-keyframe i2v

- [ ] Update `ImageToVideoRequest` type to include optional `keyframes` field
- [ ] Build UI for adding multiple keyframes (image + frame_index + strength)
- [ ] Validate: max 8 keyframes, no duplicate frame_index, max one at frame 0
- [ ] Calculate frame count from duration/fps for frame_index picker: `8 * round((duration * fps - 1) / 8) + 1`
- [ ] Keep backward compat: `image_uri` alone still works

### Temporal retake

- [ ] Update retake UI to actually use `start_time` and `duration` (they work now)
- [ ] Add mode selector: `replace_audio_and_video`, `replace_video_only`, `replace_audio`
- [ ] Show timeline scrubber for selecting retake region
- [ ] Update `RetakeMode` type

---

## 8. Flux 2 Image LoRAs (v1.1, 2026-04-09)

Flux 2 Dev and Flux 2 Klein now support **per-request LoRAs** via a **folder-drop discovery system**. This is a **separate** system from the existing LTX video LoRAs — distinct endpoints, distinct registry, distinct ID namespace.

### 8.1 Discovery endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/flux-loras` | GET | Yes | List discovered Flux LoRAs |
| `/v1/flux-loras/rescan` | POST | Yes | Re-scan `flux_loras/` directory |

There is intentionally **no upload/delete endpoint**. Files are managed server-side (operator drops `.safetensors` into `flux_loras/`, optionally with a same-named `.json` sidecar for display metadata).

### 8.2 Response shape (`GET /v1/flux-loras`)

```json
{
  "loras": [
    {
      "id": "my-style-v2",
      "name": "My Style v2",
      "filename": "MyStyleV2.safetensors",
      "size_bytes": 201431592,
      "model_compat": ["flux2-dev"],
      "description": "Painterly style, 2000 steps",
      "trigger_word": "inthestyleof"
    }
  ],
  "count": 1
}
```

- `id` is a slug of the filename stem (stable across restarts; changes if operator renames the file).
- `model_compat` is advisory only — the backend does not enforce it. The frontend should filter the dropdown client-side by the currently selected model.

### 8.3 Request shape (all three Flux endpoints)

`TextToImageRequest`, `ImageToImageRequest`, and `ImageEditRequest` gain an optional `lora` field:

```json
{
  "prompt": "a cyberpunk cat",
  "model": "flux2-dev",
  "lora": {"id": "my-style-v2", "strength": 0.8},
  "seed": 42,
  "turbo": true
}
```

Shape is identical to the LTX `lora` field (`{id, strength}`, `strength` is 0.0–2.0, default 1.0). Omit the `lora` field or pass `null` to generate without a LoRA.

### 8.4 Latency behavior (important) — v1.1.1 update

- **First request** with a new `(model, lora_id)` pair → server does a full pipeline reload (~30–60 s extra on Dev, including CPU offload hook setup). This is required because the LoRA adapter has to be attached before the bf16 pipeline is handed to the offload manager.
- **Subsequent requests** with the **same** `(model, lora_id)` pair → cache hit, normal generation speed.
- **Changing `strength` is now FREE** (v1.1.1 change). Strength is applied at inference time via a runtime `set_adapters([...], [strength])` call — O(ms), no reload. Users can scrub the strength slider at will.
- Changing `lora_id`, switching models (`flux2-dev` ↔ `flux2-klein`), or removing the LoRA field triggers the full reload path.

**Backend change (v1.1.1):** FP8 layerwise casting was dropped; Flux 2 Dev now runs full bf16 with CPU offload. This eliminates a screendoor/grid artifact that was visible in v1.1 outputs, and also decouples strength from the cache key. The `lora: {id, strength}` request shape is unchanged — **no client-side code changes are required** to benefit from the strength-slider UX improvement. Clients that were hiding the strength slider due to the reload cost should now expose it freely.

Surface a "Loading LoRA…" indicator on the first call after a `(model, lora_id)` change so users understand the ~30–60 s delay. Do **not** show the indicator for strength-only changes.

### 8.5 Error cases

- `404 {"error":"Flux LoRA not found: <id>"}` — the requested LoRA isn't in the registry (stale cache or operator deleted the file). Clear the client's selection and refetch `GET /v1/flux-loras`.
- `500` during generation with a LoRA — usually a malformed LoRA file or key-format mismatch. Surface the error and suggest removing the LoRA.

### 8.6 Migration checklist (Flux LoRA)

- [ ] Add `FluxLoRAInfo` / `FluxLoRAListResponse` TypeScript types
- [ ] Add `lora?: LoRAInput | null` to `TextToImageRequest`, `ImageToImageRequest`, `ImageEditRequest`
- [ ] Fetch Flux LoRA list on form open (`GET /v1/flux-loras`)
- [ ] LoRA dropdown in t2i / i2i / image-edit forms with strength slider (0.0–2.0, default 1.0)
- [ ] Client-side filter dropdown by `model_compat` against currently selected model
- [ ] "Refresh" button → `POST /v1/flux-loras/rescan` then refetch list
- [ ] "Loading LoRA…" indicator on first request after a `(model, lora, strength)` change
- [ ] Handle 404 by clearing selection and refetching
- [ ] Empty-state copy explaining folder-drop (no upload UI)

**Full integration details** (including UI flow and UX patterns): see `docs/frontend-lora-integration.md` section 10.
