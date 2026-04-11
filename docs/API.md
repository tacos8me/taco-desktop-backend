# taco-backend — Complete API Reference

**Server version:** v1.1.7 (2026-04-11)
**Base URL:** `http://<host>:8090`
**Auth:** Bearer token in `Authorization` header. Required on ALL endpoints except `/health` and `/v1/approved-images/events`.
**Content-Type:** JSON requests unless noted. Responses are JSON unless a binary media type is documented.

> **Maintenance rule:** Any commit that adds, removes, or changes an endpoint (URL, method, request/response shape, status codes) MUST update this file in the same commit. This doc is the contract.

---

## Table of contents

1. [Conventions](#conventions)
2. [Common request types](#common-request-types)
3. [System & health](#system--health)
4. [v1 synchronous generation](#v1-synchronous-generation) — blocks until media is ready
5. [v2 async generation + jobs](#v2-async-generation--jobs) — 202 → poll → fetch
6. [Uploads](#uploads)
7. [LoRAs (LTX)](#loras-ltx)
8. [Flux LoRAs](#flux-loras)
9. [Chat completions](#chat-completions)
10. [SSE session tokens](#sse-session-tokens)
11. [Approved images](#approved-images)
12. [Character-consistency vision ranking](#character-consistency-vision-ranking)
13. [Generation history](#generation-history)
14. [Compositions](#compositions)
15. [Error codes](#error-codes)
16. [Endpoint index](#endpoint-index)

---

## Conventions

### Authentication

```
Authorization: Bearer <api-key>
```

- Keys live in `.api_keys` on the server (one per line). Empty file ⇒ auth disabled.
- Constant-time compare (`secrets.compare_digest`) against every configured key.
- `401 {"error": "Invalid or missing API key"}` on any mismatch.
- **No-auth endpoints:** `/health`, `/v1/approved-images/events` (SSE uses a disposable token — see §10).

### Error shape

Every error response has this exact body:

```json
{"error": "<message>", "message": "<message>", "detail": "<message>"}
```

All three fields carry the same string so clients can use whichever they already parse. Filesystem paths are redacted to `"Internal server error"` before leaving the process.

### Storage URIs

Uploads and generated media are referenced by `storage://<uuid>` URIs. They resolve to files under the server's `UPLOAD_DIR`. Clients never see raw paths. See §6 for the upload flow.

### CORS

`^https?://(localhost|192\.168\.\d+\.\d+)(:\d+)?$` is allowed. Methods: `GET, POST, PUT, DELETE`. Headers: `Authorization, Content-Type`.

### Media types

| Kind | Content-Type |
|---|---|
| Video generation result | `video/mp4` |
| Image generation result | `image/webp` (lossless VP8L) |
| Preview frame (v2 `/preview`) | `image/jpeg` (quality 80) |
| Thumbnail (history) | `image/jpeg` (quality 70, 256 px wide) |
| Chat completions | `application/json` (OpenAI-shaped) |
| SSE | `text/event-stream` |

### Rate limiting / queue

- Single shared `_inference_lock` serializes all GPU work (FP8 + LTX cannot run concurrently in the same process).
- v2 job queue cap: `config.MAX_QUEUE_DEPTH`. Exceeding returns `429 {"error": "queue_full"}` with `Retry-After: 30`.
- `503` with `Retry-After: 300` while the system is paused.

### Single-GPU swap mode (v1.1.4+)

LTX and Flux both target `cuda:0` and are **mutually exclusive**. The server auto-swaps:

- Video request → evicts Flux if needed, ensures LTX resident (cold load ≈ 25–30 s after a Flux session).
- Image request → evicts LTX if resident (≈ 3 s), Flux's own offload hooks page weights in (≈ 15–60 s first call).
- Long stretches of image-only or video-only workloads pay zero swap overhead.
- `cuda:1` is reserved for external training — taco-backend never touches it.

---

## Common request types

### `LoRAInput`

```json
{"id": "my-style", "strength": 1.0}
```

| Field | Type | Constraints |
|---|---|---|
| `id` | string | Must exist in the corresponding registry (`/v1/loras` for LTX, `/v1/flux-loras` for Flux). |
| `strength` | float | `0.0 ≤ x ≤ 2.0`, default `1.0`. Flux LoRAs are adapter-mode — strength changes are free. LTX LoRAs require a reload on strength change. |

### `KeyframeInput`

```json
{"image_uri": "storage://...", "frame_index": "first", "strength": 1.0}
```

| Field | Type | Notes |
|---|---|---|
| `image_uri` | string | `storage://<uuid>` — upload via §6 first. |
| `frame_index` | int \| `"first"` \| `"middle"` \| `"last"` | Default `0`. Negative ints count from end (`-1` = last). Symbolic values resolved after `num_frames` is computed. Duplicate resolved indices → `422`. Out-of-range → `422`. |
| `strength` | float | `0.0 ≤ x ≤ 1.0`. Recommended: first=1.0, middle=0.5, last=1.0. |

Up to 8 keyframes per request.

### Shared constraints

| Field | Constraint |
|---|---|
| `prompt` | ≤ 10 000 chars |
| `camera_motion` | ≤ 200 chars |
| `duration` | `0 < x ≤ 30` seconds |
| `fps` | `0 < x ≤ 60` |
| `model` (video) | `"ltx-2-3-fast"` \| `"ltx-2-3-pro"` \| `"ltx-2-3-hq"` |
| `model` (image) | `"flux2-dev"` \| `"flux2-klein"` |
| `resolution` | `"1920x1080"` \| `"1080x1920"` \| `"2560x1440"` \| `"1440x2560"` \| `"3840x2160"` \| `"2160x3840"` |
| `width` / `height` (Flux) | `64 ≤ x ≤ 4096`, snapped to multiples of 16 server-side |
| `num_inference_steps` (Flux) | `1 ≤ x ≤ 100`. Defaults: 50 (dev), 4 (edit/klein). Turbo mode overrides to 8. |
| `guidance_scale` (Flux) | `0 ≤ x ≤ 20`. Klein silently ignores this (distilled, no CFG). |

Frame counts are derived as `8k + 1` closest to `duration * fps`. Actual frame count can drift by ±4 frames.

---

## System & health

### `GET /health` — (no auth)

Liveness + model readiness. Never blocks.

```json
{
  "status": "ok" | "paused",
  "ltx": "ready" | "not_loaded" | "paused",
  "flux": "ready" | "not_loaded" | "paused",
  "chat": "ready" | "not_loaded",
  "queue": {"queued": 0, "processing": 0, "completed": 12, "failed": 1}
}
```

### `POST /v1/system/pause`

Evicts LTX and Flux, cancels all queued (not yet processing) v2 jobs, and flips the server into a paused state. Use before a training run.

- While paused: all generation endpoints return `503` with `Retry-After: 300`.
- `/health` still responds and reports `"paused"`.

Response: `{"status": "paused" | "already_paused"}`

### `POST /v1/system/resume`

Reloads LTX (and Flux if `LOAD_FLUX=1`). Response: `{"status": "ready" | "already_running"}`. On failure: `500 {"error": "resume_failed", "status": "paused"}`.

### `POST /v1/flux/unload`

Unloads Flux only (LTX stays). Response: `{"status": "unloaded" | "already_unloaded"}`.

### `POST /v1/flux/reload`

Reloads Flux only. Response: `{"status": "loaded" | "already_loaded"}`.

### `POST /v1/ltx/unload`

Unloads LTX only (Flux stays). In single-GPU mode this fully frees `cuda:0` so a subsequent Flux request has room. The next video request auto-reloads LTX (~25–30 s cold). Response: `{"status": "unloaded" | "already_unloaded"}`.

### `POST /v1/ltx/reload`

Reloads LTX. Response: `{"status": "loaded" | "already_loaded"}`.

---

## v1 synchronous generation

Synchronous endpoints block until the result media is in RAM and return it directly. Good for quick one-shots and for clients that don't want to poll. **For anything more than a few requests, use v2 (§5).**

All v1 generation endpoints can return:

- `503` — system paused
- `500` — Flux requested but `LOAD_FLUX=0`
- `422` — invalid LoRA, keyframe bounds, or retake content rejection
- `404` — referenced upload not found
- `500` — generic failure (error text sanitized of paths)

### `POST /v1/text-to-video`

**Body:** `TextToVideoRequest`

```json
{
  "prompt": "a cat walks across a sunlit hardwood floor",
  "model": "ltx-2-3-pro",
  "resolution": "1920x1080",
  "duration": 5.0,
  "fps": 24,
  "generate_audio": false,
  "camera_motion": "slow dolly in",
  "lora": {"id": "my-style", "strength": 1.0},
  "enhance_prompt": false
}
```

**Response:** `200 video/mp4` raw bytes.

### `POST /v1/image-to-video`

Takes either `image_uri` (single start frame) OR `keyframes` (up to 8). Mutually exclusive — providing both returns `422`.

```json
{
  "prompt": "she turns to face the camera",
  "image_uri": "storage://...",
  "image_strength": 0.85,
  "keyframes": null,
  "model": "ltx-2-3-pro",
  "resolution": "1920x1080",
  "duration": 6.0,
  "fps": 24,
  "generate_audio": false,
  "lora": null,
  "enhance_prompt": false
}
```

With keyframes:

```json
{
  "prompt": "walk cycle",
  "keyframes": [
    {"image_uri": "storage://a", "frame_index": "first", "strength": 1.0},
    {"image_uri": "storage://b", "frame_index": "middle", "strength": 0.5},
    {"image_uri": "storage://c", "frame_index": "last", "strength": 1.0}
  ],
  "model": "ltx-2-3-pro",
  "resolution": "1920x1080",
  "duration": 6.0,
  "fps": 24
}
```

**Response:** `200 video/mp4`.

### `POST /v1/audio-to-video`

Generates video timed to an uploaded audio track. Optional conditioning image.

```json
{
  "prompt": "lip-synced interview, warm studio light",
  "audio_uri": "storage://...",
  "image_uri": "storage://...",
  "model": "ltx-2-3-fast",
  "resolution": "1080x1920",
  "duration": 6.0,
  "fps": 24
}
```

**Response:** `200 video/mp4`.

### `POST /v1/retake`

Regenerates a span of an existing video.

```json
{
  "video_uri": "storage://...",
  "start_time": 1.5,
  "duration": 3.0,
  "mode": "replace_video",
  "prompt": "make the sky overcast",
  "lora": null
}
```

`mode` ∈ `"replace_audio_and_video" | "replace_video" | "replace_video_only" | "replace_audio"`.

**Response:** `200 video/mp4`. On any failure: `422 Content rejected or generation failed: …`.

### `POST /v1/text-to-image`

```json
{
  "prompt": "cinematic portrait of a woman in a red coat, overcast afternoon",
  "model": "flux2-dev",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 50,
  "guidance_scale": 4.0,
  "seed": null,
  "turbo": false,
  "lora": {"id": "mystyle", "strength": 0.8}
}
```

- `seed: null` → server picks a random 32-bit seed.
- `turbo: true` → server forces 8 steps + turbo sigma schedule + guidance 2.5.
- Width/height snapped to multiples of 16.

**Response:** `200 image/webp` (lossless VP8L).

### `POST /v1/image-to-image`

Same fields as `text-to-image` plus required `image_uri`. Response `200 image/webp`.

### `POST /v1/image-edit`

Multi-image edit (Klein KV by default).

```json
{
  "prompt": "replace the background with a rainy street",
  "image_uris": ["storage://a"],
  "model": "flux2-klein",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 4,
  "guidance_scale": 4.0,
  "lora": null
}
```

`image_uris`: 1–10 entries. Response `200 image/webp`.

---

## v2 async generation + jobs

v2 endpoints return `202 Accepted` with a `job_id`. Poll `/v2/jobs/{job_id}` until `status == "completed"`, then GET the result URL.

### Common submission response

```json
{
  "job_id": "job_01H...",
  "status": "queued",
  "poll_url": "/v2/jobs/job_01H...",
  "stream_url": "/v2/jobs/job_01H.../stream"
}
```

On pause: `503 {"error": "system_paused"}` + `Retry-After: 300`.
On full queue: `429 {"error": "queue_full"}` + `Retry-After: 30`.

### Endpoints

All accept the **same bodies** as their v1 counterparts:

| Endpoint | Request shape |
|---|---|
| `POST /v2/text-to-video` | `TextToVideoRequest` |
| `POST /v2/image-to-video` | `ImageToVideoRequest` |
| `POST /v2/audio-to-video` | `AudioToVideoRequest` |
| `POST /v2/retake` | `RetakeRequest` |
| `POST /v2/text-to-image` | `TextToImageRequest` |
| `POST /v2/image-to-image` | `ImageToImageRequest` |
| `POST /v2/image-edit` | `ImageEditRequest` |

All return `202` + submission envelope.

### `GET /v2/jobs/{job_id}` — status poll

```json
{
  "job_id": "job_...",
  "status": "queued" | "processing" | "completed" | "failed" | "cancelled",
  "type": "text-to-video" | ...,
  "progress": 0.42,
  "phase": "denoising" | "decoding" | "encoding" | "saving" | null,
  "queue_position": 3,
  "error": {"code": "generation_failed", "message": "..."} | null,
  "result_url": "/v2/jobs/job_.../result" | null,
  "result_storage_uri": "storage://..." | null,
  "result_media_type": "video/mp4" | "image/webp" | null
}
```

- `progress` is only populated while `processing` (range `0.0–1.0`). Denoising reports up to `0.90`; the top 10% is reserved for post-denoise phases. Completed jobs report `1.0`.
- `phase` is only populated while `processing`. Typical sequence: `denoising` → `decoding` (LTX only, VAE decode) → `encoding` (ffmpeg or WEBP) → `saving` (upload-store write). Use this to render "Decoding video…" / "Encoding MP4…" / etc. instead of a frozen percentage during the silent post-denoise tail.
- `queue_position` is only populated while `queued`.
- `result_url` / `result_storage_uri` / `result_media_type` are only populated when `completed`.
- `404` if the job id is unknown or expired (job store TTL applies).

### `GET /v2/jobs/{job_id}/preview`

Returns a low-res preview JPEG at three possible states:

| State | Response |
|---|---|
| Flux step-end callback already cached a preview | `200 image/jpeg` (live as the image denoises) |
| Completed video job — first frame extracted lazily via PyAV | `200 image/jpeg` (cached on first call) |
| Queued / processing video / no data yet | **`204 No Content`** — **not** `404`. Keep polling. |
| Unknown job id | `404 Job not found` |

**Client note:** treat `204` as "no preview yet, poll again". Historic clients that treated `404` as "broken job" will need to switch to `204`.

### `GET /v2/jobs/{job_id}/result`

- `200` with `FileResponse` (streaming) + `Cache-Control: no-store` when complete.
- `409 Job result not ready` if the job isn't `completed`.
- `404 Job not found` / `404 Result file expired or not found`.

### `DELETE /v2/jobs/{job_id}` — cancel

- `200 {"job_id": "...", "status": "cancelled"}` if queued or processing.
- `409 Cannot cancel a finished job` if already completed / failed / cancelled.
- `404 Job not found`.

### `GET /v2/jobs/{job_id}/stream` — SSE live updates

Server-Sent Events stream for live job state. **Use this instead of polling `/v2/jobs/{id}`** — one long-lived connection replaces the entire poll loop.

**Auth**: `EventSource` can't set custom headers, so the endpoint accepts either a bearer `Authorization` header (for programmatic clients) or a `?token=<sse-token>` query param (for browsers). Get a short-lived token via `POST /v1/sse-token`.

**Event shape**: each `data:` line contains the same JSON as `GET /v2/jobs/{id}` (status, progress, phase, queue_position, error, result_url, …). Drop it into your existing polling parser as-is.

**Delivery semantics**:
- Emits one event immediately on connect with the current state.
- Emits again every time `(status, progress, phase, error_code)` changes. Progress is rounded to 3 decimals to avoid flooding on micro-ticks.
- Emits a `: keepalive` comment every 15 s during idle periods (queued with no position change) to prevent intermediate proxies from closing the connection.
- Emits one final event on terminal state (completed / failed / cancelled), then closes the stream.
- Emits `event: error` with `{"error": "job_expired"}` and closes if the job is evicted from the store mid-stream.

**Status codes**:
- `200 text/event-stream` — stream opened.
- `404 Job not found` — unknown job id (before stream opens).
- `401 Missing API key` — neither bearer nor valid token.

**Browser example**:

```js
// 1. Submit the job (normal POST with bearer)
const { job_id } = await fetch("/v2/text-to-video", { ... }).then(r => r.json());

// 2. Get a disposable SSE token
const { token } = await fetch("/v1/sse-token", { method: "POST", headers: { Authorization: `Bearer ${KEY}` } }).then(r => r.json());

// 3. Open the live stream
const es = new EventSource(`/v2/jobs/${job_id}/stream?token=${token}`);
es.onmessage = (ev) => {
  const { progress, phase, status, result_url } = JSON.parse(ev.data);
  setProgress(progress);
  setPhase(phase);
  if (status === "completed") {
    fetch(result_url, { headers: { Authorization: `Bearer ${KEY}` } })
      .then(r => r.blob())
      .then(showResult);
    es.close();
  } else if (status === "failed" || status === "cancelled") {
    es.close();
  }
};
es.addEventListener("error", (ev) => {
  // connection dropped OR server emitted `event: error` — retry or fall back to polling
});
```

**curl example**:

```bash
curl -N -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JID/stream"
```

---

## Uploads

Two-step: get a presigned-style URL, then PUT the bytes.

### `POST /v1/upload`

```json
{
  "upload_url": "http://host:8090/uploads/put/<upload_id>",
  "storage_uri": "storage://<upload_id>",
  "required_headers": {}
}
```

Use the returned `storage_uri` in any generation request that takes an `image_uri`, `audio_uri`, `video_uri`, or `keyframes[].image_uri`.

### `PUT /uploads/put/{upload_id}`

- Body: raw file bytes. No multipart wrapping.
- Max size: `upload_store.MAX_UPLOAD_BYTES`. Over → `413`.
- Success: `201 Created` (empty body).

Upload URLs are not signed — they rely on bearer auth plus the unguessable UUID. Treat `storage://` URIs as capabilities.

---

## LoRAs (LTX)

Full-fat registry with upload + delete. Stored in `LORAS_DIR/<filename>.safetensors` with `registry.json` metadata.

### `GET /v1/loras`

```json
{
  "loras": [
    {
      "id": "my-style",
      "name": "My Style",
      "filename": "my_style.safetensors",
      "base_model": "ltx-2.3",
      "size_bytes": 123456789,
      "uploaded_at": 1712345678.9,
      "description": "...",
      "trigger_word": "mystyle",
      "strategy": "full"
    }
  ],
  "count": 1
}
```

### `POST /v1/loras` — `201`

Multipart form. Fields:

| Field | Required | Notes |
|---|---|---|
| `file` | yes | `.safetensors` file |
| `name` | yes | Human-readable |
| `description` | no | |
| `base_model` | no | Default `ltx-2.3` |
| `trigger_word` | no | |
| `strategy` | no | |

- `400 Expected multipart/form-data` / `400 Missing 'file' field` / `400 File must be a .safetensors file`
- `422 Missing 'name' field`
- `413` if over `config.MAX_LORA_SIZE_BYTES`
- `400` with a validation error text if the registry rejects it (e.g., duplicate id)

Returns the same row shape as `GET /v1/loras`.

### `DELETE /v1/loras/{lora_id}`

- `200 {"deleted": true, "id": "<id>"}` on success
- `404 LoRA not found: <id>` otherwise

---

## Flux LoRAs

**Folder-drop model** — no upload endpoint. Files live under `FLUX_LORAS_DIR/`. The registry scans for `.safetensors`, using the slugified filename stem as the id. Optional sidecar `<stem>.json` adds metadata. Manage with `cp` / `rm` on the host, then `POST /v1/flux-loras/rescan`.

### `GET /v1/flux-loras`

```json
{
  "loras": [
    {
      "id": "cinematic-portrait",
      "name": "Cinematic Portrait",
      "filename": "cinematic_portrait.safetensors",
      "size_bytes": 123456789,
      "model_compat": ["flux2-dev", "flux2-klein"],
      "description": "...",
      "trigger_word": "cinematic"
    }
  ],
  "count": 1
}
```

### `POST /v1/flux-loras/rescan`

Re-walks `FLUX_LORAS_DIR`. Returns `{"rescanned": true, "count": N}`.

---

## Chat completions

### `POST /v1/chat/completions`

Thin proxy to the external llama-swap server — the shape is OpenAI-compatible.

```json
{
  "model": "gemma-3-12b-nvfp4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

`content` can be a string (plain text) or an OpenAI multimodal list (`[{"type": "text", ...}, {"type": "image_url", ...}]`).

Response: raw OpenAI chat-completion JSON, passed through from the upstream.

- `500 Chat model not loaded` if the proxy isn't configured.
- `422 Messages list cannot be empty`.

---

## SSE session tokens

### `POST /v1/sse-token`

EventSource can't send custom headers, so SSE endpoints accept a short-lived query-param token instead of a bearer header.

```json
{"token": "<32 url-safe bytes>", "expires_in": 300}
```

- `401 Missing API key` if no valid bearer token in the request.
- Tokens are single-use in the sense that they expire after 5 minutes. No refresh.
- Pass to SSE as `?token=...`.

---

## Approved images

A per-API-key "approved feed" — noodle-i approves an image, noodle-v watches the feed.

### `POST /v1/approved-images` — `201`

```json
{
  "image_uri": "storage://...",
  "prompt": "...",
  "model": "flux2-dev",
  "width": 1024,
  "height": 1024
}
```

Response: `{"id": "<16 hex>", "status": "approved"}`. `401` if no api key, `400` if `image_uri` is missing.

### `GET /v1/approved-images?limit=50&offset=0`

Returns a JSON array of entries (per-api-key scoped):

```json
[
  {
    "id": "...",
    "image_uri": "storage://...",
    "prompt": "...",
    "model": "flux2-dev",
    "width": 1024,
    "height": 1024,
    "created_at": 1712345678.9,
    "image_url": "/v1/approved-images/<id>/file"
  }
]
```

`limit` is clamped elsewhere; default 50. `api_key_hash` is stripped from responses.

### `GET /v1/approved-images/events` — SSE (no bearer required)

Accepts `?token=<sse-token>` OR a bearer header. Streams newly-added entries as they land:

```
data: {"id": "...", "image_uri": "...", "image_url": "/v1/approved-images/.../file", ...}

```

Heartbeat: polls the manifest every 2 s via `mtime`. Closes on client disconnect. `401` if neither token nor bearer resolves.

### `GET /v1/approved-images/{image_id}/file`

Returns `image/webp` (or the original file's mime if different). `404` if the manifest entry doesn't exist for this api key or the file has been evicted.

---

## Character-consistency vision ranking

### `POST /v2/char/rank`

Routes a two-image comparison to the Gemma 4 31B vision model on llama-swap. Used by noodle-i's character mode to score generated outputs against a reference.

```json
{
  "rank_image_uri": "storage://ref",
  "generated_image_uri": "storage://gen",
  "prompt": "The original generation prompt"
}
```

Response: a strict-JSON body produced by the vision model:

```json
{
  "score": 8.25,
  "analysis": {
    "face_match": 9,
    "eyes": 8,
    "proportions": 8,
    "overall_likeness": 8
  },
  "edits": {
    "add": ["slightly narrower jawline"],
    "remove": [],
    "modify": {}
  }
}
```

- `404` if either storage URI can't be resolved.
- `500 Chat model not loaded` if the llama-swap proxy isn't ready.
- `500 Vision model did not return valid JSON` if the model output can't be parsed.

---

## Generation history

Per-api-key history of completed v2 jobs, with thumbnails, keyed by SHA-256 of the api key. 30-day retention.

### `GET /v2/history?limit=50&offset=0&type=<filter>`

`type` can be:

- `"image"` — any `*-image*` job type
- `"video"` — any `*-video*` job type
- A specific job type string (e.g., `"text-to-video"`)
- Unset — all types

Response is a JSON array; `limit` is clamped to `200`.

```json
[
  {
    "id": "job_...",
    "prompt": "...",
    "model": "ltx-2-3-pro",
    "width": 1920,
    "height": 1080,
    "turbo": false,
    "status": "completed",
    "created_at": 1712345678.9,
    "error": null,
    "thumbnail_url": "/v2/history/job_.../thumbnail",
    "image_url": "/v2/history/job_.../image"
  }
]
```

### `GET /v2/history/{generation_id}/image`

Returns the full-size generation. Media type is `video/mp4` for video jobs, `image/webp` otherwise. `404` if the entry or file is missing.

### `GET /v2/history/{generation_id}/thumbnail`

Returns `image/jpeg` (256 px wide). For video jobs, the thumbnail is the first frame extracted via PyAV. `404` if no thumbnail was produced (very old entries, or decode failures).

---

## Compositions

Multi-clip composition timelines (noodle-v's export pipeline).

### `POST /v2/compositions` — `201`

```json
{
  "name": "My Cut",
  "clips": [...],
  "transitions": [...]
}
```

Returns the created row (shape owned by `composition_store`).

### `GET /v2/compositions?limit=50&offset=0`

Returns an array of the caller's compositions (clamped to 200).

### `GET /v2/compositions/{comp_id}`

Returns the full composition row. `404` if not found or owned by another key.

### `PUT /v2/compositions/{comp_id}`

Body: same as create. Response: `{"status": "updated"}` or `404`.

### `DELETE /v2/compositions/{comp_id}`

Response: `{"status": "deleted"}` or `404`.

### `POST /v2/compositions/{comp_id}/export`

Enqueues an `EXPORT_COMPOSITION` job. Returns the same `202` envelope as any v2 submit. Poll `/v2/jobs/{job_id}` for progress.

---

## Error codes

Codes the backend actively returns:

| Status | Meaning |
|---|---|
| `200` | OK (sync generation returns binary; other responses are JSON) |
| `201` | Resource created (LoRA upload, upload PUT, approved image) |
| `202` | v2 job queued |
| `204` | `/v2/jobs/{id}/preview` has nothing yet — poll again |
| `400` | Malformed request (multipart / missing field / bad content-type) |
| `401` | Missing or invalid bearer token |
| `404` | Unknown id, storage URI, file, composition, LoRA, history entry |
| `409` | `v2/jobs/{id}/result` not ready; cancel on finished job |
| `413` | Upload or LoRA over size cap |
| `422` | Pydantic validation, keyframe bounds, LoRA id mismatch, retake content rejection |
| `429` | Queue full (`queue_full`, `Retry-After: 30`) |
| `500` | Unhandled internal error, Flux disabled, chat proxy not ready |
| `503` | Paused (`Retry-After: 300`) |

---

## Endpoint index

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | no | Liveness + model status |
| POST | `/v1/system/pause` | yes | Evict all models + cancel queued jobs |
| POST | `/v1/system/resume` | yes | Reload all models |
| POST | `/v1/flux/unload` | yes | Unload Flux only |
| POST | `/v1/flux/reload` | yes | Reload Flux only |
| POST | `/v1/ltx/unload` | yes | Unload LTX only |
| POST | `/v1/ltx/reload` | yes | Reload LTX only |
| POST | `/v1/text-to-video` | yes | Sync video gen |
| POST | `/v1/image-to-video` | yes | Sync i2v / keyframe gen |
| POST | `/v1/audio-to-video` | yes | Sync a2v |
| POST | `/v1/retake` | yes | Sync retake |
| POST | `/v1/text-to-image` | yes | Sync Flux t2i |
| POST | `/v1/image-to-image` | yes | Sync Flux i2i |
| POST | `/v1/image-edit` | yes | Sync Flux multi-image edit |
| POST | `/v1/chat/completions` | yes | Chat proxy |
| POST | `/v1/upload` | yes | Get upload URL |
| PUT | `/uploads/put/{upload_id}` | yes | Upload bytes |
| GET | `/v1/loras` | yes | List LTX LoRAs |
| POST | `/v1/loras` | yes | Upload LTX LoRA (multipart) |
| DELETE | `/v1/loras/{lora_id}` | yes | Delete LTX LoRA |
| GET | `/v1/flux-loras` | yes | List Flux LoRAs |
| POST | `/v1/flux-loras/rescan` | yes | Re-scan Flux LoRA folder |
| POST | `/v2/text-to-video` | yes | Async video gen |
| POST | `/v2/image-to-video` | yes | Async i2v |
| POST | `/v2/audio-to-video` | yes | Async a2v |
| POST | `/v2/retake` | yes | Async retake |
| POST | `/v2/text-to-image` | yes | Async Flux t2i |
| POST | `/v2/image-to-image` | yes | Async Flux i2i |
| POST | `/v2/image-edit` | yes | Async Flux edit |
| GET | `/v2/jobs/{job_id}` | yes | Poll job status |
| GET | `/v2/jobs/{job_id}/preview` | yes | Preview JPEG (204 when empty) |
| GET | `/v2/jobs/{job_id}/result` | yes | Download final media |
| GET | `/v2/jobs/{job_id}/stream` | yes (bearer OR token) | SSE live status/progress/phase stream |
| DELETE | `/v2/jobs/{job_id}` | yes | Cancel job |
| POST | `/v1/sse-token` | yes | Issue 5-min SSE token |
| POST | `/v1/approved-images` | yes | Approve an image |
| GET | `/v1/approved-images` | yes | List approved images |
| GET | `/v1/approved-images/events` | no (token or bearer) | SSE feed |
| GET | `/v1/approved-images/{id}/file` | yes | Fetch approved image file |
| POST | `/v2/char/rank` | yes | Vision character consistency rank |
| GET | `/v2/history` | yes | Per-key generation history |
| GET | `/v2/history/{id}/image` | yes | Full-size history media |
| GET | `/v2/history/{id}/thumbnail` | yes | History thumbnail |
| POST | `/v2/compositions` | yes | Create composition |
| GET | `/v2/compositions` | yes | List compositions |
| GET | `/v2/compositions/{id}` | yes | Get composition |
| PUT | `/v2/compositions/{id}` | yes | Update composition |
| DELETE | `/v2/compositions/{id}` | yes | Delete composition |
| POST | `/v2/compositions/{id}/export` | yes | Enqueue composition export job |

Total: 48 routes (47 HTTP handlers + `/health`).

---

## Curl examples

**Upload + sync text-to-image:**

```bash
API="http://localhost:8090"
KEY="your-key"

# 1. Get an upload slot (only needed for i2i/i2v)
curl -X POST "$API/v1/upload" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d '{}'

# 2. Straight text-to-image
curl -X POST "$API/v1/text-to-image" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"cinematic portrait","model":"flux2-dev","width":1024,"height":1024}' \
  --output out.webp
```

**Async video job → poll → fetch:**

```bash
JOB=$(curl -s -X POST "$API/v2/text-to-video" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"a cat","model":"ltx-2-3-fast","resolution":"1920x1080","duration":4,"fps":24}' \
  | jq -r '.job_id')

# Poll
while true; do
  STATUS=$(curl -s -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JOB" | jq -r '.status')
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && echo "failed" && exit 1
  sleep 2
done

# Fetch
curl -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JOB/result" --output out.mp4
```

---

## Changelog

- **v1.1.7** (2026-04-11)
  - `GET /v2/jobs/{id}/stream` — SSE endpoint that was previously advertised in the submission envelope but never implemented. One long-lived connection replaces the 240-GET polling loop per video job. Emits on state change + keepalive every 15 s. Accepts bearer header or `?token=` query param (for browser `EventSource`).
- **v1.1.6** (2026-04-11)
  - `/v2/jobs/{id}` status: new `phase` field ("denoising" / "decoding" / "encoding" / "saving" / null) so clients can render labels during the post-denoise tail instead of a frozen percentage. Denoising now reports progress up to `0.90` (was `0.99`); the top 10 % maps to the post-denoise phases.
  - `/v2/jobs/{id}/preview`: reuses the on-disk thumbnail produced by `history.save()` via zero-copy `FileResponse`. Fallback lazy extraction still exists for jobs without api_key but is now offloaded via `asyncio.to_thread` so the event loop is never blocked.
  - `history.save()` now runs in a background `asyncio.to_thread` task instead of on the queue worker's event loop. The worker dequeues the next job immediately; the previous ~300 ms thumbnail window no longer stalls the queue.
  - SQLite history DB switched to WAL mode — readers no longer block behind the single writer. `/v2/history` list reads run concurrently with the worker's save.
- **v1.1.5** (2026-04-09)
  - `/v2/jobs/{id}/preview`: returns `204` (not `404`) while no preview is available.
  - Lazy first-frame extraction for completed video jobs via PyAV; cached on job.
  - `history_store` now generates thumbnails for video jobs (was silently failing on `PIL.UnidentifiedImageError`).
- **v1.1.4** (2026-04-09) — Single-GPU swap mode
  - `LTX_DEVICE = FLUX_DEVICE = "cuda:0"`. `cuda:1` reserved for external training.
  - `evict_all()` leak fix: cleared `worker._model_ledger` + `_source_ledger`. Reclaims 99 % of LTX VRAM on unload (verified 66.9 GB → 683 MiB).
  - Added `/v1/ltx/{unload,reload}` endpoints mirroring the existing Flux pair.
  - Auto-swap helpers `_ensure_ltx_resident` / `_ensure_flux_ready` wired into `_dispatch_job` and every v1 sync handler.
- **v1.1.3** — VAE force-upcast fix (pre-hook + fp32 params).
- **v1.1.2** — Lossless VP8L WEBP for Flux output (fixed "screendoor" artifact).
- **v1.1.1** — Dropped FP8 layerwise casting on Flux 2 Dev; adapter-mode LoRA (strength changes are free).
- **v1.1** — Flux LoRA folder-drop; first/middle/last keyframes; char rank; gen history.
