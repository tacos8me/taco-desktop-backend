# taco-backend — Frontend Quickstart

Everything you need to ship a working client in 5 minutes. For everything else → [docs/API.md](./API.md).

## What's new in v1.10.0 (2026-04-20)

- **Seamless MusicVideo chain conditioning** (v1.10.0) — non-first clips are conditioned on the last 3 safe frames of their predecessor so motion and visual state flow continuously across cuts. Composition export trims the duplicated tail so the 3 chain frames appear exactly once. Result: no visible seam at clip boundaries. Full FE orchestration spec in [docs/handover-frontend-v1.10-chain.md](./handover-frontend-v1.10-chain.md).
- **`POST /v2/video/extract-frames`** (v1.10.0) — server-side PyAV helper. Body `{video_uri, frame_indices: [int]}` (1–16, sorted+deduped). Returns lossless PNGs as `storage://` URIs. Bearer + capability-URL security. Bounded concurrency (semaphore 2 + 30 s timeout). Output bytes count against `PER_KEY_UPLOAD_BYTES_PER_DAY`.
- **`AudioToVideoRequest.keyframes`** (v1.10.0) — a2v now accepts the same `keyframes: list[KeyframeInput]` shape as i2v (mutually exclusive with `image_uri`+`image_strength`; 422 on conflict). Legacy single-keyframe path unchanged.
- **`tailTrimFrames: int` per-clip composition field** (v1.10.0, default 0) — backend trims the last N frames of each input at export time. Cascades into beat-gap atrim + force-IDR seam math.
- **`GET /uploads/get/{upload_id}`** (v1.9.1) — read back a previously-uploaded file. Content-Type inferred from magic bytes.
- **Composition audio overlay** (v1.9.0) — `POST /v2/compositions/{id}/export` accepts optional `{"audio_uri": "storage://<id>"}`.
- **Multi-provider remote pool** (v1.9.0) — up to 2 local + 4 Modal + 2 RunPod = **8 concurrent video jobs** at peak.
- **Video outpaint** (v1.7.0) — `POST /v2/video-outpaint`. See [docs/outpaint-frontend-guide.md](./outpaint-frontend-guide.md).
- **Turbo hardening** (v1.5) — `systemctl`-based cuda:1 tenant eviction on entry, 20 s drain deadline with automatic rollback on failure.
- All changes additive. Legacy env vars (`LTX_REMOTE_SIDECAR_URL`), legacy pool body shape (`{"count": N}`), and the video-only export call keep working unchanged.

## The 60-second version

```
Public base URL: https://api.noodlefinger.io        (canonical, Cloudflare-proxied)
LAN / dev URL:   http://<host>:8090                 (uvicorn direct, no CF)
Auth:            Authorization: Bearer <api-key>    (required on every request)
Pattern:         POST /v2/<op>  →  SSE /v2/jobs/<id>/stream  →  GET /v2/jobs/<id>/result
```

> `taco.noodlefinger.io` was retired 2026-04-18 and no longer resolves.

## The golden path

For any generation (video or image):

1. **Submit** — `POST /v2/text-to-video` (or `/v2/text-to-image`, etc.) → returns `{ "job_id": "...", "stream_url": "/v2/jobs/.../stream" }` with HTTP 202.
2. **Stream** — Open `GET /v2/jobs/{id}/stream` as an EventSource. Receive progress+phase updates in real time.
3. **Fetch** — When the stream event says `"status": "completed"`, GET `/v2/jobs/{id}/result` for raw MP4 or WEBP bytes.

**Prefer the SSE stream over polling `/v2/jobs/{id}`.** One long-lived connection replaces 240+ GETs per video job. Fall back to polling every 1–2 s only if your client can't do `EventSource`.

## Full browser example (image generation with phase UI)

```js
// In prod, use the public base:   const API = "https://api.noodlefinger.io";
// For local dev against uvicorn:
const API = "http://localhost:8090";
const KEY = "your-api-key";

// 1. Submit the job
const job = await fetch(`${API}/v2/text-to-image`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "cinematic portrait of a woman in a red coat",
    model: "flux2-dev",
    width: 1024,
    height: 1024,
  }),
}).then(r => r.json());

// 2. Get a short-lived SSE token (EventSource can't set custom headers)
const { token } = await fetch(`${API}/v1/sse-token`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${KEY}` },
}).then(r => r.json());

// 3. Stream live progress + phase
const es = new EventSource(`${API}/v2/jobs/${job.job_id}/stream?token=${token}`);
es.onmessage = async (ev) => {
  const { status, progress, phase, result_url } = JSON.parse(ev.data);
  ui.setProgress(progress);      // 0.0 → 1.0
  ui.setPhase(phase);            // "denoising" | "decoding" | "encoding" | "saving" | null

  if (status === "completed") {
    const blob = await fetch(`${API}${result_url}`, {
      headers: { "Authorization": `Bearer ${KEY}` },
    }).then(r => r.blob());
    ui.showImage(URL.createObjectURL(blob));
    es.close();
  } else if (status === "failed" || status === "cancelled") {
    ui.showError(status);
    es.close();
  }
};
```

## Progress + phase: how to render it

Denoising reports progress up to **0.90**; the top 10% of the bar covers post-denoise work. Render the `phase` label instead of a frozen percentage once `progress >= 0.90`.

| `progress` range | `phase` | UI label |
|---|---|---|
| 0.00–0.90 | `"denoising"` | "Generating… {round(progress*100)}%" |
| 0.90 | `"decoding"` *(video only)* | "Decoding video frames…" |
| 0.95 | `"encoding"` | "Encoding output…" |
| 0.99 | `"saving"` | "Finalizing…" |
| 1.00 | `null` (status = completed) | done — fetch `result_url` |

`phase` is only populated while `status` is `processing`. For `queued`/`completed`/`failed`/`cancelled`, it's `null`.

## Upload flow (for any input file)

Two steps for anything that needs an image, video, or audio input:

```js
// 1. Ask for an upload slot
const { upload_url, storage_uri } = await fetch(`${API}/v1/upload`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${KEY}` },
}).then(r => r.json());

// 2. PUT the raw bytes
await fetch(upload_url, {
  method: "PUT",
  body: fileBlob,
  headers: { "Authorization": `Bearer ${KEY}` },
});

// 3. Reference by the returned storage_uri in your generation request
await fetch(`${API}/v2/image-to-video`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({
    image_uri: storage_uri,     // "storage://abc123..."
    prompt: "she turns toward the camera",
    model: "ltx-2-3-pro",
    resolution: "1920x1080",
    duration: 6.0,
    fps: 24,
  }),
});
```

- Max upload size: **1 GB**
- Storage URIs look like `storage://<uuid>` — treat them as capabilities
- Uploads expire with their referencing job (don't hang onto `storage_uri`s forever)

## Models at a glance

### Video — `/v2/text-to-video`, `/v2/image-to-video`, `/v2/audio-to-video`, `/v2/retake`, `/v2/video-outpaint`

| Model | Latency (~5s @ 1080p) | Use case |
|---|---|---|
| `ltx-2-3-fast` | ~15 s | Previews, fast iteration, audio-to-video |
| `ltx-2-3-pro` | ~65 s | Production default |
| `ltx-2-3-hq` | ~90 s | Final render (res2s + CFG++ sampler, best motion) |
| `ic-lora-outpaint` via `/v2/video-outpaint` | ~35-45 s full, ~20 s `skip_stage_2` | Canvas expansion — see [outpaint-frontend-guide.md](./outpaint-frontend-guide.md) |

### Image — `/v2/text-to-image`, `/v2/image-to-image`, `/v2/image-edit`

| Model | Latency (1024×1024) | Use case |
|---|---|---|
| `flux2-dev` + `turbo: true` | ~15 s (8 steps) | Fast preview, iteration |
| `flux2-dev` | ~60 s (50 steps) | Highest quality |
| `flux2-klein` | ~10 s (4 steps) | Multi-reference editing, multi-subject consistency |
| `joyai-edit` | ~78 s (30 steps) | **Instruction-based single-image edits** (remove objects, move things, camera moves). Sidecar-hosted. |

### Music — `/v1/music`, `/v2/music` (v1.2)

| Model | Latency | Use case |
|---|---|---|
| ACE xl-base + LM | ~2–10 s | Text-to-music, covers, repainting, stem extraction. Concurrent with video on cuda:1. |

### Chat / vision
- `POST /v1/chat/completions` — OpenAI-compatible proxy (llama-swap)
- `POST /v2/char/rank` — character-consistency scorer (reference vs generated → JSON with `face_match`, `eyes`, `proportions`, `overall_likeness`, suggested edits)

## Request shape cheatsheet

### `POST /v2/text-to-video`
```json
{
  "prompt": "a cat walks across a sunlit hardwood floor",
  "model": "ltx-2-3-pro",
  "resolution": "1920x1080",
  "duration": 5.0,
  "fps": 24,
  "generate_audio": false,
  "camera_motion": "slow dolly in",
  "enhance_prompt": false
}
```
Valid `resolution`: `1920x1080`, `1080x1920`, `2560x1440`, `1440x2560`, `3840x2160`, `2160x3840`.

### `POST /v2/text-to-image`
```json
{
  "prompt": "cinematic portrait",
  "model": "flux2-dev",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 50,
  "guidance_scale": 4.0,
  "turbo": false,
  "lora": {"id": "mystyle", "strength": 0.8}
}
```

### `POST /v2/image-to-video` with keyframes
```json
{
  "prompt": "walk cycle",
  "keyframes": [
    {"image_uri": "storage://a", "frame_index": "first",  "strength": 1.0},
    {"image_uri": "storage://b", "frame_index": "middle", "strength": 0.5},
    {"image_uri": "storage://c", "frame_index": "last",   "strength": 1.0}
  ],
  "model": "ltx-2-3-pro",
  "resolution": "1920x1080",
  "duration": 6.0,
  "fps": 24
}
```
`frame_index` accepts `int | "first" | "middle" | "last"` plus negative ints (Python-style: `-1` = last, `-12` = 12 frames before end). Up to 8 keyframes. Mutually exclusive with `image_uri`.

### `POST /v2/video-outpaint` (v1.7.0)
```json
{
  "video_uri": "storage://abc123",
  "prompt": "extend the scene naturally, matching lighting and style",
  "target_resolution": "1920x1080",
  "position": "center",
  "duration": 5.0,
  "fps": 24,
  "seed": 42,
  "conditioning_strength": 1.0,
  "skip_stage_2": false
}
```
- Source is letterboxed into `target_resolution` with pure black; IC-LoRA fills the black.
- `position` ∈ `"center" | "left" | "right" | "top" | "bottom" | "top_left" | "top_right" | "bottom_left" | "bottom_right"`.
- `lora` defaults to `{"id": "ic-lora-outpaint", "strength": 1.0}`. Override for custom outpaint LoRAs.
- Output is **silent MP4** (no audio passthrough in v1.7.0). Full integration walkthrough in [docs/outpaint-frontend-guide.md](./outpaint-frontend-guide.md).

### `POST /v2/image-edit` (Flux Klein multi-reference)
```json
{
  "prompt": "replace the background with a rainy street",
  "image_uris": ["storage://a", "storage://b"],
  "model": "flux2-klein",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 4
}
```
`image_uris`: 1–10 entries. Klein ignores `guidance_scale` — the server strips it silently.

### `POST /v2/image-edit` (JoyAI instruction-based, v1.1.8)
```json
{
  "prompt": "Remove the construction crane from the top of the building.",
  "image_uris": ["storage://abc123"],
  "model": "joyai-edit",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 30,
  "guidance_scale": 4.0
}
```
- **Exactly one** `image_uris` entry — multi-reference is rejected with `422`.
- **LoRA is not supported** — requests with the `lora` field return `422`.
- The server wraps the prompt in a chat template (`<|im_start|>user\n<image>\n{prompt}<|im_end|>\n`) before calling the sidecar — send plain English.
- Phase stays on `"encoding"` for the entire ~78 s sidecar call. Render a spinner, not a moving percentage.
- Requires `LOAD_JOYAI=1` on the server. If unset / sidecar down: `503 joyai_disabled` or `503 sidecar_unreachable` — fall back to `flux2-klein`.

### `POST /v2/music` (v1.2)
```json
{
  "prompt": "upbeat electronic dance track with heavy bass",
  "lyrics": "[Instrumental]",
  "duration": 60.0,
  "audio_format": "mp3",
  "task_type": "text2music",
  "num_inference_steps": 50,
  "guidance_scale": 7.0,
  "bpm": 128
}
```
- Phase for music jobs is `"generating"` (not `"denoising"`).
- Requires `LOAD_ACE=1`. Returns `503` when disabled or during turbo mode.
- Music queue cap: 5. Returns `429 music_queue_full` when exceeded.

### `POST /v2/batch` (v1.2)
```json
{
  "items": [
    {"type": "text-to-image", "params": {"prompt": "a cat", "model": "flux2-klein", "width": 1024, "height": 1024, "num_inference_steps": 4}},
    {"type": "text-to-video", "params": {"prompt": "a dog walking", "model": "ltx-2-3-fast", "resolution": "1920x1080", "duration": 5, "fps": 24}}
  ],
  "priority": "normal"
}
```
- Returns `202` with `batch_id`. Poll `GET /v2/batch/{batch_id}` for status + partial results. `DELETE /v2/batch/{batch_id}` to cancel.
- Download individual results: `GET /v2/batch/{batch_id}/result/{index}` (0-based index).
- Items sorted images-first to minimize GPU swaps. In turbo mode, 2 video items process concurrently.

## Common pitfalls

- **Image dims must be multiples of 16** (width/height). Video dims must be multiples of 64. The server floors silently — you'll lose pixels if you forget.
- **Video frame count is `8k + 1`** — derived from `duration × fps`. Actual frame count can drift ±4 frames.
- **Klein ignores `guidance_scale`** (distilled, no CFG). Include it or omit it, same result.
- **LoRA strength changes are free** on Flux — same endpoint, different `strength`, no reload.
- **LoRA file changes are NOT free** — ~30 s Dev reload, ~5 s Klein reload.
- **First request after a Flux↔LTX swap** pays 3–30 s extra. Subsequent same-type requests stay fast.
- **`/v2/jobs/{id}/preview` returns `204`** (not `404`) when no preview is ready yet. Keep polling.
- **SSE tokens expire after 5 minutes.** Issue a new one if your stream disconnects.
- **Max queue depth is 10.** `429 queue_full` with `Retry-After: 30` when exceeded.
- **Auth is enforced on every endpoint** except `/health` and `/v1/approved-images/events`. Don't forget the bearer.
- **`joyai-edit` requires exactly one `image_uri`** (not 1–10 like flux2-klein) — `422` otherwise.
- **`joyai-edit` stays on phase `"encoding"` for the entire ~78 s sidecar call** — render it as a spinner, not a frozen percentage. Progress sits at `0.90` the whole time.
- **Turbo mode** (`POST /v1/system/turbo`) claims both GPUs for LTX — Flux, ACE (music), and JoyAI all return `503` while active. 2x video throughput.
- **Music jobs use phase `"generating"`** (not `"denoising"`). The ACE sidecar is opaque to per-step callbacks.

## Error envelope

All error responses have the same shape — pick whichever field your parser already handles:

```json
{"error": "message text", "message": "message text", "detail": "message text"}
```

### Status codes you'll see

| Code | Meaning |
|---|---|
| `200` | OK — result bytes or JSON payload |
| `201` | Resource created (upload PUT, LoRA upload, approved image) |
| `202` | v2 job queued — use `job_id` + `stream_url` |
| `204` | Preview not ready yet — keep polling, **not** an error |
| `401` | Missing or invalid bearer token |
| `404` | Unknown id (job, upload, LoRA, history entry) |
| `409` | Result not ready OR trying to cancel a finished job |
| `413` | Upload exceeds 1 GB |
| `422` | Validation failure (bad keyframe, incompatible LoRA, retake content rejection) |
| `429` | Queue full (`Retry-After: 30`) |
| `500` | Internal error — filesystem paths scrubbed from the message |
| `503` | System paused for maintenance (`Retry-After: 300`) |

## History (per-api-key, 30-day retention)

- `GET /v2/history?limit=50&offset=0[&type=image|video|text-to-video|…]` — array of entries with `image_url` + `thumbnail_url`
- `GET /v2/history/{id}/image` — full-size media (MP4 or WEBP)
- `GET /v2/history/{id}/thumbnail` — 256 px JPEG (videos use first frame)

## Browser compatibility notes

- `EventSource` is native in all modern browsers — no library needed.
- Safari's `EventSource` gets confused by very long idle periods; our `: keepalive` comment every 15 s keeps it happy.
- If you need to cancel a stream early: `es.close()` on the client, OR `DELETE /v2/jobs/{id}` to actually cancel the work server-side (the stream will emit one final `cancelled` event and close).

## Full spec

For system endpoints, approved-images, compositions, SSE token lifecycle, exact pydantic shapes, and everything else: **[docs/API.md](./API.md)** (61 routes).
