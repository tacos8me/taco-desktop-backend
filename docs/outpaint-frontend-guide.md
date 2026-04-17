# Video Outpaint — Frontend Integration Guide

**Applies to:** taco-backend v1.7.0 (2026-04-17) and later. Canonical reference: [docs/API.md](./API.md). Related: [retake-frontend-guide.md](./retake-frontend-guide.md), [QUICKSTART.md](./QUICKSTART.md).

## What it does

Expands a source video's canvas to a larger `target_resolution` by scaling the source proportionally, letterboxing it with pure-black padding at the chosen `position`, and using an IC-LoRA (`oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint`) to fill those black regions with temporally coherent generated content driven by your `prompt`. The result is a single video at `target_resolution` where the original footage is preserved pixel-perfect in place and the surrounding canvas becomes an AI-generated extension of the scene.

## Quick start (curl)

```bash
# 1. Upload source video
UPLOAD=$(curl -sX POST "$API/v1/upload" -H "Authorization: Bearer $KEY")
UPLOAD_URL=$(echo "$UPLOAD" | jq -r .upload_url)
STORAGE_URI=$(echo "$UPLOAD" | jq -r .storage_uri)
curl -sX PUT "$UPLOAD_URL" -H "Authorization: Bearer $KEY" --data-binary @source.mp4

# 2. Submit outpaint job
JOB_ID=$(curl -sX POST "$API/v2/video-outpaint" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"video_uri\":\"$STORAGE_URI\",\"prompt\":\"extend the scene naturally\",\"target_resolution\":\"1920x1080\",\"position\":\"center\",\"duration\":5.0,\"fps\":24}" \
  | jq -r .job_id)

# 3. Poll until done (or use SSE — see below)
while S=$(curl -s "$API/v2/jobs/$JOB_ID" -H "Authorization: Bearer $KEY" | jq -r .status); \
      [ "$S" != "completed" ] && [ "$S" != "failed" ] && [ "$S" != "cancelled" ]; do sleep 1; done

# 4. Download result (silent MP4)
curl -s "$API/v2/jobs/$JOB_ID/result" -H "Authorization: Bearer $KEY" -o out.mp4
```

## Endpoint

```
POST /v2/video-outpaint
Authorization: Bearer <api-key>
Content-Type: application/json
```

Returns `202` with the same submission envelope as every other v2 endpoint (`job_id`, `status`, `poll_url`, `stream_url`).

## Request schema

| Field | Type | Constraint | Default | Purpose |
|-------|------|-----------|---------|---------|
| `video_uri` | string | required | — | `storage://<uuid>` URI from `/v1/upload` or a prior `result_uri`. Video is scaled proportionally to fit within `target_resolution` |
| `prompt` | string | required, `<=10000` chars | — | Describes what should appear in the outpainted regions |
| `target_resolution` | string enum | see below | — | Final canvas size — reuses the same `Resolution` union as t2v/i2v |
| `position` | string enum | see 9 values | `"center"` | Where the scaled source sits within the target canvas |
| `duration` | float | `0 < d <= 30` seconds | — | Output video length. Frame count is `duration * fps` snapped to `8k + 1` |
| `fps` | float | `0 < fps <= 60` | — | Output frame rate |
| `seed` | int | `>= 0` | `0` (random) | Fixed seed for reproducibility. `0` / omitted → server picks a random 32-bit seed |
| `enhance_prompt` | bool | — | `false` | Run Gemma prompt-enhancement before generation |
| `lora` | `{id, strength}` | optional | `{id: "ic-lora-outpaint", strength: 1.0}` | Override the default outpaint LoRA. Must be a registered IC-LoRA |
| `conditioning_strength` | float | `[0.0, 1.0]` | `1.0` | Scalar attention weight on the IC-LoRA conditioning. `< 1.0` loosens fidelity to source (more creative fill, less preservation) |
| `skip_stage_2` | bool | — | `false` | When `true`, output is at half `target_resolution` (faster preview) |

Valid `target_resolution` values:
- `"1920x1080"` — 1080p landscape
- `"1080x1920"` — 1080p portrait
- `"2560x1440"` — 1440p landscape
- `"1440x2560"` — 1440p portrait
- `"3840x2160"` — 4K landscape
- `"2160x3840"` — 4K portrait

## The 9 `position` values

Where the scaled source sits inside the target canvas. Black padding fills the remainder; the LoRA fills the black.

```
+--------+--------+--------+      e.g. position = "top_left":
|   TL   |  TOP   |   TR   |      +---------+-------------+
+--------+--------+--------+      | source  |  outpaint   |
|  LEFT  | CENTER | RIGHT  |      +---------+             |
+--------+--------+--------+      |      outpaint         |
|   BL   | BOTTOM |   BR   |      +-----------------------+
+--------+--------+--------+
```

Enum: `"center"`, `"left"`, `"right"`, `"top"`, `"bottom"`, `"top_left"`, `"top_right"`, `"bottom_left"`, `"bottom_right"`.

## Response

**On submit:**

```json
{
  "job_id": "job_01H...",
  "status": "queued",
  "poll_url": "/v2/jobs/job_01H...",
  "stream_url": "/v2/jobs/job_01H.../stream"
}
```

**Polling** (`GET /v2/jobs/{id}`):

```json
{
  "job_id": "...",
  "status": "processing",
  "progress": 0.67,
  "phase": "denoising"
}
```

Phases: `denoising` → `decoding` → `encoding` → `saving` → (terminal) `completed`. Progress caps at `0.90` during denoising; the top 10 % covers post-denoise work. `skip_stage_2=true` jobs skip the stage-2 refine pass entirely.

**Download** (`GET /v2/jobs/{id}/result`):

- Content-Type: `video/mp4`
- Silent (no audio track) — source audio passthrough is deferred to v1.7.x.

## SSE vs polling

Prefer SSE. Video outpaint takes 20-45 s depending on `skip_stage_2`; polling every second is 40+ GETs that SSE replaces with one long-lived connection.

```js
// 1. Issue a short-lived SSE token (EventSource can't set custom headers)
const { token } = await fetch(`${API}/v1/sse-token`, {
  method: "POST",
  headers: { Authorization: `Bearer ${KEY}` },
}).then(r => r.json());

// 2. Stream
const es = new EventSource(`${API}/v2/jobs/${jobId}/stream?token=${token}`);
es.onmessage = (ev) => {
  const { status, progress, phase } = JSON.parse(ev.data);
  // update UI
  if (status === "completed" || status === "failed" || status === "cancelled") es.close();
};
```

Fall back to `GET /v2/jobs/{id}` every 1-2 s only if EventSource isn't available. SSE tokens expire after 5 minutes; request a new one if the stream disconnects.

## Timing expectations

| Scenario | Wall-clock |
|----------|-----------|
| 3 s @ 1080p, `skip_stage_2=true` (half-res preview) | ~20 s |
| 5 s @ 1080p, full 2-stage | ~35-45 s |
| 5 s @ 1440p, full 2-stage | ~55-75 s |
| 5 s @ 4K, full 2-stage | ~2-3 min |

Turbo mode (`POST /v1/system/turbo`) doubles throughput but does not reduce per-job latency. The Modal remote pool (v1.6, up to 4 extra workers) adds concurrency, not speed.

## Known limitations

- **Silent output** — no audio passthrough in v1.7.0. The source's audio track is discarded. Planned for v1.7.x.
- **Dark-content gotcha** — the LoRA uses pure-black (RGB `(0,0,0)`) as its fill sentinel. Night scenes, deep shadows, or heavily underexposed footage can confuse the model into "filling" regions that should have been preserved. Workaround: apply gamma 2.0 to the source before upload, then gamma 0.5 to the output after decode. Automate this in the frontend if your users regularly upload dark footage.
- **Same-aspect-ratio = no-op** — if the source's aspect ratio already matches `target_resolution`, there's no black padding to fill, and the output is just a re-encode of the scaled source. Validate aspect mismatch client-side before submitting.
- **Max duration** — 30 s (enforced server-side via `duration <= 30`).
- **Frame count snapping** — `duration * fps` is snapped to the nearest `8k + 1`. A request for `duration=5, fps=24` produces 113 or 121 frames, not exactly 120.
- **LoRA stays fused across stage 2** — the server keeps the outpaint LoRA active through both stages (upstream `ICLoraPipeline` drops it at stage 2). Acceptable deviation to avoid ~30 s re-fusion mid-request.

## Full JavaScript example

```js
const API = "http://localhost:8090";
const KEY = "your-api-key";

async function outpaintVideo(file, prompt, targetResolution = "1920x1080", position = "center") {
  // 1. Upload source
  const slot = await fetch(`${API}/v1/upload`, {
    method: "POST",
    headers: { Authorization: `Bearer ${KEY}` },
  }).then(r => r.json());

  await fetch(slot.upload_url, {
    method: "PUT",
    body: file,
    headers: { Authorization: `Bearer ${KEY}` },
  });

  // 2. Submit outpaint
  const job = await fetch(`${API}/v2/video-outpaint`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      video_uri: slot.storage_uri,
      prompt,
      target_resolution: targetResolution,
      position,
      duration: 5.0,
      fps: 24,
      conditioning_strength: 1.0,
      skip_stage_2: false,
    }),
  }).then(r => r.json());

  if (job.error) throw new Error(job.error);

  // 3. Get an SSE token and stream
  const { token } = await fetch(`${API}/v1/sse-token`, {
    method: "POST",
    headers: { Authorization: `Bearer ${KEY}` },
  }).then(r => r.json());

  return new Promise((resolve, reject) => {
    const es = new EventSource(`${API}/v2/jobs/${job.job_id}/stream?token=${token}`);
    es.onmessage = async (ev) => {
      const { status, progress, phase } = JSON.parse(ev.data);
      ui.setProgress(progress);
      ui.setPhase(phase);

      if (status === "completed") {
        es.close();
        const blob = await fetch(`${API}/v2/jobs/${job.job_id}/result`, {
          headers: { Authorization: `Bearer ${KEY}` },
        }).then(r => r.blob());
        resolve(URL.createObjectURL(blob));
      } else if (status === "failed" || status === "cancelled") {
        es.close();
        reject(new Error(`Job ${status}`));
      }
    };
    es.onerror = () => { es.close(); reject(new Error("SSE connection lost")); };
  });
}
```

## Error handling

| Status | Meaning | Client action |
|--------|---------|---------------|
| `202` | Job queued, use `stream_url` / `poll_url` | Proceed to step 3 |
| `401` | Missing / invalid bearer | Fix auth, don't retry blindly |
| `404` | Unknown `video_uri` or LoRA id | Surface to user: "source video not found" or "LoRA not found" |
| `422` | Validation (bad resolution, bad position, duration out of range, frame count mismatch) | Surface the `error` field to user — it describes the exact issue |
| `429 queue_full` | Queue depth >= 10 | Wait `Retry-After` seconds (default 30) and retry |
| `500` | Internal failure, or LoRA resolve returned None | Retry once with exponential backoff; log and escalate if persistent |
| `503 turbo_mode_active` | Turbo is on and mode conflicts (outpaint works under turbo, but other-type endpoints don't) | **N/A for outpaint — outpaint runs in turbo mode normally.** If you see 503 here it's transient; wait `Retry-After: 300` and retry |
| `503 system_paused` | `/v1/system/pause` in effect | Wait and retry per `Retry-After` |

Retry-safe codes: `429`, `500`, `503`. Surface directly: `401`, `404`, `422`.

## Custom outpaint LoRA

Pass `lora` to override the default. The LoRA must be registered with `strategy: "ic_lora_outpaint"` in `registry.json`, and the file must live in `loras/`.

```json
{
  "video_uri": "storage://...",
  "prompt": "...",
  "target_resolution": "1920x1080",
  "position": "center",
  "duration": 5.0,
  "fps": 24,
  "lora": { "id": "my-custom-outpaint", "strength": 0.9 }
}
```

- `strength` range: `[0.0, 2.0]` (registry-validated).
- If the id isn't in the registry: `404 "LoRA not found"`.
- If it's registered with a non-outpaint strategy: server returns `422` (not yet implemented — currently any registered LoRA path works, but future versions may enforce strategy match).
- Custom LoRAs currently fall back to single-machine dispatch over the Modal remote pool; only the pre-staged default LoRA runs on Modal containers for v1.7.0.

## See also

- **[API.md](./API.md)** — canonical reference: exact pydantic shape, full error envelope, all status codes
- **[retake-frontend-guide.md](./retake-frontend-guide.md)** — regenerate a time window of an existing video
- **[QUICKSTART.md](./QUICKSTART.md)** — upload flow, SSE token lifecycle, auth, shared patterns
- **[models.md](./models.md)** — LoRA registry and strategy dispatch
- **[CHANGELOG.md](../CHANGELOG.md)** — v1.7.0 release notes (implementation detail, not needed for integration)
