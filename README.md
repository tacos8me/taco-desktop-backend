# taco-backend

Dual-GPU inference server for AI video, image, and music generation. Powers [noodle-i](https://i.noodlefinger.io) (image gen), [noodle-v](https://v.noodlefinger.io) (video gen), and [m.noodlefinger.io](https://m.noodlefinger.io) (music video gen).

- **Video**: LTX-2.3 (22B transformer) on cuda:0 — text-to-video, image-to-video, audio-to-video, temporal retake
- **Image**: Flux 2 Dev/Klein KV on cuda:0 — text-to-image, image-to-image, multi-reference editing
- **Image edit**: JoyAI-Image-Edit-Diffusers via out-of-process sidecar on cuda:1 (`127.0.0.1:8092`) — instruction-based single-image editing
- **Music (v1.2)**: ACE Step xl-base+LM on cuda:1 (`127.0.0.1:8001`) — text-to-music, covers, repainting, stem extraction
- **Chat**: Gemma 3 12B via llama-swap proxy
- **Dual-GPU layout**: cuda:0 = LTX ↔ Flux (2-tenant swap), cuda:1 = ACE + JoyAI (coexisting). See [GPU Management](#gpu-management) below.
- **Turbo mode (v1.2)**: toggle via `/v1/system/turbo` — claims both GPUs for LTX, 2 concurrent video workers, 2x throughput
- **Dashboard (v1.2)**: real-time GPU telemetry at `/dashboard`

## Quick Start

```bash
# Service management
systemctl --user start taco-backend
systemctl --user stop taco-backend
systemctl --user restart taco-backend

# Logs
journalctl --user -u taco-backend -f

# Health check
curl http://localhost:8090/health

# Run tests
uv run --no-sync pytest tests/ -q -p no:cacheprovider
```

---

## API Reference

Base URL: `https://i.noodlefinger.io` (production) or `http://localhost:8090` (local)

All endpoints except `/health` require: `Authorization: Bearer <api-key>`

### The Async Pattern (use this)

Every generation endpoint has a sync `/v1/...` (blocks until done) and async `/v2/...` (returns immediately) variant. **Use v2 for production.**

```
POST /v2/text-to-video        →  202 {"job_id": "xxx", "stream_url": "/v2/jobs/xxx/stream"}
GET  /v2/jobs/xxx/stream      →  SSE: {"status":"processing","progress":0.45,"phase":"denoising"}
                                  SSE: {"status":"processing","progress":0.90,"phase":"decoding"}
                                  SSE: {"status":"processing","progress":0.95,"phase":"encoding"}
                                  SSE: {"status":"completed","result_url":"/v2/jobs/xxx/result"}
GET  /v2/jobs/xxx/result      →  raw MP4 or WEBP bytes
```

**Prefer the SSE stream** at `GET /v2/jobs/{id}/stream` over polling `GET /v2/jobs/{id}` — one long-lived connection replaces 240+ GETs per video job. The event payload is the same JSON shape as the poll response, so the same parser works for both. Fall back to polling every 1–2 s only if your client can't do EventSource.

Jobs report both `progress` (0.0–1.0, capped at 0.90 during denoising) and `phase` (`"denoising" | "decoding" | "encoding" | "saving" | null`). The top 10% of the progress bar maps to post-denoise work (VAE decode, ffmpeg/WEBP encode, upload) — render the `phase` label instead of a frozen percentage during that window.

### Upload Flow

Images and audio must be uploaded before use. The upload ID is a 32-char hex UUID.

```bash
# 1. Create upload slot
UPLOAD=$(curl -s -H "Authorization: Bearer $KEY" https://i.noodlefinger.io/v1/upload)
URL=$(echo $UPLOAD | jq -r .upload_url)
URI=$(echo $UPLOAD | jq -r .storage_uri)

# 2. Upload raw bytes
curl -X PUT --data-binary @file.png "$URL"

# 3. Reference in generation requests
# → "image_uri": "storage://abc123...",  "audio_uri": "storage://def456..."
```

Max upload: 1GB. Upload IDs must be 32 hex chars (UUID without dashes).

---

## Video Generation (LTX-2.3)

**GPU**: cuda:0 (RTX PRO 6000 96GB) — shared with Flux in [GPU management](#gpu-management). LTX is loaded on demand: the first video request after server start, after a `/v1/ltx/unload`, or after any Flux image request pays a **cold LTX load of ~7–30 s** (depends on OS page cache). Subsequent video-to-video requests stay fast.

### Models

| Model | ID | Steps | Speed (~5s 1080p) | Best For |
|-------|-----|-------|-------------------|----------|
| Fast | `ltx-2-3-fast` | 8 | ~15s | Previews, iteration |
| Pro | `ltx-2-3-pro` | 30+5 | ~65s | Production |
| HQ | `ltx-2-3-hq` | 15+5 (res2s) | ~90s | Final render |

### Endpoints

| Sync | Async | Description |
|------|-------|-------------|
| `POST /v1/text-to-video` | `POST /v2/text-to-video` | Generate video from text |
| `POST /v1/image-to-video` | `POST /v2/image-to-video` | Video from image(s) |
| `POST /v1/audio-to-video` | `POST /v2/audio-to-video` | Video synced to audio |
| `POST /v1/retake` | `POST /v2/retake` | Re-render video segment |

### Request Fields

#### Text-to-Video
```json
{
  "prompt": "a cinematic sunset over the ocean",
  "model": "ltx-2-3-fast",
  "resolution": "1920x1080",
  "duration": 5.0,
  "fps": 24.0,
  "generate_audio": false,
  "camera_motion": "slow zoom in",
  "enhance_prompt": false,
  "lora": {"id": "uuid", "strength": 1.0}
}
```

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `prompt` | string | required | max 10,000 chars |
| `model` | string | required | `ltx-2-3-fast`, `ltx-2-3-pro`, `ltx-2-3-hq` |
| `resolution` | string | required | `1920x1080`, `1080x1920`, `2560x1440`, `1440x2560`, `3840x2160`, `2160x3840` |
| `duration` | float | required | 0 < x <= 30 seconds |
| `fps` | float | required | 0 < x <= 60 |
| `generate_audio` | bool | `false` | Generate audio track alongside video |
| `camera_motion` | string | `null` | Prepended to prompt as `[camera_motion]`, max 200 chars |
| `enhance_prompt` | bool | `false` | Gemma 3 rewrites prompt cinematically (+2-5s) |
| `lora` | object | `null` | `{"id": "lora_uuid", "strength": 0.0-2.0}` |

**Response** (v1): raw MP4 bytes (`Content-Type: video/mp4`)

**Frame math**: Frames snapped to `8k+1` internally (9, 17, 25, 33, 41, 49...).

#### Image-to-Video

Same as text-to-video, plus:

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `image_uri` | string | `null` | Single reference image (use this OR keyframes) |
| `image_strength` | float | `0.85` | Strength for single `image_uri` (0.0-1.0) |
| `keyframes` | array | `null` | See keyframe format below |

- Either `image_uri` or `keyframes`, not both
- Max 8 keyframes, at most one at frame 0
- `strength`: 0.0 (fully denoised) to 1.0 (fully constrained)

**Keyframe format**:
```json
{"image_uri": "storage://uuid", "frame_index": 0, "strength": 1.0}
```

`frame_index` accepts:
- **Integers**: `0`, `30`, `60` — absolute frame position
- **Negative integers**: `-1` (last frame), `-12` (12 frames before end, recommended "landing room")
- **Symbolic**: `"first"` (frame 0), `"middle"` (num_frames/2), `"last"` (num_frames-1)

**First/mid/last example** (recommended for music video transitions):
```json
{
  "keyframes": [
    {"image_uri": "storage://start", "frame_index": "first", "strength": 1.0},
    {"image_uri": "storage://mid", "frame_index": "middle", "strength": 0.5},
    {"image_uri": "storage://end", "frame_index": "last", "strength": 1.0}
  ]
}
```

**First + last only**:
```json
{
  "keyframes": [
    {"image_uri": "storage://start", "frame_index": "first", "strength": 1.0},
    {"image_uri": "storage://end", "frame_index": -1, "strength": 1.0}
  ]
}
```

#### Audio-to-Video

For music video generation (m.noodlefinger.io):

```json
{
  "prompt": "cinematic music video, neon lights, rain",
  "audio_uri": "storage://uuid",
  "image_uri": "storage://uuid",
  "model": "ltx-2-3-fast",
  "resolution": "1920x1080",
  "duration": 5.0,
  "fps": 24.0,
  "enhance_prompt": true
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `audio_uri` | string | required | Uploaded audio file (mp3, wav, etc.) |
| `image_uri` | string | `null` | Optional visual anchor for first frame |
| `duration` | float | `6.0` | How many seconds of audio to use (0-30) |

**Key behavior**:
- Audio is **frozen** — used as conditioning signal, not regenerated
- Original audio waveform is returned in the output MP4
- `image_uri` is optional but **strongly recommended** for visual coherence
- `enhance_prompt` recommended for short/terse prompts
- All three models supported: fast (~15s), pro (~65s), hq (~90s)

**Music video workflow**:
```bash
# 1. Upload audio track
AUDIO=$(curl -s -H "Authorization: Bearer $KEY" https://i.noodlefinger.io/v1/upload)
curl -X PUT --data-binary @track.mp3 "$(echo $AUDIO | jq -r .upload_url)"
AUDIO_URI=$(echo $AUDIO | jq -r .storage_uri)

# 2. Upload reference image (optional)
IMG=$(curl -s -H "Authorization: Bearer $KEY" https://i.noodlefinger.io/v1/upload)
curl -X PUT --data-binary @reference.png "$(echo $IMG | jq -r .upload_url)"
IMG_URI=$(echo $IMG | jq -r .storage_uri)

# 3. Generate
JOB=$(curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"prompt\":\"cinematic music video\",\"audio_uri\":\"$AUDIO_URI\",\"image_uri\":\"$IMG_URI\",\"model\":\"ltx-2-3-fast\",\"resolution\":\"1920x1080\",\"duration\":5,\"fps\":24,\"enhance_prompt\":true}" \
  https://i.noodlefinger.io/v2/audio-to-video | jq -r .job_id)

# 4. Poll until completed
while true; do
  STATUS=$(curl -s https://i.noodlefinger.io/v2/jobs/$JOB | jq -r .status)
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && echo "FAILED" && break
  sleep 3
done

# 5. Download result
curl https://i.noodlefinger.io/v2/jobs/$JOB/result --output musicvideo.mp4
```

#### Retake (re-render video segment)

```json
{
  "video_uri": "storage://uuid",
  "start_time": 1.0,
  "duration": 2.5,
  "mode": "replace_video",
  "prompt": "make it more dramatic"
}
```

| Mode | Video | Audio |
|------|-------|-------|
| `replace_audio_and_video` | Re-render | Re-render |
| `replace_video` | Re-render | Keep original |
| `replace_video_only` | Re-render | No audio track |
| `replace_audio` | Keep original | Re-render |

---

## Image Generation (Flux 2)

**GPU**: cuda:0 (RTX PRO 6000 96GB) — shared with LTX in [GPU management](#gpu-management). If LTX is currently resident, the first image request pays a **~3 s LTX eviction** before the usual forward pass (~22 s Dev cache-hit, ~45 s Dev cold, ~3 s Klein). After an image request, LTX is **not** auto-reloaded — it stays evicted until the next video request triggers a cold load.

### Models

| Model | ID | Steps | Speed (1024x1024) | Notes |
|-------|-----|-------|-------------------|-------|
| Dev | `flux2-dev` | 20–50 | ~50–90s | Full bf16, highest quality, TE offloaded to CPU between requests |
| Dev Turbo | `flux2-dev` + `turbo:true` | 8 | ~25–35s | 8-step sigma schedule; compose with `flux2-turbo` folder-drop LoRA for full distilled fidelity |
| Klein KV | `flux2-klein` | 4 | ~3s | Ultra-fast, full bf16 resident, editing |
| JoyAI Edit | `joyai-edit` | 30 | ~78s | **v1.1.8** — instruction-based single-image editing via out-of-process sidecar on `127.0.0.1:8092`. Exactly 1 `image_uri`. No LoRA. `LOAD_JOYAI=1` required. |

Precision: **full bf16 throughout** (matches ComfyUI default). FP8 layerwise casting was dropped in v1.1.1 after diagnosing screendoor/grid artifacts traced to the FP8 + fused-LoRA interaction. Dev uses `enable_model_cpu_offload` because all components together exceed 96 GB in bf16; Klein fits comfortably and stays fully resident.

Models swap on demand — first request to a different model adds ~20–30 s load time.

### Endpoints

| Sync | Async | Description |
|------|-------|-------------|
| `POST /v1/text-to-image` | `POST /v2/text-to-image` | Image from text |
| `POST /v1/image-to-image` | `POST /v2/image-to-image` | Single-ref edit |
| `POST /v1/image-edit` | `POST /v2/image-edit` | Multi-ref edit (Flux Klein / Dev) OR single-image instruction edit (`joyai-edit`, v1.1.8) |

### Request Fields

```json
{
  "prompt": "a cat wearing sunglasses",
  "model": "flux2-klein",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 4,
  "guidance_scale": 4.0,
  "seed": 42,
  "turbo": false
}
```

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `prompt` | string | required | max 10,000 chars |
| `model` | string | `flux2-dev` | `flux2-dev` or `flux2-klein` |
| `width` | int | `1024` | 64-4096, snapped to multiples of 16 |
| `height` | int | `1024` | 64-4096, snapped to multiples of 16 |
| `num_inference_steps` | int | `50` | 1-100. Use 4 for Klein, 50 for Dev |
| `guidance_scale` | float | `4.0` | 0-20. Ignored by Klein (distilled, no CFG) |
| `seed` | int | random | For reproducibility. Same seed + params = same image |
| `turbo` | bool | `false` | Dev only — forces 8 steps, guidance 2.5 |
| `lora` | object | `null` | `{id, strength}` — see Flux LoRA section below |

For `image-to-image`: add `"image_uri": "storage://uuid"`

For `image-edit`:
```json
{
  "prompt": "blend these styles",
  "image_uris": ["storage://uuid1", "storage://uuid2"],
  "model": "flux2-klein",
  "num_inference_steps": 4
}
```
- `flux2-klein` / `flux2-dev`: 1–10 reference images, all references condition every output (style/composition blending).
- **`joyai-edit` (v1.1.8)**: **exactly one** `image_uri`, instruction-based editing. LoRA is not supported (422). Prompts are plain English — the server wraps them in a chat template server-side. Latency ~78 s for 30 steps at 1024². Runs in a sidecar on `127.0.0.1:8092`; if `LOAD_JOYAI` is unset or the sidecar is down the request returns `503 joyai_disabled` / `503 sidecar_unreachable` — fall back to `flux2-klein`.

**Response** (v1): raw WEBP bytes (`Content-Type: image/webp`, quality 95)

### Flux LoRAs (folder-drop)

Drop `.safetensors` files into `/mnt/nvme-1/servers/taco-backend/flux_loras/` — they're picked up on server start and via `POST /v1/flux-loras/rescan`. There is no upload endpoint by design; files are managed with `cp` / `rm`. The LoRA's `id` is the slugified filename stem (e.g., `My Style v2.safetensors` → `my-style-v2`).

Optionally place a same-named `.json` sidecar alongside the `.safetensors` for display metadata:
```json
{
  "name": "Painterly Style",
  "description": "Oil-painting look, trained on 2000 steps",
  "trigger_word": "inthestyleof",
  "model_compat": ["flux2-dev"]
}
```

**Endpoints:**

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/flux-loras` | GET | Yes | List discovered Flux LoRAs |
| `/v1/flux-loras/rescan` | POST | Yes | Re-scan the folder (after `cp`) |

**Use in generation** — add `lora` to any text-to-image / image-to-image / image-edit request:
```json
{
  "prompt": "a cyberpunk cat",
  "model": "flux2-dev",
  "lora": {"id": "my-style-v2", "strength": 0.8},
  "seed": 42
}
```

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `lora.id` | string | required | ID from `GET /v1/flux-loras` |
| `lora.strength` | float | `1.0` | 0.0-2.0 |

**Behavior:**
- Works with both `flux2-dev` and `flux2-klein` (diffusers auto-converts Flux 1/2 LoRA formats). Dev-trained LoRAs applied to Klein return **422** with a clean error (different transformer dimensions).
- Composable with `turbo: true` (Turbo is sigma-schedule-based, LoRA is orthogonal).
- LoRAs are attached as **adapters** (not fused). Changing `lora.strength` is a **free runtime operation** — no reload. Only changing `lora.id` (or switching models) triggers a full pipeline reload (~30–60 s for Dev).
- Discovery is case-insensitive via slugification. Renaming the file changes the ID.

---

## Job Queue

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/jobs/{id}/stream` | GET | **SSE live updates** — status / progress / phase. **Use this instead of polling.** |
| `/v2/jobs/{id}` | GET | Poll status + progress (fallback if you can't do SSE) |
| `/v2/jobs/{id}/result` | GET | Download result (MP4/WEBP) |
| `/v2/jobs/{id}/preview` | GET | Preview JPEG (during processing, 204 if none yet) |
| `/v2/jobs/{id}` | DELETE | Cancel job |

**Status response** (identical shape for poll and SSE `data:` payload):
```json
{
  "job_id": "unguessable_token",
  "status": "queued | processing | completed | failed | cancelled",
  "progress": 0.5,
  "phase": "denoising | decoding | encoding | saving | null",
  "queue_position": 3,
  "result_url": "/v2/jobs/{id}/result",
  "result_media_type": "video/mp4 | image/webp",
  "error": {"code": "cuda_oom", "message": "..."}
}
```

### SSE stream (`/v2/jobs/{id}/stream`)

EventSource-compatible. Emits one event on connect with the current state, then emits again only when `(status, progress, phase, error_code)` changes. Closes the stream on terminal state (completed / failed / cancelled) after one final event. Idle periods (e.g. job sitting queued) get a `: keepalive` comment every 15 s so intermediate proxies don't drop the connection.

**Auth**: browsers can't set custom headers on `EventSource`, so the endpoint accepts either a bearer `Authorization` header **or** a `?token=<sse-token>` query-param. Get a 5-minute disposable token via `POST /v1/sse-token`.

```js
// Browser
const { token } = await fetch("/v1/sse-token", {
  method: "POST",
  headers: { Authorization: `Bearer ${KEY}` },
}).then(r => r.json());

const es = new EventSource(`/v2/jobs/${job_id}/stream?token=${token}`);
es.onmessage = (ev) => {
  const { status, progress, phase, result_url } = JSON.parse(ev.data);
  setProgress(progress);
  setPhase(phase);
  if (status === "completed") {
    fetchResult(result_url);
    es.close();
  } else if (status === "failed" || status === "cancelled") {
    es.close();
  }
};
```

```bash
# curl
curl -N -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JID/stream"
```

- Max queue depth: 10 (returns 429 `Retry-After: 30`)
- Results expire after 10 minutes
- Job IDs are unguessable tokens, but auth is still required for all job endpoints

### Batch Scheduler (v1.2)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v2/batch` | POST | Submit a batch of 1–50 generation jobs |
| `GET /v2/batch/{id}` | GET | Poll batch status + partial results |
| `DELETE /v2/batch/{id}` | DELETE | Cancel remaining items in a batch |

```json
{
  "items": [
    {"type": "text-to-image", "params": {"prompt": "a cat", "model": "flux2-klein", "width": 1024, "height": 1024, "num_inference_steps": 4}},
    {"type": "text-to-video", "params": {"prompt": "a dog", "model": "ltx-2-3-fast", "resolution": "1920x1080", "duration": 5, "fps": 24}}
  ],
  "priority": "normal",
  "callback_url": null
}
```

Items are sorted images-first (Klein before Dev) to minimize GPU swaps. In turbo mode, the batch worker processes 2 video items concurrently via `asyncio.gather`. Supported types: `text-to-image`, `image-to-image`, `image-edit`, `text-to-video`, `image-to-video`.

---

## Music Generation (v1.2)

**GPU**: cuda:1 (ACE Step xl-base+LM, ~18 GB resident, concurrent with JoyAI). Runs as a separate systemd service (`ace-step`) on `127.0.0.1:8001`.

| Sync | Async | Description |
|------|-------|-------------|
| `POST /v1/music` | `POST /v2/music` | Generate music from text/lyrics |

### Request Fields

```json
{
  "prompt": "upbeat electronic dance track",
  "lyrics": "[Instrumental]",
  "duration": 60.0,
  "audio_format": "mp3",
  "task_type": "text2music",
  "num_inference_steps": 50,
  "guidance_scale": 7.0,
  "bpm": 128,
  "seed": 42
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | string | required | Music description, max 10,000 chars |
| `lyrics` | string | `"[Instrumental]"` | Song lyrics, max 50,000 chars |
| `duration` | float | `60.0` | 0–600 seconds |
| `audio_format` | string | `"mp3"` | `mp3`, `flac`, `wav`, `wav32`, `opus`, `aac` |
| `task_type` | string | `"text2music"` | `text2music`, `cover`, `repaint`, `extract`, `lego`, `complete` |
| `source_audio_uri` | string | null | Required for cover/repaint/extract/lego/complete tasks |
| `reference_audio_uri` | string | null | Optional reference audio |
| `bpm` | int | null | 30–300 |
| `key_scale` | string | null | Musical key |
| `num_inference_steps` | int | `50` | 1–200 |
| `guidance_scale` | float | `7.0` | 0–15 |

- Latency: ~2–10 s per track depending on duration and steps
- Phase for music jobs: `"generating"` (not `"denoising"`)
- Music queue cap: `MAX_MUSIC_PENDING` (default 5), returns `429 music_queue_full`
- Gated by `LOAD_ACE=1` env var. Returns `503` when disabled or during turbo mode.
- Response (v1): raw audio bytes with appropriate `Content-Type` (e.g., `audio/mpeg` for mp3)

---

## System & Management

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Server status |
| `/dashboard` | GET | No | GPU management dashboard (static HTML SPA) |
| `/v1/system/pause` | POST | Yes | Free all GPU VRAM (evicts LTX + Flux, cancels queued jobs) |
| `/v1/system/resume` | POST | Yes | Reload models after pause |
| `/v1/system/turbo` | POST | Yes | Toggle turbo mode — dual-GPU LTX for 2x video throughput (v1.2) |
| `/v1/system/gpu` | GET | Yes | nvidia-smi GPU telemetry (2 s cache) — memory, temp, utilization, tenant info (v1.2) |
| `/v1/flux/unload` | POST | Yes | Unload Flux from cuda:0 |
| `/v1/flux/reload` | POST | Yes | Reload Flux to cuda:0 |
| `/v1/ltx/unload` | POST | Yes | Unload LTX from cuda:0 (full encoder-hub + transformer reclaim) |
| `/v1/ltx/reload` | POST | Yes | Reload LTX to cuda:0 |
| `/v1/loras` | GET | Yes | List LTX video LoRAs |
| `/v1/loras` | POST | Yes | Upload LTX LoRA (multipart: `file`, `name`, `description`) |
| `/v1/loras/{id}` | DELETE | Yes | Delete LTX LoRA |
| `/v1/flux-loras` | GET | Yes | List Flux LoRAs (see Flux LoRAs section — folder-drop, no upload) |
| `/v1/flux-loras/rescan` | POST | Yes | Re-scan `flux_loras/` directory |
| `/v1/chat/completions` | POST | Yes | Chat/vision proxy to llama-swap (OpenAI-compatible) |

### GPU Management

**LTX ↔ Flux, mutually exclusive on cuda:0 (2-tenant swap).** Their combined active memory footprint exceeds the 96 GB physical budget. The dispatcher automatically swaps between them on every inference request, inside the inference lock, so there is no race window and **the client API contract is unchanged**.

- Before any **video** request the dispatcher calls `_ensure_ltx_resident()` — loads LTX if not already resident, evicts Flux if present.
- Before any **Flux** image request the dispatcher calls `_ensure_flux_ready()` — evicts LTX if resident.
- After an image request, LTX is **not** eagerly reloaded; it stays evicted until the next video request.
- **cuda:1** runs ACE (music, `127.0.0.1:8001`) and JoyAI (image edit, `127.0.0.1:8092`) concurrently — no swap needed, combined ~68 GB fits within 96 GB.

**Latency:**

| Scenario | Added swap cost | Notes |
|----------|-----------------|-------|
| Video → video | 0 s | LTX stays resident, warm forward pass |
| Image → image (same model) | 0 s | Flux stays resident |
| Image → image (model change) | ~20–30 s | Normal Flux Dev↔Klein swap |
| **Flux → LTX** | **+7–30 s** | Cold LTX load (`~7 s` warm page cache, `~30 s` cold) |
| **LTX → Flux** | **+3 s** | LTX eviction, then the usual Flux forward pass |
| **Turbo entry** | **~20 s** | Evict ACE+JoyAI+Flux, load dual-GPU LTX |
| **Turbo exit** | **~15 s** | Evict dual-GPU LTX, restore single-GPU, restart ACE+JoyAI |

Operators who want explicit control can still call `/v1/ltx/unload`, `/v1/ltx/reload`, `/v1/flux/unload`, `/v1/flux/reload` manually.

#### Turbo mode (v1.2)

Toggle via `POST /v1/system/turbo` (body: `{"enable": true/false}`). Claims both GPUs for LTX with **2 concurrent denoiser workers** — processes **2 video jobs at a time**. While active, Flux/ACE/JoyAI/music endpoints all return `503 turbo_mode_active`. Dashboard provides UI controls for turbo toggle.

### History

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/history` | GET | List past generations (`?limit=50&offset=0&type=text-to-video`) |
| `/v2/history/{id}/image` | GET | Download generation result (MP4/WEBP) |
| `/v2/history/{id}/thumbnail` | GET | 256px JPEG thumbnail |

Per-API-key scoped. 30-day retention.

### Approved Images (noodle-i → noodle-v pipeline)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/approved-images` | POST | Submit image for approval |
| `/v1/approved-images` | GET | List approved images (`?limit=50&offset=0`) |
| `/v1/approved-images/{id}/file` | GET | Download approved image |
| `/v1/approved-images/events` | GET | SSE stream of new approvals (use `?token=` from `/v1/sse-token`) |
| `/v1/sse-token` | POST | Get 5-minute disposable token for SSE connections |

### Compositions (video editing)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/compositions` | POST | Create composition |
| `/v2/compositions` | GET | List compositions |
| `/v2/compositions/{id}` | GET | Get composition |
| `/v2/compositions/{id}` | PUT | Update composition |
| `/v2/compositions/{id}` | DELETE | Delete composition |
| `/v2/compositions/{id}/export` | POST | Export as video (async job) |

### Character Ranking (vision)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/char/rank` | POST | Rank character consistency between reference and generated image |

Request: `{"rank_image_uri": "storage://...", "generated_image_uri": "storage://...", "prompt": "..."}`

---

## Error Handling

```json
{"error": "code", "message": "Human-readable description"}
```

| HTTP | Meaning | Action |
|------|---------|--------|
| 401 | Invalid API key | Check `Authorization: Bearer` header |
| 404 | Not found | Check ID; results expire after 10min |
| 409 | Not ready | Job still processing — keep polling |
| 413 | Upload too large | Max 1GB per upload |
| 422 | Validation error | Check request field constraints |
| 429 | Queue full (10 jobs) | Wait 30s, retry |
| 500 | GPU OOM or generation failed | Reduce resolution or wait |
| 503 | System paused | Server in maintenance, wait for resume |

---

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LOAD_FLUX` | `false` | Set `1`/`true` to enable Flux on cuda:0 |
| `LOAD_JOYAI` | `false` | Set `1`/`true` to enable the JoyAI image-edit sidecar on cuda:1 (`127.0.0.1:8092`) |
| `LOAD_ACE` | `false` | Set `1`/`true` to enable ACE music generation on cuda:1 (`127.0.0.1:8001`) |
| `ACE_SIDECAR_URL` | `http://127.0.0.1:8001` | URL of the ACE Step sidecar |
| `LTX_DEVICE` | `cuda:0` | Device for LTX video generation (shared with Flux in swap mode) |
| `FLUX_DEVICE` | `cuda:0` | Device for Flux image generation (shared with LTX in swap mode) |
| `GEMMA_VARIANT` | `default` | `default` (standard) or `sikaworld` (uncensored) |
| `MAX_MUSIC_PENDING` | `5` | Max concurrent music generation jobs |
| `TURBO_GPU_DEVICES` | `["cuda:0", "cuda:1"]` | GPUs assigned to LTX in turbo mode |
| `MAX_BATCH_QUEUE_DEPTH` | `5` | Max concurrent batch submissions |
| `MAX_BATCH_ITEMS` | `50` | Max items per batch request |

### GPU Topology (v1.2)

| Device | GPU | VRAM | Role | Active residency |
|--------|-----|------|------|------------------|
| cuda:0 | RTX PRO 6000 Blackwell | 96 GB | **LTX ↔ Flux** (2-tenant swap, auto-swap on dispatch) | LTX active ~79 GB; Flux active ~81 GB (Dev bf16 + `enable_model_cpu_offload`) / ~32 GB (Klein resident) |
| cuda:1 | RTX PRO 6000 Blackwell | 96 GB | **ACE** (~18 GB) + **JoyAI** (~50 GB) — coexisting, no swap | Combined ~68 GB |

**Memory budget**: LTX active (~79 GB) + Flux active (~81 GB) > 96 GB physical, so they cannot coexist on cuda:0. The dispatcher auto-swaps inside the inference lock — clients do not orchestrate it. ACE and JoyAI coexist on cuda:1 without swapping. In turbo mode, both GPUs are assigned to LTX and ACE+JoyAI+Flux are evicted. See [GPU Management](#gpu-management) for latency details.

---

## Development

```bash
# Manual run
bash run.sh

# Tests (IMPORTANT: use --no-sync to prevent nvidia package downgrades)
uv run --no-sync pytest tests/ -q -p no:cacheprovider
```

### Project Structure

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app, all endpoints, job queue, batch scheduler, turbo mode |
| `split_model_manager.py` | LTX-2 pipeline, transformer swapping, VAE decode |
| `flux_manager.py` | Flux 2 pipeline, model swapping (Dev/Klein), adapter-mode LoRA |
| `ace_client.py` | ACE music generation sidecar client (httpx → ace-step on cuda:1:8001) |
| `job_queue.py` | Async job queues with per-device workers |
| `upload_store.py` | UUID file storage for uploads/results |
| `history_store.py` | SQLite generation history (per API key) |
| `lora_registry.py` | LoRA management with JSON registry |
| `chat_manager.py` | llama-swap proxy for chat/vision |
| `nvfp4_loader.py` | NVFP4→BF16 dequantizer (comfy-kitchen) |
| `dashboard.html` | GPU management dashboard SPA (served at /dashboard) |
| `config.py` | All configuration |
| `run.sh` | Startup script (env vars + uvicorn) |
| `taco-backend.service` | systemd user service |
| `CLAUDE.md` | Architecture internals |
| `AGENTS.md` | Optimization backlog |

### Dependencies

- Python 3.13+, uv package manager
- PyTorch 2.11.0+cu130 (Blackwell sm_120, FlexAttention/FA4)
- diffusers 0.38.0.dev0 from git main (required for Flux2KleinKVPipeline)
- LTX-2 repo at `/mnt/nvme-1/repos/LTX-2` (editable install)
- cuDNN 9.20+, cuBLAS 13.2+ (manually pinned — revert on `uv sync`)
- comfy-kitchen (NVFP4 dequantization for Sikaworld text encoder)
