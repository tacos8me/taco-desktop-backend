# taco-backend

Multi-GPU inference server for AI video and image generation. Powers [noodle-i](https://i.noodlefinger.io) (image gen), [noodle-v](https://v.noodlefinger.io) (video gen), and [m.noodlefinger.io](https://m.noodlefinger.io) (music video gen).

- **Video**: LTX-2.3 (22B transformer) on cuda:1 — text-to-video, image-to-video, audio-to-video, temporal retake
- **Image**: Flux 2 Dev/Klein KV on cuda:0 — text-to-image, image-to-image, multi-reference editing
- **Chat**: Gemma 3 12B via llama-swap proxy
- **Serialized**: Flux and LTX share a single inference lock (FP8 cuBLAS constraint)

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
POST /v2/text-to-video  →  202 {"job_id": "xxx"}
GET  /v2/jobs/xxx       →  {"status": "processing", "progress": 0.5}
GET  /v2/jobs/xxx       →  {"status": "completed", "result_url": "/v2/jobs/xxx/result"}
GET  /v2/jobs/xxx/result →  raw MP4 or WEBP bytes
```

Poll `/v2/jobs/{id}` every 2-5 seconds until `status` is `completed` or `failed`.

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

**GPU**: cuda:1 (RTX PRO 6000 96GB)

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

**GPU**: cuda:0 (RTX PRO 6000 96GB)

### Models

| Model | ID | Steps | Speed (1024x1024) | Notes |
|-------|-----|-------|-------------------|-------|
| Dev | `flux2-dev` | 50 | ~60s | Highest quality |
| Dev Turbo | `flux2-dev` + `turbo:true` | 8 | ~10s | Fused Turbo LoRA |
| Klein KV | `flux2-klein` | 4 | ~1.5s | Ultra-fast, editing |

Models swap on demand — first request to a different model adds ~5-10s swap time.

### Endpoints

| Sync | Async | Description |
|------|-------|-------------|
| `POST /v1/text-to-image` | `POST /v2/text-to-image` | Image from text |
| `POST /v1/image-to-image` | `POST /v2/image-to-image` | Single-ref edit |
| `POST /v1/image-edit` | `POST /v2/image-edit` | Multi-ref edit (Klein) |

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
- 1-10 reference images. All references condition every output (style/composition blending).

**Response** (v1): raw WEBP bytes (`Content-Type: image/webp`, quality 95)

---

## Job Queue

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/jobs/{id}` | GET | Poll status + progress |
| `/v2/jobs/{id}/result` | GET | Download result (MP4/WEBP) |
| `/v2/jobs/{id}/preview` | GET | Preview JPEG (during processing) |
| `/v2/jobs/{id}` | DELETE | Cancel job |

**Status response**:
```json
{
  "job_id": "unguessable_token",
  "status": "queued | processing | completed | failed | cancelled",
  "progress": 0.5,
  "queue_position": 3,
  "result_url": "/v2/jobs/{id}/result",
  "result_media_type": "video/mp4 | image/webp",
  "error": {"code": "cuda_oom", "message": "..."}
}
```

- Max queue depth: 10 (returns 429 `Retry-After: 30`)
- Results expire after 10 minutes
- Job IDs are unguessable — no auth needed to poll/fetch

---

## System & Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status (no auth) |
| `/v1/system/pause` | POST | Free GPU VRAM for training |
| `/v1/system/resume` | POST | Reload all models |
| `/v1/loras` | GET | List LoRAs |
| `/v1/loras` | POST | Upload LoRA (multipart form) |
| `/v1/loras/{id}` | DELETE | Delete LoRA |
| `/v2/history` | GET | List past generations |
| `/v2/history/{id}/image` | GET | Download generation result |
| `/v2/history/{id}/thumbnail` | GET | 256px JPEG thumbnail |

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
| `GEMMA_VARIANT` | `default` | `default` (standard) or `sikaworld` (uncensored) |

### GPU Topology

| Device | GPU | VRAM | Model | Memory |
|--------|-----|------|-------|--------|
| cuda:0 | RTX PRO 6000 | 96GB | Flux 2 | ~18GB (Klein) / ~77GB (Dev) |
| cuda:1 | RTX PRO 6000 | 96GB | LTX-2.3 | ~69GB |
| cuda:2 | RTX PRO 4000 | 24GB | Unused | — |

Flux and LTX share a single inference lock (FP8 layerwise casting causes cuBLAS crashes with concurrent multi-GPU inference).

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
| `server.py` | FastAPI app, all endpoints, job queue |
| `split_model_manager.py` | LTX-2 pipeline, transformer swapping, VAE decode |
| `flux_manager.py` | Flux 2 pipeline, model swapping (Dev/Klein), FP8 |
| `job_queue.py` | Async job queues with per-device workers |
| `upload_store.py` | UUID file storage for uploads/results |
| `history_store.py` | SQLite generation history (per API key) |
| `lora_registry.py` | LoRA management with JSON registry |
| `chat_manager.py` | llama-swap proxy for chat/vision |
| `nvfp4_loader.py` | NVFP4→BF16 dequantizer (comfy-kitchen) |
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
