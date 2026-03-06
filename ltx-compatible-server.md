# LTX-Compatible Server API Contract

Reference documentation for implementing a custom LTX-Video inference server compatible with taco-desktop. This server replaces the official `https://api.ltx.video` endpoint.

## Authentication

All endpoints expect `Authorization: Bearer {api_key}` by default. When `customApiRequiresAuth` is disabled in taco-desktop settings, no auth headers are sent. Your server can ignore auth entirely for local network use.

## Endpoints

### POST /v1/text-to-video (REQUIRED)

Minimum viable endpoint for text-to-video generation.

**Request:**
```json
{
  "prompt": "a cat sitting on a windowsill",
  "model": "ltx-2-3-fast",
  "resolution": "1920x1080",
  "duration": 6.0,
  "fps": 24.0,
  "generate_audio": false,
  "camera_motion": "dolly_in"
}
```

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `prompt` | string | yes | |
| `model` | string | yes | `ltx-2-3-fast`, `ltx-2-3-pro` |
| `resolution` | string | yes | `1920x1080`, `1080x1920`, `2560x1440`, `1440x2560`, `3840x2160`, `2160x3840` |
| `duration` | float | yes | 6, 8, 10, 12, 14, 16, 18, 20 (depends on model/resolution) |
| `fps` | float | yes | 24, 25, 48, 50 |
| `generate_audio` | bool | yes | |
| `camera_motion` | string | no | `dolly_in`, `dolly_out`, `dolly_left`, `dolly_right`, `jib_up`, `jib_down`, `static`, `focus_shift` (omitted when "none") |

**Response (200):** Return the video directly:
```
Content-Type: video/mp4

[raw mp4 bytes]
```

OR return a JSON body with a download URL (the client will GET it):
```json
{
  "video_url": "http://your-server:8080/outputs/abc123.mp4"
}
```
The client checks these keys in order: `video_url`, `output_video`, `output_video_url`, `output_url`, `url` (also nested under `result`).

**Timeout:** Client waits up to 1200 seconds (20 minutes).

**Recommended:** Return raw video bytes with `Content-Type: video/mp4` — simplest implementation.

---

### POST /v1/image-to-video (OPTIONAL)

Same as text-to-video, with an additional `image_uri` field:
```json
{
  "prompt": "the cat jumps off the windowsill",
  "image_uri": "storage://uploaded-image-id",
  "model": "ltx-2-3-fast",
  "resolution": "1920x1080",
  "duration": 6.0,
  "fps": 24.0,
  "generate_audio": false
}
```

Requires the `/v1/upload` endpoint to be implemented (see below). The `image_uri` value comes from the upload response's `storage_uri`.

---

### POST /v1/audio-to-video (OPTIONAL)

```json
{
  "prompt": "a musician playing guitar",
  "audio_uri": "storage://uploaded-audio-id",
  "image_uri": "storage://uploaded-image-id",
  "model": "ltx-2-3-fast",
  "resolution": "1920x1080"
}
```

Note: No `duration`, `fps`, `generate_audio`, or `camera_motion` fields.
`image_uri` is optional (can be omitted/null).

---

### POST /v1/retake (OPTIONAL)

Re-generate a portion of an existing video.

```json
{
  "video_uri": "storage://uploaded-video-id",
  "start_time": 2.0,
  "duration": 4.0,
  "mode": "replace_audio_and_video",
  "prompt": "the cat looks surprised"
}
```

| Field | Type | Values |
|-------|------|--------|
| `mode` | string | `replace_audio_and_video`, `replace_video`, `replace_video_only`, `replace_audio` |
| `prompt` | string | Optional, included only if non-empty |

**Timeout:** 600 seconds. Response format same as generation endpoints.

**Note:** 422 status is interpreted as "Content rejected by safety filters" by the client.

---

### POST /v1/upload (OPTIONAL — required for image/audio/retake)

Two-step upload flow.

**Step 1 — Get upload credentials:**
```
POST /v1/upload
Authorization: Bearer {api_key}
```
Empty body. Response:
```json
{
  "upload_url": "http://your-server:8080/uploads/put/abc123",
  "storage_uri": "storage://abc123",
  "required_headers": {}
}
```

**Step 2 — Upload file bytes:**
```
PUT {upload_url}
Content-Type: image/png
{...required_headers}

[raw file bytes]
```
Return 200 or 201 on success. Timeout: 300 seconds.

**For a local server:** The simplest implementation is to have `/v1/upload` return a URL pointing back to your server (e.g., `http://localhost:8080/uploads/put/{uuid}`), store the uploaded file to disk, and use `storage://{uuid}` as the reference. Your generation endpoints then resolve `storage://` URIs to local file paths.

---

### POST /v1/prompt-embedding (NOT NEEDED)

This endpoint returns pickle-serialized PyTorch tensors for text encoding. When a custom endpoint is configured, taco-desktop forces local text encoding instead. You do not need to implement this endpoint.

---

### GET /health (RECOMMENDED)

Simple health check. taco-desktop can optionally test connectivity to your server.

```json
{
  "status": "ok"
}
```

## Response Headers

- `x-request-id`: Optional. If present, included in error messages for debugging.

## Error Handling

- Non-200 responses: Client reads up to 500 chars of response body for error messages.
- JSON error payloads: Client checks `error`, `message`, `detail` fields.
- 422 on retake: Interpreted as content safety filter rejection.

## Quick Start: Minimal Server

For a minimal text-to-video-only server, you need ONE endpoint:

```python
from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()

@app.post("/v1/text-to-video")
async def text_to_video(request: dict):
    prompt = request["prompt"]
    model = request["model"]
    resolution = request["resolution"]
    duration = request["duration"]
    fps = request["fps"]

    # Your inference code here
    video_bytes = run_ltx_inference(prompt, model, resolution, duration, fps)

    return Response(content=video_bytes, media_type="video/mp4")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Point taco-desktop's Custom API Endpoint setting at `http://{your-rtx6000-ip}:8080` and you're done.
