# Retake Tab — Frontend Integration Guide

**Applies to:** taco-backend v1.7.0 (2026-04-17) and later. See [docs/API.md](./API.md) for the full canonical reference. Related: [outpaint-frontend-guide.md](./outpaint-frontend-guide.md).

**Public base URL:** `https://api.noodlefinger.io` (canonical, Cloudflare-proxied). For local dev, use `http://localhost:8090`. `taco.noodlefinger.io` was retired 2026-04-18.

## What it does

Regenerate a section of an existing video. Pick a time window, choose what to replace (video, audio, or both), optionally change the prompt. Everything outside the window stays untouched.

## Endpoint

```
POST /v2/retake
Authorization: Bearer <api-key>
Content-Type: application/json
```

## Request

```json
{
  "video_uri": "storage://abc123",
  "start_time": 2.0,
  "duration": 3.0,
  "mode": "replace_audio_and_video",
  "prompt": "a person dancing energetically",
  "lora": { "id": "audioreactivev1", "strength": 0.8 }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `video_uri` | string | yes | `storage://` URI from a previous generation's `result_uri` |
| `start_time` | float | yes | Seconds, >= 0 |
| `duration` | float | yes | Seconds, > 0, max 30 |
| `mode` | string | yes | See modes below |
| `prompt` | string | no | New prompt for the regenerated section. Omit to reuse original context |
| `lora` | object | no | `{ "id": "lora-id", "strength": 0.0-1.0 }` |

### Modes

| Mode | Video | Audio | Use case |
|------|-------|-------|----------|
| `replace_audio_and_video` | Regenerated | Regenerated | Full redo of a section |
| `replace_video` | Regenerated | Preserved | Fix visuals, keep audio |
| `replace_video_only` | Regenerated | Preserved | Alias for above |
| `replace_audio` | Preserved | Regenerated | Fix audio, keep visuals |

## Response

```json
{
  "job_id": "abc123",
  "status": "queued",
  "poll_url": "/v2/jobs/abc123",
  "stream_url": "/v2/jobs/abc123/stream"
}
```

## Poll / Stream

Same as all other v2 jobs:

```
GET /v2/jobs/{job_id}          → { status, progress, phase }
GET /v2/jobs/{job_id}/stream   → SSE (preferred)
GET /v2/jobs/{job_id}/result   → video/mp4 download
```

Phases: `denoising` → `decoding` → `encoding` → `saving`

## UI Flow

1. User picks a video from history (has `result_uri`)
2. Show timeline scrubber — user drags to select `[start_time, start_time + duration]`
3. Mode selector: radio buttons for the 4 modes
4. Optional: prompt text field (pre-fill from original gen's prompt if available)
5. Submit → poll/stream → show result
6. Side-by-side: original vs retake

## Example

```javascript
const resp = await fetch('/v2/retake', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    video_uri: selectedVideo.result_uri,
    start_time: scrubber.start,
    duration: scrubber.end - scrubber.start,
    mode: 'replace_audio_and_video',
    prompt: promptInput.value || undefined,
  }),
});
const { job_id, stream_url } = await resp.json();

// SSE for live progress
const es = new EventSource(`${stream_url}?token=${apiKey}`);
es.onmessage = (e) => {
  const { status, progress, phase } = JSON.parse(e.data);
  updateProgressBar(progress, phase);
  if (status === 'completed' || status === 'failed') es.close();
};
```

## Constraints

- Max duration: 30 seconds per retake
- Resolution: matches source video (no resize)
- FPS: matches source video
- The source `video_uri` must be a valid `storage://` URI (from upload or previous gen)
- Retake is single-stage (no upsampling) — quality matches the source

## See also

- **[outpaint-frontend-guide.md](./outpaint-frontend-guide.md)** — expand a video's canvas rather than replace a window (v1.7.0)
- **[QUICKSTART.md](./QUICKSTART.md)** — base SSE / upload / auth flow shared across all v2 endpoints
- **[API.md](./API.md)** — canonical reference for every endpoint, status code, and pydantic shape
