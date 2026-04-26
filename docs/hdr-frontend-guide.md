# `/v2/video-hdr` — Frontend integration guide (v1.14.0)

A short reference for noodle-v (and any other client) to wire up the new HDR-expansion endpoint. Mirrors the v1.7.0 outpaint guide; if you already wired up `/v2/video-outpaint` you can copy that flow almost verbatim.

## What it does

Takes an LDR video (your typical SDR rendering — Rec. 709 / sRGB-ish) and produces a same-canvas video with **expanded dynamic range** baked into the pixel values: brighter highlights, recovered shadow detail, more dimensional lighting, less clipped sky/sun/specular reflections. It does **not** add HDR10 / PQ / BT.2020 metadata to the MP4 — that's a separate post-step (see "What it doesn't do" below).

Powered by Lightricks' `LTX-2.3-22b-IC-LoRA-HDR` IC-LoRA. Architecturally identical to the v1.7.0 outpaint pipeline: source video is VAE-encoded as an in-context reference, the LoRA fuses HDR-expansion knowledge into the distilled denoiser, and stage-1 + stage-2 produce the output. The only difference vs outpaint: target dims = source dims (no canvas expansion).

## What it doesn't do

- **No HDR10/PQ/BT.2020 transfer-curve encoding.** The output MP4 is still SDR-encoded H.264 with CRF 18; the HDR-ness lives in the pixel values, not in the container metadata. If a downstream player needs true HDR signaling, post-process with ffmpeg (e.g. `-vf zscale=...:transfer=smpte2084 -c:v libx265 -x265-params hdr10`).
- **No audio passthrough.** Output is silent. If the source had audio, mux it back in client-side via ffmpeg.
- **No exact-source-resolution guarantee.** The backend snaps source dims to the nearest /64 multiple (LTX requirement). Diff is ≤32 px per axis. Pre-encode at /64-aligned dims if you need exact preservation.

## Endpoint

```
POST /v2/video-hdr
Authorization: Bearer <api-key>
Content-Type: application/json
```

### Request body

```jsonc
{
  "video_uri": "storage://<32-hex>",        // required — source LDR video
  "prompt": "string",                       // required — describes the HDR look you want
  "duration": 6.0,                          // required, 0 < x ≤ 30
  "fps": 24,                                // required, 0 < x ≤ 60

  "seed": 0,                                // optional; 0 → server picks random
  "enhance_prompt": false,                  // optional; Gemma rewriter
  "lora": null,                             // optional; null → defaults to {"id": "ic-lora-hdr", "strength": 1.0}
  "conditioning_strength": 1.0,             // optional; 0..1, scalar attention weight on the LoRA conditioning
  "skip_stage_2": false                     // optional; true = fast half-res preview
}
```

### Response

`202 Accepted` + standard async submission envelope:

```json
{
  "job_id": "abc123...",
  "status": "queued",
  "queue_position": 2
}
```

Then poll `GET /v2/jobs/{job_id}` or subscribe via `GET /v2/jobs/{job_id}/stream?token=<sse>` for live status. Result lands at `GET /v2/jobs/{job_id}/result` when status is `completed`. Same lifecycle as every other v2 video endpoint — see API.md "Jobs lifecycle" section.

## Recommended prompts

The LoRA responds to descriptions of *what the HDR look should emphasize*, not generic "make it HDR." Good shapes:

- `"preserve natural skin tones, expand highlights"` — for talking-head / portrait content.
- `"recover shadow detail, deepen contrast, keep saturation natural"` — for cinematic landscapes.
- `"bright sun-drenched, expanded specular highlights, cool shadows"` — for outdoor / midday.
- `"low-key lighting, deep blacks, preserve neon highlights"` — for night / club / neon.

Avoid: `"hdr"` alone, `"more dynamic range"`, or pure adjectives. The model wants subject context.

## Conditioning strength

`conditioning_strength` ∈ [0, 1] is the scalar attention weight on the IC-LoRA's video reference. Pragmatic ranges:

| Value | Effect |
|-------|--------|
| `1.0` (default) | Strongly faithful to source content; LoRA expands tonality without altering composition. **Start here.** |
| `0.85–0.95` | Slightly looser fidelity; can unlock more aggressive highlight roll-off and shadow lift. |
| `0.7–0.85` | LoRA reinterprets lighting more freely. Use only when source is muddy or lacks contrast. |
| `< 0.7` | Source becomes a soft suggestion; output quality varies wildly. Not recommended. |

## `skip_stage_2` — fast preview AND closest-to-canonical output

`skip_stage_2: true` skips the upsample + refine stage. Two reasons to use it:

1. **Speed**: ~half the total latency, half the resolution.
2. **Authenticity**: upstream's reference `ICLoraPipeline` (April-13 ltx-pipelines sync) drops the IC-LoRA for stage 2 — meaning canonical HDR-LoRA inference is **stage-1-only with the LoRA active**. Our `_run_outpaint` keeps the LoRA fused through stage 2 (cache-key constraint that would cost ~30 s to reload it). For HDR specifically — where the LoRA was trained against upstream behavior — `skip_stage_2: true` produces output closer to the LoRA author's intent. Stage-2-on may slightly over-drive highlights or saturation in the upsampled result.

Recommendation: ship two preview buttons in your UI — "Fast HDR Preview" (skip_stage_2=true) and "Final HDR" (skip_stage_2=false), let the user A/B.

## Latency

Approximate, on cuda:0 LTX (no Modal/RunPod):

| Source duration / size | `skip_stage_2=true` | `skip_stage_2=false` |
|---|---|---|
| 3 s @ 720p | ~12 s | ~25 s |
| 5 s @ 1080p | ~20 s | ~40 s |
| 6 s @ 1080p | ~24 s | ~50 s |

Turbo mode (cuda:0 + cuda:1 sidecar) ~halves these.

## Defaults summary

If the client sends only `video_uri` + `prompt` + `duration` + `fps`, the backend fills in:

- `lora: {"id": "ic-lora-hdr", "strength": 1.0}`
- `seed: <random 32-bit>`
- `enhance_prompt: false`
- `conditioning_strength: 1.0`
- `skip_stage_2: false`
- `width / height: <source dims snapped to nearest /64>`

## Error responses

| Status | Body | When |
|---|---|---|
| `400` | `{"error": "missing_lora", "message": "lora not found: <id>"}` | Custom `lora.id` doesn't exist in registry |
| `404` | `{"error": "video_not_found"}` | `video_uri` doesn't resolve to a real upload |
| `422` | Pydantic field errors | Invalid duration/fps/conditioning_strength |
| `422` | `{"error": "video_probe_failed", "message": "<detail>"}` | Source MP4 has no video stream / is corrupted |
| `500` | `{"error": "HDR LoRA resolve returned None — registry misconfigured"}` | Default `ic-lora-hdr` not registered. Run `bash scripts/register_hdr_lora.sh` on the host. |

## Curl example

```bash
JOB=$(curl -s -X POST "$API/v2/video-hdr" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "video_uri": "storage://abc123",
    "prompt": "expand highlights and shadow detail, preserve natural skin tones",
    "duration": 6,
    "fps": 24
  }' | jq -r .job_id)

# Stream live status
TOKEN=$(curl -sX POST "$API/v1/sse-token" -H "Authorization: Bearer $KEY" | jq -r .token)
curl -s "$API/v2/jobs/$JOB/stream?token=$TOKEN"

# Or poll
while true; do
  STATUS=$(curl -s "$API/v2/jobs/$JOB" -H "Authorization: Bearer $KEY" | jq -r .status)
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && exit 1
  sleep 2
done

# Download result
curl -s "$API/v2/jobs/$JOB/result" -H "Authorization: Bearer $KEY" -o output-hdr.mp4
```

## Comparison vs `/v2/video-outpaint`

| | outpaint (v1.7.0) | hdr (v1.14.0) |
|---|---|---|
| LoRA | `ic-lora-outpaint` | `ic-lora-hdr` |
| `target_resolution` | client-supplied, can be > source | derived from source (snapped to /64) |
| `position` | 9 placement values | always `center` (server-set) |
| Canvas | larger than source (letterboxed) | same as source |
| Output | silent MP4, source content + LoRA-painted padding | silent MP4, same canvas, expanded dynamic range |
| `conditioning_strength` | low values loosen LoRA fidelity in fill regions | low values let LoRA reinterpret lighting more freely |
| `skip_stage_2` | speed escape hatch | speed + closest-to-canonical-LoRA-output |

Both endpoints share the same `_run_outpaint` pipeline server-side — the only difference is the request body shape and the LoRA file resolved.
