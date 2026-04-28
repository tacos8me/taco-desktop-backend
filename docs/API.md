# taco-backend — Complete API Reference

**Server version:** v1.16.1 (2026-04-28)
**Public base URL:** `https://api.noodlefinger.io` *(canonical; Cloudflare-proxied via the shared `noodle` tunnel)*
**LAN / dev base URL:** `http://<host>:8090` (uvicorn direct)

> **Deprecation note (v1.8.1):** `https://taco.noodlefinger.io` was retired as of 2026-04-18 — DNS no longer resolves. All traffic must use `api.noodlefinger.io`.
**Source of truth:** `/mnt/nvme-1/servers/taco-backend/server.py`. This document is the client contract — any commit that adds, removes, or changes an endpoint (URL, method, request/response shape, status codes, auth) MUST update this file in the same commit.

> **New to the API?** Start with [docs/QUICKSTART.md](./QUICKSTART.md). This doc is the exhaustive spec.
> **See also:** [GPU architecture](./gpu-architecture.md) · [Model specs](./models.md) · [Configuration](./configuration.md) · [Operator tuning](./operator-tuning.md) (rate-limit + concurrency env vars, v1.16.1)

---

## Experimental — v1.12 segment chain conditioning

> **Status:** experimental, opt-in. Ships in v1.12.0 (2026-04-20). Recommended for new MusicVideo compositions; gate FE UI behind `flags.v112_seamless_segment` until visually regression-tested in production. The v1.11.5 3-PNG-keyframes path is fully supported and remains the default for clients that do not opt in.

v1.12 adds **multi-frame video-segment chain conditioning** — a proper fix for the "subject drift at seam 2+" artifact in the v1.10–v1.11.5 keyframes flow. The new surface is a *single optional field* on existing endpoints plus one new helper endpoint. Non-opted-in clients see no behavioral change.

| Surface | What it is |
|---|---|
| [`POST /v2/video/extract-segment`](#post-v2videoextract-segment-v1120-experimental) | Extract a contiguous 9-pixel-frame (or 17/25/33) MP4 segment from a stored video upload. Mirrors the shape/security of `/v2/video/extract-frames`. |
| `AudioToVideoRequest.segment_uri`, `ImageToVideoRequest.segment_uri` | Optional `string \| null`. 3-way mutually exclusive with `image_uri` and `keyframes`. Backend VAE-encodes the segment as a multi-latent-frame tensor and hard-pins 9 consecutive target pixel frames at every sigma step. |
| Composition clip `segmentUri` | Audit/re-export trail on the clip that *produced* the tail segment. Backend does not re-extract from this field. |
| Composition root `chainMode: "seamless-segment"` | New enum value. Distinguishes v1.12 compositions from legacy `"seamless"` (v1.11.5 keyframes) and `"hardcut"`. |
| Composition clip `tailTrimFrames: 9` | Drop 9 frames off the end of each non-final clip (was 3/6 in v1.11.x). |

**Why this exists.** v1.10–v1.11.5 attempted chain conditioning with 3 PNG frames at `frame_index=[0, 1, 2]` strength 1.0. LTX can only hard-pin pixel frame 0 that way (via `VideoConditionByLatentIndex`); frames 1–2 are soft-guided context tokens. Each new clip's "anchor" was a free-generated frame from the prior clip, so subject identity drifted cliff-wise past seam 2. v1.12 encodes a 9-frame video segment as a 2-latent-frame tensor, producing a single `VideoConditionByLatentIndex(latent=segment, latent_idx=0, strength=1.0)` that pins 9 consecutive target pixel frames (LTX causal VAE: latent 0 → pixel 0, latent 1 → pixels 1–8).

**Known limitation.** The segment is re-encoded by LTX's causal VAE, which replicates frame 0 for causal padding — so the 9-frame-segment latents are not bit-identical to what the prior clip's full-length encoding produced. Expected residual RMSE: 1–3 on a 0–255 scale. Visually imperceptible in most content. The v1.13 candidate (re-use the prior clip's saved final latents) eliminates this entirely.

**Full frontend spec:** [docs/handover-frontend-v1.10-chain.md](./handover-frontend-v1.10-chain.md) top section. Three-sentence summary of the flow: on clip N completion, call `/v2/video/extract-segment` with `start_frame = num_frames - 9` and `num_frames = 9`; submit clip N+1's a2v/i2v request with `segment_uri = <returned>` (omit `image_uri` and `keyframes`, the validator rejects mixed modes); on composition save, use `tailTrimFrames: 9` on non-final clips and `chainMode: "seamless-segment"` at the composition root.

---

## Table of contents

- [Conventions](#conventions) — auth, error shape, storage URIs, CORS, media types, rate limiting
- [Common types](#common-types) — `LoRAInput`, `KeyframeInput`, `Resolution`, `ModelName`, `ImageModelName`, `RetakeMode`, `OutpaintPosition`, shared constraints
- [System & health](#system--health)
  - [`GET /health`](#get-health) · [`GET /dashboard`](#get-dashboard) · [`GET /v1/system/gpu`](#get-v1systemgpu)
  - [`POST /v1/system/pause`](#post-v1systempause) · [`POST /v1/system/resume`](#post-v1systemresume)
  - [`POST /v1/flux/unload`](#post-v1fluxunload) · [`POST /v1/flux/reload`](#post-v1fluxreload) · [`POST /v1/ltx/unload`](#post-v1ltxunload) · [`POST /v1/ltx/reload`](#post-v1ltxreload)
  - [`POST /v1/system/turbo`](#post-v1systemturbo) · [`GET /v1/system/pool`](#get-v1systempool) · [`POST /v1/system/pool/remote-workers`](#post-v1systempoolremote-workers)
  - [`GET/POST /v1/system/config`](#get-v1systemconfig) · [`POST /v1/system/config/reset`](#post-v1systemconfigreset)
  - [`GET/POST /v1/system/flux-config`](#get-v1systemflux-config) · [`POST /v1/system/flux-config/reset`](#post-v1systemflux-configreset)
  - [`GET/POST /v1/system/sampler`](#get-v1systemsampler)
- [Uploads](#uploads) — [`POST /v1/upload`](#post-v1upload) · [`PUT /uploads/put/{upload_id}`](#put-uploadsputupload_id) · [`GET /uploads/get/{upload_id}`](#get-uploadsgetupload_id-v191)
- [LoRA registries](#lora-registries) — [LTX](#ltx-loras) · [Flux](#flux-loras)
- [v1 sync generation](#v1-sync-generation)
  - [`POST /v1/text-to-video`](#post-v1text-to-video) · [`POST /v1/image-to-video`](#post-v1image-to-video) · [`POST /v1/audio-to-video`](#post-v1audio-to-video) · [`POST /v1/retake`](#post-v1retake)
  - [`POST /v1/text-to-image`](#post-v1text-to-image) · [`POST /v1/image-to-image`](#post-v1image-to-image) · [`POST /v1/image-edit`](#post-v1image-edit)
  - [`POST /v1/music`](#post-v1music)
- [v2 async generation](#v2-async-generation)
  - [Submission envelope](#submission-envelope)
  - [`POST /v2/text-to-video`](#post-v2text-to-video) · [`POST /v2/image-to-video`](#post-v2image-to-video) · [`POST /v2/audio-to-video`](#post-v2audio-to-video) · [`POST /v2/retake`](#post-v2retake)
  - [`POST /v2/video-outpaint`](#post-v2video-outpaint) **(v1.7.0)**
  - [`POST /v2/video-hdr`](#post-v2video-hdr) **(v1.14.0)**
  - [`POST /v2/text-to-image`](#post-v2text-to-image) · [`POST /v2/image-to-image`](#post-v2image-to-image) · [`POST /v2/image-edit`](#post-v2image-edit) · [`POST /v2/music`](#post-v2music)
- [Jobs lifecycle](#jobs-lifecycle) — [`GET /v2/jobs/{id}`](#get-v2jobsjob_id) · [`GET /v2/jobs/{id}/stream`](#get-v2jobsjob_idstream) · [`GET /v2/jobs/{id}/preview`](#get-v2jobsjob_idpreview) · [`GET /v2/jobs/{id}/result`](#get-v2jobsjob_idresult) · [`DELETE /v2/jobs/{id}`](#delete-v2jobsjob_id)
- [Batch scheduler](#batch-scheduler) — [`POST /v2/batch`](#post-v2batch) · [`GET /v2/batch/{id}`](#get-v2batchbatch_id) · [`GET /v2/batch/{id}/result/{index}`](#get-v2batchbatch_idresultindex) · [`DELETE /v2/batch/{id}`](#delete-v2batchbatch_id)
- [History](#history) — [`GET /v2/history`](#get-v2history) · [`GET /v2/history/{id}`](#get-v2historygeneration_id) · [`GET /v2/history/{id}/image`](#get-v2historygeneration_idimage) · [`GET /v2/history/{id}/thumbnail`](#get-v2historygeneration_idthumbnail) · [`DELETE /v2/history/{id}`](#delete-v2historygeneration_id)
- [Chat & vision](#chat--vision) — [`POST /v1/chat/completions`](#post-v1chatcompletions) · [`POST /v2/char/rank`](#post-v2charrank)
- [Approved images](#approved-images) — [`POST /v1/approved-images`](#post-v1approved-images) · [`GET /v1/approved-images`](#get-v1approved-images) · [`GET /v1/approved-images/events`](#get-v1approved-imagesevents) · [`GET /v1/approved-images/{id}/file`](#get-v1approved-imagesimage_idfile)
- [Compositions](#compositions)
- [Video utilities](#video-utilities) — [`POST /v2/video/extract-frames`](#post-v2videoextract-frames-v1100) · [`POST /v2/video/extract-segment`](#post-v2videoextract-segment-v1120-experimental)
- [SSE session tokens](#sse-session-tokens)
- [Error taxonomy](#error-taxonomy)
- [Endpoint index](#endpoint-index)
- [Curl examples](#curl-examples)
- [Changelog](#changelog)

---

## Conventions

### Authentication

```
Authorization: Bearer <api-key>
```

- Keys live in `.api_keys` on the server (one per line). Empty file ⇒ auth disabled process-wide.
- Constant-time compare (`secrets.compare_digest`) against every configured key.
- Middleware rejects with `401 {"error": "Invalid or missing API key", "message": "...", "detail": "..."}` on any mismatch.
- **No-auth endpoints (public):** `GET /health`, `GET /v1/approved-images/events` (SSE, server-filtered by api_key_hash), `GET /v2/jobs/{id}/stream` (SSE, via bearer header or `?token=<sse-token>` query param — browsers use the query param since `EventSource` cannot set custom headers).
- **Removed from public surface (v1.8.1):** `GET /dashboard` and `GET /v1/system/gpu` now require a bearer token and are ONLY served by the LAN-only admin companion on port 8099 (see `dashboard_server.py`). On the public host they respond with 401.
- **Tenancy (v1.8.1 / SEC P0-1):** every `/v2/jobs/{id}` and `/v2/batch/{id}` endpoint enforces that the caller's bearer matches the resource's owner key. Cross-tenant access returns `404 Not found` with the same shape as an unknown ID (no existence oracle). Jobs and batches created before tenancy was enforced — or under auth-disabled mode — have an empty `api_key` and remain accessible to everyone (backwards-compat). History endpoints have always been tenancy-scoped via SQL `api_key_hash` filters.
- **Admin gate (v1.8.2 / SEC P0-2):** 12 mutation endpoints — `POST /v1/system/{pause,resume,turbo,config,config/reset,flux-config,flux-config/reset,sampler,pool/remote-workers}` and `POST /v1/{flux,ltx}/{unload,reload}` — additionally require the caller's bearer to appear in `.admin_keys` (or `TACO_ADMIN_KEY`). Mismatch returns `403 admin_required`. If `.admin_keys` is empty, the server falls back to the backwards-compat bridge: every `.api_keys` entry is treated as admin and a WARN is logged at boot. Read endpoints (`GET /v1/system/{pool,config,flux-config,sampler}`) remain user-level (any valid bearer).

### Error shape

Every error response has this exact body:

```json
{"error": "<message>", "message": "<message>", "detail": "<message>"}
```

All three fields carry the same string so clients can parse whichever they already read. Filesystem paths are redacted to `"Internal server error"` before leaving the process (triggered when the message contains `/mnt/`, `/home/`, or `/tmp/`).

### Storage URIs

Uploads and generated media are referenced by `storage://<uuid>` URIs, resolved to files under `UPLOAD_DIR`. Clients never see raw paths. See [Uploads](#uploads) for the two-step upload flow. Treat `storage://` URIs as capabilities — anyone with the URI plus a valid API key can read the file.

### CORS

`allow_origin_regex = ^https?://(localhost|192\.168\.\d+\.\d+)(:\d+)?$`
`allow_methods = GET, POST, PUT, DELETE`
`allow_headers = Authorization, Content-Type`

### Media types

| Kind | Content-Type |
|---|---|
| Video generation result | `video/mp4` |
| Image generation result | `image/webp` (lossless VP8L, quality 95) |
| Preview frame (`/v2/jobs/{id}/preview`) | `image/jpeg` (quality 80) |
| Thumbnail (`/v2/history/{id}/thumbnail`) | `image/jpeg` (quality 70, 256 px wide) |
| Music result | `audio/mpeg` \| `audio/flac` \| `audio/wav` \| `audio/opus` \| `audio/aac` (mirrors request `audio_format`) |
| Chat completions | `application/json` (OpenAI-shaped) |
| SSE streams | `text/event-stream` |

### Rate limiting and queue depth

| Ceiling | Config | Overflow response |
|---|---|---|
| v2 job queue (global) | `MAX_QUEUE_DEPTH = 30` *(v1.16.1: was 10)* | `429 {"error": "queue_full"}` + `Retry-After: 30` |
| v2 job queue (per bearer) | `PER_KEY_QUEUE_CAP = 15` *(v1.16.1: was 3)* | `429 {"error": "per_key_queue_full"}` + `Retry-After: 30` |
| v2 music pending jobs (global) | `MAX_MUSIC_PENDING = 5` | `429 {"error": "music_queue_full"}` + `Retry-After: 30` |
| v2 music pending jobs (per bearer) | `PER_KEY_MUSIC_CAP = 5` *(v1.16.1: was 2)* | `429 {"error": "per_key_queue_full"}` + `Retry-After: 30` |
| Batch queue (global) | `MAX_BATCH_QUEUE_DEPTH = 5` | `429 {"error": "batch_queue_full"}` + `Retry-After: 30` |
| Batch queue (per bearer) | `PER_KEY_BATCH_CAP = 5` *(v1.16.1: was 2)* | `429 {"error": "per_key_queue_full"}` + `Retry-After: 30` |
| Items per batch | `MAX_BATCH_ITEMS = 50` | `422` from Pydantic `max_length` |
| Upload size | `MAX_UPLOAD_BYTES = 1 GiB` | `413` |
| LoRA upload size | `MAX_LORA_SIZE_BYTES = 1 GiB` | `413` |

> All four per-key / global queue caps are env-overridable. See [`docs/operator-tuning.md`](./operator-tuning.md) for the override pattern, validation steps, and a full operator-facing changelog of v1.16.1's HTTP-layer + systemd tuning.

### System-wide state

- **Paused** (`POST /v1/system/pause`): all generation returns `503 {"error": "system_paused"}` + `Retry-After: 300`. `/health` still responds. Queued jobs are cancelled with error `"System paused"`.
- **Turbo active** (`POST /v1/system/turbo`): Flux, ACE, JoyAI, ERNIE, and music endpoints return `503 turbo_mode_active`. Auto-exit is triggered when a JoyAI or music request arrives (blocks ~15 s, then serves).
- **Single shared `_inference_lock`** serializes all in-process GPU work (Flux + LTX on cuda:0 cannot run concurrently).

### GPU layout (v1.7.0)

- **cuda:0** (RTX PRO 6000 96 GB) — LTX ↔ Flux, mutually exclusive, auto-swap on dispatch.
- **cuda:1** (RTX PRO 6000 96 GB) — ACE (~18 GB, port 8001) coexisting with either JoyAI (~50 GB, port 8092) or ERNIE-Image (~33 GB, port 8094); JoyAI and ERNIE are mutually exclusive.
- **Turbo mode** claims cuda:1 for a second LTX worker + optional multi-provider remote-sidecar pool (Modal + RunPod, v1.9.0). Total capacity while turbo is on: `2 local + sum(providers[*].active)` concurrent video workers — up to `2 + 4 (modal) + 2 (runpod) = 8` with both providers configured at default max.

---

## MCP integration

An [MCP](https://modelcontextprotocol.io) server wrapping this entire API ships at [github.com/tacos8me/noodle-portal](https://github.com/tacos8me/noodle-portal) under the `mcp/` subdirectory. LLM clients (Claude Code, Cursor, Continue, Codex CLI) can call it directly as a stdio subprocess. See [docs/MCP.md](MCP.md) for the canonical reference — connection, security, usage transcripts, full tool list.

### Install one-liner (Claude Code)

```bash
claude mcp add --scope user \
  -e NOODLEFINGER_API_KEY=nf_live_sk_... \
  noodlefinger uvx -- \
  --from "git+https://github.com/tacos8me/noodle-portal#subdirectory=mcp" \
  noodlefinger-mcp
```

### What you get

- **Discovery (tier-0, anonymous):** `search_endpoints`, `get_endpoint`, `list_groups`, `list_flows`, `get_flow`, `get_changelog` — pure local JSON reads against the bundled catalog. Useful as an LLM-readable form of this very document.
- **Single-job execution (tier-1, authenticated):** `submit_job` + `wait_for_job` + `download_job_result` collapses the upload → POST /v2/* → poll → GET result loop into three tool calls.
- **`cut_music_video` orchestration (tier-1):** one tool call runs music gen + N a2v clips with seamless chain conditioning + composition + export, with checkpointed session resume on failure.

Tier-1 tools are only registered when `NOODLEFINGER_API_KEY` is set; the same binary is safe to install anonymously for docs lookup.

See [docs/MCP.md](MCP.md) for full setup, security model, and usage transcripts.

---

## Common types

### `LoRAInput`

```json
{"id": "my-style", "strength": 1.0}
```

| Field | Type | Constraint | Default |
|---|---|---|---|
| `id` | string | Must exist in the corresponding registry (`/v1/loras` for LTX video + outpaint; `/v1/flux-loras` for Flux image) | required |
| `strength` | float | `0.0 ≤ x ≤ 2.0` | `1.0` |

Flux LoRAs run in adapter mode — strength changes are free (`pipe.set_adapters`, ~0 ms). LTX LoRAs are fused permanently at load time — a strength change forces a full transformer reload (~10 s cache miss on second pass). Cache key: `(model_name, lora_path)` for Flux (strength NOT in key); `(state_name, user_lora_tuple)` for LTX.

### `KeyframeInput`

```json
{"image_uri": "storage://...", "frame_index": "first", "strength": 1.0}
```

| Field | Type | Constraint | Default |
|---|---|---|---|
| `image_uri` | string | Must be a `storage://<uuid>` resolvable via [Uploads](#uploads) | required |
| `frame_index` | int \| `"first"` \| `"middle"` \| `"last"` | Symbolic values resolved after `num_frames` is derived. Negative ints count from end (`-1` = last frame, `-12` = 12 frames before end) | `0` |
| `strength` | float | `0.0 ≤ x ≤ 1.0`. Recommended: `first=1.0`, `middle=0.5`, `last=1.0` | `1.0` |

**Resolution rules:**
- `"first"` → `0`, `"middle"` → `num_frames // 2`, `"last"` → `num_frames - 1`
- Negative int `i` → `num_frames + i`
- Duplicate resolved indices → `422 "Duplicate frame_index values after resolution"`
- Resolved index `< 0` or `>= num_frames` → `422 "Resolved frame_index ... is out of range"`

Up to **8 keyframes per request**. `422 "At most 8 keyframes are allowed"` if exceeded.

**v1.12 compatibility:** `keyframes` is now 3-way mutually exclusive with `image_uri` and the new `segment_uri` (see [experimental callout](#experimental--v112-segment-chain-conditioning)). Sending more than one triggers `422 "Specify at most one of: image_uri, keyframes, segment_uri"`. The classical 3-PNG keyframes flow is fully supported and is the default for clients that don't opt into segment mode; we recommend new MusicVideo compositions use `segment_uri` instead.

### `Resolution`

Literal string union (server.py:642). All dimensions are multiples of 64; the server snaps via `_snap_to_multiple(..., 64)` rounding up if you send a non-standard value through the internal helper.

```
"1920x1080" | "1080x1920" | "2560x1440" | "1440x2560" | "3840x2160" | "2160x3840"
```

### `ModelName` (LTX video)

```
"ltx-2-3-fast" | "ltx-2-3-pro" | "ltx-2-3-hq"
```

| Model | Pipeline | Steps | Notes |
|---|---|---|---|
| `ltx-2-3-fast` | Distilled transformer | 8 | No CFG, uses `DISTILLED_SIGMA_VALUES` |
| `ltx-2-3-pro` | Dev transformer + stage 2 dev_lora (5 steps) | 30 Euler + CFG + STG | Default for quality |
| `ltx-2-3-hq` | Dev + distilled_lora@0.25 stage 1 (15 steps res2s), stage 2 dev_lora@0.50 | 15+5 | Highest quality |

### `ImageModelName`

```
"flux2-dev" | "flux2-klein" | "joyai-edit" | "ernie-image"
```

| Model | Endpoint eligibility | Steps default | Notes |
|---|---|---|---|
| `flux2-dev` | t2i, i2i, image-edit | 50 | Full CFG, guidance `4.0` default |
| `flux2-klein` | t2i, i2i, image-edit | 4 | Distilled, `guidance_scale` ignored |
| `joyai-edit` | image-edit only | 30 | Single-image edit sidecar on cuda:1:8092. `lora` NOT supported. Exactly 1 `image_uri`. |
| `ernie-image` | text-to-image only | 50 | Baidu ERNIE 8B DiT sidecar on cuda:1:8094. Resolutions: 1024×1024, 848×1264, 1264×848, others. |

### `RetakeMode`

```
"replace_audio_and_video" | "replace_video" | "replace_video_only" | "replace_audio"
```

### `OutpaintPosition` (v1.7.0)

```
"center" | "left" | "right" | "top" | "bottom" | "top_left" | "top_right" | "bottom_left" | "bottom_right"
```

### Shared scalar constraints

| Field | Constraint | Default |
|---|---|---|
| `prompt` | ≤ 10 000 chars (`max_length=10000` in Pydantic) | required where present |
| `camera_motion` | ≤ 200 chars | `null` |
| `duration` (video) | `0 < x ≤ 30` seconds | varies |
| `fps` (video) | `0 < x ≤ 60` | `24` (a2v, elsewhere required) |
| `num_frames` (internal) | Derived as `8k + 1` nearest `duration × fps` via `_duration_to_frames` | — |
| `width` / `height` (image) | `64 ≤ x ≤ 4096`, snapped to multiples of 16 server-side | `1024` |
| `num_inference_steps` (image) | `1 ≤ x ≤ 100` | `50` (dev), `4` (klein), `30` (joyai) |
| `guidance_scale` (image) | `0 ≤ x ≤ 20` | `4.0` (Klein silently ignores, JoyAI respects) |
| `seed` | `null` → server picks a random 32-bit uint; explicit ints are used verbatim | `null` |
| `image_uris` (image-edit) | Length `1–10` for `flux2-dev` / `flux2-klein`; **exactly `1`** for `joyai-edit` | required |

### Invariants (easy to miss)

- **Frame count** is always `8k + 1` — the server rounds `duration × fps` up to the nearest value. Actual output frame count drifts ±4 frames from what the math says.
- **Resolution dims** are always multiples of 64, snapped up by `_resolution_to_dims`.
- **Flux image dims** are always multiples of 16, snapped down by `(x // 16) * 16` in the handler.
- **seed** on every video/image endpoint: if `null`/absent, server assigns `random.randint(0, 2**32 - 1)` and stores the resolved integer in history.
- **LoRA requests**: 404 is returned when the LoRA id isn't registered. LTX LoRAs resolve via `lora_registry`; Flux LoRAs resolve via `flux_lora_registry`; outpaint resolves via `lora_registry` (id defaults to `ic-lora-outpaint`).

---

## System & health

### `GET /health`

**Auth:** none.
**Response:** `200 application/json`.

```json
{
  "status": "ok" | "paused",
  "ltx": "ready" | "not_loaded" | "paused",
  "flux": "ready" | "not_loaded" | "paused",
  "ace": "enabled" | "disabled" | "paused",
  "ernie": "ready" | "enabled" | "disabled" | "paused",
  "chat": "ready" | "not_loaded",
  "queue": {"queued": 0, "processing": 0, "completed": 12, "failed": 1}
}
```

### `GET /dashboard`

**Auth:** none.
**Response:** `200 text/html` — static SPA from `dashboard.html`. Real-time GPU telemetry, turbo controls, advanced generation controls (14 tunable parameters with presets + reset).
**Errors:** `404` if the file is missing.

### `GET /v1/system/gpu`

**Auth:** none.
2-second cached `nvidia-smi` snapshot. Used by the dashboard.

```json
{
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA RTX PRO 6000",
      "memory_used_mb": 65432,
      "memory_total_mb": 98304,
      "temperature_c": 52,
      "utilization_pct": 85,
      "power_draw_w": 250.0
    }
  ],
  "turbo": false,
  "sampler": "cfg_pp",
  "gen_config": { "...": "see /v1/system/config" },
  "flux_config": { "...": "see /v1/system/flux-config" },
  "gpu0_tenant": "ltx" | "flux" | "idle",
  "gpu1_tenant": "ltx-sidecar" | "ace+joyai" | "idle"
}
```

On `nvidia-smi` failure: `{"gpus": [], "turbo": ..., "error": "..."}` (200).

### `POST /v1/system/pause`

Evict all models, cancel queued v2 jobs and batches, flip server to paused. Safe to call while paused (returns `already_paused`).

**Response:** `{"status": "paused" | "already_paused"}`
**Errors:** `500 {"error": "pause_failed", "status": "paused"}`.

### `POST /v1/system/resume`

Reloads LTX (and Flux if `LOAD_FLUX=1`).

**Response:** `{"status": "ready" | "already_running"}`
**Errors:** `500 {"error": "resume_failed", "status": "paused"}`.

### `POST /v1/flux/unload`

Unloads Flux only (LTX stays). In single-GPU swap mode this frees cuda:0 for a video request.

**Response:** `{"status": "unloaded" | "already_unloaded"}`
**Errors:** `500 {"error": "flux_unload_failed"}`.

### `POST /v1/flux/reload`

**Response:** `{"status": "loaded" | "already_loaded"}`
**Errors:** `500 {"error": "flux_reload_failed"}`.

### `POST /v1/ltx/unload`

Unloads LTX only (Flux stays). Next video request auto-reloads (~25-30 s cold).

**Response:** `{"status": "unloaded" | "already_unloaded"}`
**Errors:** `500 {"error": "ltx_unload_failed"}`.

### `POST /v1/ltx/reload`

**Response:** `{"status": "loaded" | "already_loaded"}`
**Errors:** `500 {"error": "ltx_reload_failed"}`.

### `POST /v1/system/turbo`

Claim/release cuda:1 for dual-GPU LTX. Evicts ACE/JoyAI/ERNIE/Flux on entry. Flux, ACE, JoyAI, ERNIE, music endpoints return `503` while active.

**Body:** `{"enable": true | false}` (`TurboRequest`)
**Response:**
```json
{
  "turbo": true,
  "flux_device": "cuda:0",
  "ltx_device": "cuda:0",
  "ace_status": "unloaded" | "loaded" | "disabled",
  "joyai_status": "unloaded" | "loaded" | "disabled"
}
```

**Errors:**
- `409 {"error": "already_enabled", "turbo": true}`
- `409 {"error": "already_disabled", "turbo": false}`
- `503 {"error": "system_paused"}` + `Retry-After: 300`
- `500 {"error": "turbo_toggle_failed"}`

Turbo entry ~20 s, exit ~15 s. v1.6+: turbo stacks the optional remote-sidecar pool for up to `2 + LTX_REMOTE_SIDECAR_MAX_WORKERS` (default max 4) concurrent workers.

### `GET /v1/system/pool`

Inspect the LTX remote-sidecar pool. **v1.9.0 multi-provider** — each provider (`modal`, `runpod`) has independent target/active/max counts. Legacy flat `remote_*` fields are kept as backwards-compat aliases for the `modal` provider.

```json
{
  "turbo_active": true,
  "providers": {
    "modal":  {"configured": true,  "url": "https://...modal.run",         "target": 3, "active": 3, "max": 4},
    "runpod": {"configured": true,  "url": "https://api.runpod.ai/v2/.../lb", "target": 1, "active": 1, "max": 2}
  },
  "remote_sidecar_configured": true,
  "remote_sidecar_url": "https://...modal.run",
  "remote_worker_target": 3,
  "remote_worker_active": 3,
  "remote_worker_max": 4
}
```

- `providers` — authoritative per-provider state. Each entry:
  - `configured` — the URL env var is set
  - `target` — operator's desired count; persists across turbo toggles
  - `active` — currently live worker tasks (always `0` when `turbo_active=false` — the pool is turbo-scoped)
  - `max` — upper bound (`LTX_MODAL_MAX_WORKERS` / `LTX_RUNPOD_MAX_WORKERS`)
- Flat `remote_*` fields alias the `modal` provider for pre-v1.9 clients.

Total concurrent video workers when turbo is on: `2 local + sum(providers[*].active)`.

### `GET /v1/system/workers` (v1.13.0)

Live per-worker state for dashboards. One entry per active worker task: local cuda:0 main, local cuda:1 turbo sidecar (when turbo is on), and one per active remote provider slot.

```json
{
  "turbo_active": true,
  "providers": {
    "local":  {"count": 2},
    "modal":  {"target": 4, "active": 4, "max": 10},
    "runpod": {"target": 0, "active": 0, "max": 2}
  },
  "workers": [
    {
      "id": "queue-worker",
      "provider": "local-main",
      "slot": 0,
      "status": "busy",
      "current_job": {
        "id": "op6B5rCryVy2hd423pycuA",
        "type": "audio-to-video",
        "model": "ltx-2-3-fast",
        "width": 2560, "height": 1472,
        "num_frames": 145, "fps": 24.0,
        "phase": "denoising",
        "progress": 0.62,
        "started_at": 1776796257.1,
        "elapsed_sec": 34.2,
        "current_step": 18, "total_steps": 30
      }
    },
    { "id": "turbo-worker", "provider": "local-sidecar", "slot": 0, "status": "idle", "current_job": null },
    { "id": "modal-0", "provider": "modal", "slot": 0, "status": "busy", "current_job": { ... } },
    { "id": "modal-1", "provider": "modal", "slot": 1, "status": "idle", "current_job": null }
  ]
}
```

**Worker IDs:**
- `queue-worker` — cuda:0 main worker (always present).
- `turbo-worker` — cuda:1 local sidecar (present when `turbo_active=true`).
- `modal-<N>` / `runpod-<N>` — one per spawned remote slot (`<N>` is 0-indexed position in the provider's worker-task list).

**Status:** `"busy"` when a job is currently dispatched to this worker; `"idle"` otherwise. Inferred from `job.worker_id` on in-flight jobs in the store — zero external calls to Modal/RunPod.

**`current_job`** is null when idle; when busy, live-updated via the denoiser callbacks (progress + current_step update every sigma step; phase transitions from `denoising` → `decoding` → `encoding` → `saving`).

**Typical dashboard poll cadence:** 2 s. Response size is small (~200 B per worker × 6-10 workers = ~2 KB).

Auth: bearer required. No admin gate — read-only.

### `POST /v1/system/pool/remote-workers`

Set target remote-worker counts. Clamped per-provider to `[0, providers[p].max]`. Immediate if turbo is on; stored for next turbo-enable otherwise.

**Body (two shapes accepted):**
- `{"count": 3}` — legacy v1.6 shape, scales **modal only**
- `{"modal": 3, "runpod": 1}` — per-provider targets (v1.9.0)

Unknown provider keys return `400 {"error": "unknown_provider: [...]"}`. Unconfigured providers return `400 {"error": "provider_not_configured: runpod"}`.

**Response:** same shape as `GET /v1/system/pool`, plus `"applied_now": true/false` indicating whether the live pool was scaled immediately (turbo on) or the target was just stored for next turbo-enable (turbo off).

**Errors:**
- `400 {"error": "remote_sidecar_not_configured: set LTX_MODAL_SIDECAR_URL or LTX_RUNPOD_SIDECAR_URL in .env"}`
- `400 {"error": "invalid_json"}` / `"body_must_be_object"`
- `400 {"error": "count_must_be_int"}` / `"count_must_be_nonneg"`
- `500 {"error": "pool_scale_failed: ..."}`

### `POST /v1/system/pool/remote-workers/{provider}` (v1.9.0)

Cleanest RESTful variant — scale a single provider. Path `{provider}` must be `modal` or `runpod`.

**Body:** `{"count": N}` (`PoolCountRequest`, `count >= 0`)
**Response:** same shape as `GET /v1/system/pool`, plus `applied_now`.

**Errors:**
- `400 {"error": "unknown_provider: foo"}` — path segment not in `{modal, runpod}`
- `400 {"error": "provider_not_configured: runpod"}` — URL env var empty
- `500 {"error": "pool_scale_failed: ..."}`

### `GET /v1/system/config`

Full LTX generation-config snapshot. Persisted to `.gen_config.json`, survives restart.

```json
{
  "sampler": "cfg_pp",
  "eta_stage1": 1.0,
  "eta_default": 0.0,
  "fast_stage1_steps": 8,
  "pro_stage1_steps": 30,
  "scheduler_max_shift": 2.05,
  "scheduler_base_shift": 0.95,
  "cfg_scale": 3.0,
  "stg_scale": 1.0,
  "stg_blocks": [28],
  "rescale_scale": 0.7,
  "modality_scale": 3.0,
  "stage2_sigmas": [0.85, 0.725, 0.4219, 0.0]
}
```

Defaults live in `split_model_manager._DEFAULT_GEN_CONFIG`.

### `POST /v1/system/config`

Merge-update LTX generation config. Send only the keys you want to change — unknown keys are silently ignored. Effect is immediate on the next generation request (no restart).

**Body (partial example):** `{"fast_stage1_steps": 12, "cfg_scale": 4.0}`
**Response:** `{"status": "ok", "...": "<full merged config>"}`

### `POST /v1/system/config/reset`

Restore all LTX generation params to defaults. No body.
**Response:** `{"status": "reset", "...": "<default config>"}`

### `GET /v1/system/flux-config`

Flux generation-config snapshot. Persisted to `.flux_config.json`.

```json
{
  "default_model": "flux2-dev",
  "t2i_steps": 50,
  "edit_steps": 4,
  "guidance_scale": 4.0,
  "turbo": false,
  "turbo_steps": 8,
  "turbo_guidance": 2.5
}
```

### `POST /v1/system/flux-config`

Merge-update Flux config. Same semantics as `/v1/system/config`.
**Response:** `{"status": "ok", "...": "<full merged config>"}`

### `POST /v1/system/flux-config/reset`

**Response:** `{"status": "reset", "...": "<default config>"}`

### `GET /v1/system/sampler`

Alias for the sampler subset of `/v1/system/config`.

```json
{
  "sampler": "cfg_pp",
  "eta_stage1": 1.0,
  "eta_default": 0.0,
  "stage2_sigmas": [0.85, 0.725, 0.4219, 0.0]
}
```

### `POST /v1/system/sampler`

Toggle sampler. Writes into `_gen_config` (same store as `/v1/system/config`).

**Body:** `{"sampler": "cfg_pp" | "euler", "eta_stage1": 1.0, "eta_default": 0.0, "stage2_sigmas": [...]}`
(all fields optional except `sampler`; server defaults `stage2_sigmas` to `[0.85, 0.725, 0.4219, 0.0]` when `sampler=cfg_pp` and `stage2_sigmas` is not provided)
**Response:** `{"status": "ok", "sampler": "..."}`

---

## Uploads

Two-step flow: create slot, PUT bytes.

### `POST /v1/upload`

Get an upload slot. The returned URL is scoped to bearer auth plus the unguessable UUID.

**Body:** ignored.
**Response:**
```json
{
  "upload_url": "http://<host>/uploads/put/<upload_id>",
  "storage_uri": "storage://<upload_id>",
  "required_headers": {}
}
```

Use the returned `storage_uri` in any generation request that takes an `image_uri`, `audio_uri`, `video_uri`, or `keyframes[].image_uri`.

### `PUT /uploads/put/{upload_id}`

- **Body:** raw file bytes. No multipart wrapping.
- **Max size:** `MAX_UPLOAD_BYTES = 1 GiB`.
- **Response:** `201 Created` (empty body).
- **Errors:** `413 Upload exceeds 1024MB limit` when size exceeds the cap.

### `GET /uploads/get/{upload_id}` (v1.9.1)

Read back a previously-uploaded file. Use this to hydrate media players on page reload (noodle-m MusicVideo tab) or to preview files referenced by `storage://<id>` URIs.

- **Response:** `200` with the raw file bytes. `Content-Type` is inferred from the file's magic bytes — `image/jpeg`, `image/png`, `image/webp`, `audio/wav`, `audio/mpeg`, `audio/flac`, `audio/ogg`, `video/mp4`. Unknown formats return `application/octet-stream` (browsers sniff in most cases).
- **Auth:** global bearer required. The `upload_id` is a 128-bit `uuid4` hex — unforgeable, and the ID itself is the capability (no per-key ownership scoping on uploads today).
- **Errors:**
  - `400 invalid_upload_id` — malformed ID (not UUID-shaped, path-traversal attempts).
  - `404 upload_not_found` — well-formed ID but no file on disk.
  - `500 upload_read_failed` — disk I/O error reading the magic-byte head.

**Non-goals** (deferred): no TTL / expiry, no per-key ownership enforcement, no signed URLs, no HTTP Range requests. Add selectively if needed.

---

## LoRA registries

### LTX LoRAs

Full-fat registry with upload + delete. Stored in `LORAS_DIR/<filename>.safetensors` with `registry.json` metadata.

#### `GET /v1/loras`

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

#### `POST /v1/loras`

**Status code on success:** `201 Created`.
**Content-Type:** `multipart/form-data`.

| Field | Required | Notes |
|---|---|---|
| `file` | yes | `.safetensors` file |
| `name` | yes | Human-readable display name |
| `description` | no | |
| `base_model` | no | Default `"ltx-2.3"` |
| `trigger_word` | no | |
| `strategy` | no | |

**Response:** same row shape as `GET /v1/loras` items.
**Errors:**
- `400 "Expected multipart/form-data"`
- `400 "Missing 'file' field"`
- `400 "File must be a .safetensors file"`
- `400 <validation error>` (e.g. duplicate id)
- `422 "Missing 'name' field"`
- `413 "File exceeds 1024MB limit"` (cap: `MAX_LORA_SIZE_BYTES = 1 GiB`)

#### `DELETE /v1/loras/{lora_id}`

- `200 {"deleted": true, "id": "<id>"}`
- `404 "LoRA not found: <id>"`

### Flux LoRAs

**Folder-drop model** — no upload endpoint. Files live under `FLUX_LORAS_DIR/`. The registry uses the slugified filename stem as the id. Optional sidecar `<stem>.json` adds metadata. Manage with `cp` / `rm` on the host, then `POST /v1/flux-loras/rescan`.

#### `GET /v1/flux-loras`

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

#### `POST /v1/flux-loras/rescan`

**Response:** `{"rescanned": true, "count": N}`.

---

## v1 sync generation

Synchronous endpoints block until the result media is in RAM and return it directly. Good for quick one-shots. **For anything more than a few requests, use [v2 async](#v2-async-generation).**

All v1 generation endpoints can return:

| Status | Condition |
|---|---|
| `503` | System paused, turbo mode conflict, ACE/JoyAI/ERNIE sidecar unavailable |
| `500 "Flux not enabled"` / `"Flux pipeline not loaded"` | `LOAD_FLUX=0` and request targets Flux |
| `404` | Referenced `storage://` not found, LoRA id not found |
| `422` | Keyframe bounds, LoRA-model incompatibility, joyai-edit constraint violation, retake content rejection |
| `500` | Generic failure; path-containing error messages are sanitized to `"Internal server error"` |

### `POST /v1/text-to-video`

**Body:** `TextToVideoRequest`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `prompt` | string | required | ≤ 10 000 chars |
| `model` | [`ModelName`](#modelname-ltx-video) | required | |
| `resolution` | [`Resolution`](#resolution) | required | |
| `duration` | float | required | `0 < x ≤ 30` seconds |
| `fps` | float | required | `0 < x ≤ 60` |
| `generate_audio` | bool | `false` | |
| `camera_motion` | string \| null | `null` | ≤ 200 chars; appended to prompt as `[...]` |
| `lora` | [`LoRAInput`](#lorainput) \| null | `null` | |
| `enhance_prompt` | bool | `false` | Gemma prompt-rewriter. Requires `GEMMA_VARIANT` to point at an instruction-tuned snapshot (e.g. `gemma-3-12b-it-nvfp4`, v1.16.0+); falls back to the raw prompt with a WARN log otherwise (v1.15.3 safe-fallback). |

**Response:** `200 video/mp4` raw bytes.

### `POST /v1/image-to-video`

**Body:** `ImageToVideoRequest`. Accepts **at most one** of: `image_uri` (single start frame), `keyframes` (up to 8), or `segment_uri` (v1.12 chain segment). Any two or more → `422 "Specify at most one of: image_uri, keyframes, segment_uri"`.

| Field | Type | Default | Constraint |
|---|---|---|---|
| `prompt` | string | required | ≤ 10 000 chars |
| `image_uri` | string \| null | `null` | `storage://<uuid>` (single keyframe at frame 0) |
| `image_strength` | float | `0.85` | `0.0 ≤ x ≤ 1.0`. Ignored (422) when `segment_uri` is also set explicitly. |
| `keyframes` | list\<[`KeyframeInput`](#keyframeinput)\> \| null | `null` | Max 8. v1.11.5 classical flow. |
| `segment_uri` *(v1.12, experimental)* | string \| null | `null` | `storage://<uuid>` pointing at an extract-segment MP4. Hard-pins 9 consecutive target pixel frames via a multi-latent-frame `VideoConditionByLatentIndex`. See [Experimental callout](#experimental--v112-segment-chain-conditioning). |
| `model` | [`ModelName`](#modelname-ltx-video) | required | |
| `resolution` | [`Resolution`](#resolution) | required | |
| `duration` | float | required | `0 < x ≤ 30` |
| `fps` | float | required | `0 < x ≤ 60` |
| `generate_audio` | bool | `false` | |
| `lora` | [`LoRAInput`](#lorainput) \| null | `null` | |
| `enhance_prompt` | bool | `false` | |

**Response:** `200 video/mp4`.

**Errors:** `422 "keyframes list must not be empty"`, `422 "At most 8 keyframes are allowed"`, `422 "Either image_uri or keyframes is required"` (when `allow_neither=False`, i.e. i2v requires at least one of the three conditioning modes), `422 "Specify at most one of: image_uri, keyframes, segment_uri"`, `422 "Cannot specify image_strength together with segment_uri"`, `422 "Resolved frame_index N is out of range..."`, `422 "Duplicate frame_index values after resolution"`, `404` if any `storage://` URI (including `segment_uri`) fails to resolve.

### `POST /v1/audio-to-video`

**Body:** `AudioToVideoRequest`. Conditioning input is optional (audio-only is allowed); at most **one** of `image_uri`, `keyframes`, `segment_uri` may be set.

| Field | Type | Default | Constraint |
|---|---|---|---|
| `prompt` | string | required | ≤ 10 000 chars |
| `audio_uri` | string | required | `storage://<uuid>` |
| `image_uri` | string \| null | `null` | Optional conditioning image |
| `image_strength` | float | `1.0` | `0.0 ≤ x ≤ 1.0`. First-keyframe strength when `image_uri` is set. Default changed from `0.85` → `1.0` in v1.10.1 (v1.9.5–v1.9.9 silently dropped the field; v1.10.0 wired it through but kept the 0.85 default causing regression — v1.10.1 restored 1.0). Rejected (422) when `keyframes` or `segment_uri` is set and `image_strength` is explicitly provided. |
| `keyframes` *(v1.10.0)* | list\<[`KeyframeInput`](#keyframeinput)\> \| null | `null` | Max 8. Multi-keyframe chain conditioning (v1.11.5 path). |
| `segment_uri` *(v1.12, experimental)* | string \| null | `null` | `storage://<uuid>` from `/v2/video/extract-segment`. Recommended v1.12 chain flow — see [Experimental callout](#experimental--v112-segment-chain-conditioning). |
| `model` | [`ModelName`](#modelname-ltx-video) | required | |
| `resolution` | [`Resolution`](#resolution) | required | |
| `duration` | float | `6.0` | `0 < x ≤ 30` |
| `fps` | float | `24.0` | `0 < x ≤ 60` |
| `lora` | [`LoRAInput`](#lorainput) \| null | `null` | |
| `enhance_prompt` | bool | `false` | |

**Response:** `200 video/mp4`.

**Errors:** `422 "Specify at most one of: image_uri, keyframes, segment_uri"`, `422 "Cannot specify image_strength together with keyframes or segment_uri"`, other keyframe-resolution 422s identical to i2v, `404` on any unresolvable `storage://` URI.

### `POST /v1/retake`

**Body:** `RetakeRequest`. Regenerates a span of an existing video.

| Field | Type | Default | Constraint |
|---|---|---|---|
| `video_uri` | string | required | `storage://<uuid>` |
| `start_time` | float | required | `≥ 0` seconds |
| `duration` | float | required | `0 < x ≤ 30` |
| `mode` | [`RetakeMode`](#retakemode) | required | |
| `prompt` | string \| null | `null` | ≤ 10 000 chars |
| `lora` | [`LoRAInput`](#lorainput) \| null | `null` | |

**Response:** `200 video/mp4`. On any failure: `422 "Content rejected or generation failed: <detail>"`.

### `POST /v1/text-to-image`

**Body:** `TextToImageRequest`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `prompt` | string | required | ≤ 10 000 chars |
| `model` | [`ImageModelName`](#imagemodelname) | `"flux2-dev"` | `ernie-image` routes to sidecar |
| `width` | int | `1024` | `64 ≤ x ≤ 4096`, snapped to multiples of 16 |
| `height` | int | `1024` | `64 ≤ x ≤ 4096`, snapped to multiples of 16 |
| `num_inference_steps` | int | `50` | `1 ≤ x ≤ 100`; `turbo=true` overrides to `8` |
| `guidance_scale` | float | `4.0` | `0 ≤ x ≤ 20`; `turbo=true` overrides to `2.5`. Klein silently ignores. |
| `seed` | int \| null | `null` | `null` → random 32-bit uint |
| `turbo` | bool | `false` | Flux only — adds `FLUX_TURBO_SIGMAS` schedule, forces 8 steps / guidance 2.5 |
| `lora` | [`LoRAInput`](#lorainput) \| null | `null` | Flux-registry LoRA |

**Response:** `200 image/webp` (lossless VP8L, quality 95).

**`ernie-image` specifics:** Supported resolutions include 1024×1024, 848×1264, 1264×848. `lora` ignored. Returns `503 "ernie_disabled"` if `LOAD_ERNIE=0` or sidecar unreachable.

**Errors:** `500 "Flux not enabled"` if `LOAD_FLUX=0`; `422 <flux lora error>` (e.g. LoRA-model incompat).

### `POST /v1/image-to-image`

**Body:** `ImageToImageRequest` — same fields as `text-to-image` plus required `image_uri`.

| Field | Type | Constraint |
|---|---|---|
| `image_uri` | string | required, `storage://<uuid>` |
| *all other fields same as [`/v1/text-to-image`](#post-v1text-to-image)* | | |

**Response:** `200 image/webp`.

### `POST /v1/image-edit`

**Body:** `ImageEditRequest`. Dispatches to Flux (Dev or Klein, multi-image) or JoyAI (single-image sidecar) based on `model`.

| Field | Type | Default | Constraint |
|---|---|---|---|
| `prompt` | string | required | ≤ 10 000 chars |
| `image_uris` | list\<string\> | required | Length `1–10` for `flux2-dev`/`flux2-klein`; **exactly `1`** for `joyai-edit` |
| `model` | [`ImageModelName`](#imagemodelname) | `"flux2-klein"` | `joyai-edit` routes to cuda:1 sidecar |
| `width` | int | `1024` | `64 ≤ x ≤ 4096`, snapped to 16 |
| `height` | int | `1024` | `64 ≤ x ≤ 4096`, snapped to 16 |
| `num_inference_steps` | int | `4` | `1 ≤ x ≤ 100`. JoyAI default is 30 (passed by client). |
| `guidance_scale` | float | `4.0` | Klein ignores. JoyAI respects. |
| `seed` | int \| null | `null` | |
| `lora` | [`LoRAInput`](#lorainput) \| null | `null` | Flux only; **NOT supported for `joyai-edit` (422)** |
| `preserve_identity` *(v1.8.0)* | bool | `false` | **Klein-only.** Enables identity-preservation hooks. `422 preserve_identity_klein_only` if true with any other `model`. |
| `identity_strength` *(v1.8.0)* | float | `0.5` | `0.0 ≤ x ≤ 1.0`. Ignored when `preserve_identity=false`. `0.0` is treated as a no-op even if `preserve_identity=true`. |
| `identity_mode` *(v1.8.0)* | enum | `"balanced"` | `"balanced"` \| `"faithful"` \| `"loose"`. See preset table below. |

**Response:** `200 image/webp`.

**JoyAI specifics:**
- `image_uris` must be length 1 — otherwise `422 "joyai-edit requires exactly one image_uri"`.
- Prompts are plain English. The server wraps them in `<|im_start|>user\n<image>\n{prompt}<|im_end|>\n` before dispatch.
- Returns `503 "JoyAI not enabled (LOAD_JOYAI=0)"` if `LOAD_JOYAI` is unset.
- Returns `503 "sidecar_unreachable"` if the joyai-sidecar process is down. Clients should fall back to `flux2-klein`.
- Auto-exits turbo mode if active (blocks ~15 s).

<a id="preserve-identity"></a>

**Identity preservation (v1.8.0, Klein-only):**

When `preserve_identity=true` and `model="flux2-klein"`, two training-free hooks are applied on top of Klein's standard reference-conditioned edit to hold subject/facial identity under heavier prompt deviation (ported from [capitan01R/ComfyUI-Flux2Klein-Enhancer](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer#identity-preservation-nodes)):

- **IdentityGuidance** pulls the denoised latent toward the VAE-encoded first reference inside the sampling window `[0.0, 0.5]`.
- **IdentityFeatureTransfer** registers forward hooks on the middle 25%–88% of the double-stream transformer blocks' self-attention, blending generation tokens toward reference tokens in the attention output.

`identity_mode` presets:

| Preset | Guidance mode | Transfer mode | When to use |
|---|---|---|---|
| `"balanced"` *(default)* | `adaptive` (cosine-weighted pull) | `cosine_pull` | General portraits + edits; per-region smart blending |
| `"faithful"` | `direct` (unconditional blend) | `topk_replace` (top 50%) | Keeping facial structure under strong edit prompts; may fight creative prompts |
| `"loose"` | `channel_match` (stats match) | `mean_transfer` (distribution shift) | "Same character, new pose / scene" — preserves palette and lighting, lets geometry flex |

`identity_strength` scales both hooks proportionally: `guidance = strength`, `transfer = strength × 0.3` (preserves the upstream plugin's 0.50:0.15 default ratio at `strength=0.5`). `strength=0.0` collapses to the unmodified edit path.

**Known limits:**
- The first `image_uri` is used as the identity anchor. Put the identity-defining reference first if you pass multiple.
- Klein KV runs attention over reference tokens only on step 0 (K/V are cached thereafter). IdentityFeatureTransfer therefore takes effect on step 0 only; IdentityGuidance maintains identity pressure on remaining steps via latent-space correction.
- Timing delta vs. a plain Klein edit: +50–100 ms typical at 4 steps on Blackwell (dominated by VAE-encoding the reference once + cosine similarity passes).
- Identity strength > 0.8 can over-constrain the edit prompt, producing near-copies of the reference. Start at 0.5 and tune up.

### `POST /v1/music`

**Body:** `MusicGenerationRequest`. Blocks until audio is ready.

| Field | Type | Default | Constraint |
|---|---|---|---|
| `prompt` | string | required | ≤ 10 000 chars |
| `lyrics` | string | `"[Instrumental]"` | ≤ 50 000 chars |
| `duration` | float | `60.0` | `0 < x ≤ 600` seconds |
| `audio_format` | enum | `"mp3"` | `"mp3"` \| `"flac"` \| `"wav"` \| `"wav32"` \| `"opus"` \| `"aac"` |
| `seed` | int \| null | `null` | |
| `bpm` | int \| null | `null` | `30 ≤ x ≤ 300` |
| `key_scale` | string \| null | `null` | e.g. `"C major"` |
| `time_signature` | string \| null | `null` | e.g. `"4/4"` |
| `vocal_language` | string \| null | `null` | e.g. `"en"` |
| `num_inference_steps` | int | `50` | `1 ≤ x ≤ 200` |
| `guidance_scale` | float | `7.0` | `0 ≤ x ≤ 15` |
| `shift` | float | `3.0` | `1.0 ≤ x ≤ 5.0` |
| `infer_method` | enum | `"ode"` | `"ode"` \| `"sde"` |
| `use_adg` | bool | `false` | |
| `cfg_interval_start` | float | `0.0` | `0 ≤ x ≤ 1` |
| `cfg_interval_end` | float | `1.0` | `0 ≤ x ≤ 1` |
| `batch_size` | int | `1` | `1 ≤ x ≤ 8` |
| `task_type` | enum | `"text2music"` | `"text2music"` \| `"cover"` \| `"repaint"` \| `"extract"` \| `"lego"` \| `"complete"` |
| `source_audio_uri` | string \| null | `null` | Required for cover/repaint/extract/lego/complete |
| `reference_audio_uri` | string \| null | `null` | Optional reference audio |
| `audio_cover_strength` | float | `1.0` | `0 ≤ x ≤ 1` |
| `repainting_start` | float | `0.0` | `≥ 0` |
| `repainting_end` | float \| null | `null` | |
| `repaint_mode` | enum | `"balanced"` | `"conservative"` \| `"balanced"` \| `"aggressive"` |
| `repaint_strength` | float | `0.5` | `0 ≤ x ≤ 1` |
| `track_name` | string \| null | `null` | Required for extract/lego/complete |
| `thinking` | bool | `false` | LM thinking mode |
| `sample_mode` | bool | `false` | |
| `sample_query` | string \| null | `null` | |
| `lm_temperature` | float | `0.85` | `0 ≤ x ≤ 2` |
| `lm_top_p` | float | `0.9` | `0 ≤ x ≤ 1` |

**Task-type constraints:**
- `cover` / `repaint` / `extract` / `lego` / `complete` require `source_audio_uri` → else `422 "task_type '<t>' requires source_audio_uri"`.
- `extract` / `lego` / `complete` additionally require `track_name` → else `422 "task_type '<t>' requires track_name"`.

**Response:** raw audio bytes; Content-Type derived from `audio_format` (`mp3` → `audio/mpeg`, `flac` → `audio/flac`, `wav`/`wav32` → `audio/wav`, `opus` → `audio/opus`, `aac` → `audio/aac`).

**Errors:**
- `503 "System is paused for maintenance"`
- `503 "Music generation not enabled (LOAD_ACE=0)"`
- `404 "source_audio_uri not found"` / `"reference_audio_uri not found"`
- The v1 handler auto-exits turbo if active (blocks ~15 s).

---

### `POST /v1/music/analyze`

Read-only beat / onset / RMS envelope analysis of an uploaded audio file. CPU-only, no GPU swap, no LTX/Flux interaction. Bearer-auth gated. Drives the MV editing-grammar layer (see `docs/MV_EDITING.md` §7 for the response schema).

**Body:** `MusicAnalyzeRequest`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `audio_uri` | string | required | `storage://<uuid>` for the source audio |
| `analyzer` | enum | `"librosa"` | `"librosa"` \| `"madmom"` (v1.16.0) |

**Analyzer backends:**
- `"librosa"` (default) — in-process librosa beat-track + onset-detect + RMS, ~88% accuracy on pop. Byte-identical to v1.15.x behavior for callers that omit the field.
- `"madmom"` — proxied to the madmom sidecar at `MADMOM_SIDECAR_URL` (port 8095). Higher accuracy on downbeats (~+8% on cross-genre pop), CPU-only, BSD-licensed. Requires `LOAD_MADMOM=1` in the backend env and the sidecar service running. **No silent fallback** — sidecar unreachable / 5xx surfaces as `503` so callers know they didn't get the analyzer they asked for.

**Note on `enhance_prompt`:** the IT-tuned NVFP4 Gemma variant (`GEMMA_VARIANT=gemma-3-12b-it-nvfp4`, v1.16.0) is recommended for prompt-enhancement workflows because the default PT variant produces literal continuations rather than rewritten prompts.

**Response (`200`):**
```json
{
  "bpm": 124.5,
  "beats": [0.482, 0.964, 1.446, ...],
  "downbeats": [0.482, 2.410, ...],
  "onsets": [0.482, 0.840, ...],
  "rms_envelope": [[0.0, -28.4], [0.512, -22.1], ...],
  "duration_s": 156.3,
  "confidence": 0.87,
  "analyzer_used": "madmom"   // present only on the madmom branch
}
```

**Errors:**
- `404 "audio_uri not found"` — uploads.resolve failed.
- `422` — invalid `analyzer` value (Pydantic Literal validator).
- `503 "madmom analyzer disabled (LOAD_MADMOM=0); ..."` — opt-in flag is off.
- `503 "sidecar_unreachable: madmom sidecar not running at <url>"` — sidecar process down.
- `503 "sidecar_5xx (...)"` — sidecar returned 5xx.
- `504 "sidecar_timeout: ..."` — sidecar took longer than the client timeout.
- `500 "audio analysis failed: ..."` — librosa branch only.

---

## v2 async generation

v2 endpoints return `202 Accepted` with a `job_id`. Poll [`GET /v2/jobs/{id}`](#get-v2jobsjob_id) until `status == "completed"`, then GET the result URL — or subscribe to [`GET /v2/jobs/{id}/stream`](#get-v2jobsjob_idstream) for push updates.

### Submission envelope

Every v2 submit returns this `202` body (`MusicGenerationRequest` included):

```json
{
  "job_id": "job_01H...",
  "status": "queued",
  "poll_url": "/v2/jobs/job_01H...",
  "stream_url": "/v2/jobs/job_01H.../stream"
}
```

**Shared submission errors:**
- `503 {"error": "system_paused"}` + `Retry-After: 300`
- `429 {"error": "queue_full"}` + `Retry-After: 30` (v2 depth ≥ `MAX_QUEUE_DEPTH = 10`)
- `429 {"error": "music_queue_full"}` + `Retry-After: 30` (music only; ≥ `MAX_MUSIC_PENDING = 5`)
- `503 {"error": "turbo_mode_active: ACE/JoyAI unavailable while turbo mode is enabled..."}` + `Retry-After: 10` (music only)
- `503 "Music generation not enabled (LOAD_ACE=0)"` (music only)
- Validation errors bubble up as `422` / `404` (LoRA not found, referenced URI not resolvable) from body-prep helpers before the job is enqueued.

### Endpoints

All accept the **same bodies** as their v1 counterparts.

| Endpoint | Request shape | JobType | Result media type |
|---|---|---|---|
| `POST /v2/text-to-video` | `TextToVideoRequest` | `TEXT_TO_VIDEO` | `video/mp4` |
| `POST /v2/image-to-video` | `ImageToVideoRequest` | `IMAGE_TO_VIDEO` | `video/mp4` |
| `POST /v2/audio-to-video` | `AudioToVideoRequest` | `AUDIO_TO_VIDEO` | `video/mp4` |
| `POST /v2/retake` | `RetakeRequest` | `RETAKE` | `video/mp4` |
| `POST /v2/video-outpaint` | `VideoOutpaintRequest` | `VIDEO_OUTPAINT` | `video/mp4` (silent) |
| `POST /v2/video-hdr` | `VideoHdrRequest` | `VIDEO_HDR` | `video/mp4` (silent) |
| `POST /v2/text-to-image` | `TextToImageRequest` | `TEXT_TO_IMAGE` | `image/webp` |
| `POST /v2/image-to-image` | `ImageToImageRequest` | `IMAGE_TO_IMAGE` | `image/webp` |
| `POST /v2/image-edit` | `ImageEditRequest` | `IMAGE_EDIT` | `image/webp` |
| `POST /v2/music` | `MusicGenerationRequest` | `MUSIC_GENERATION` | mirrors `audio_format` |

### `POST /v2/video-outpaint`

**NEW in v1.7.0.** Expands a source video's canvas by letterboxing it into `target_resolution` with pure-black padding, then uses the **IC-LoRA Outpaint** LoRA (default id `ic-lora-outpaint`) to fill the black regions with temporally coherent content. Async only — no v1 sync endpoint.

**Body:** `VideoOutpaintRequest`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `video_uri` | string | required | `storage://<uuid>` — source video |
| `prompt` | string | required | ≤ 10 000 chars — describes desired fill |
| `target_resolution` | [`Resolution`](#resolution) | required | Final canvas size |
| `position` | [`OutpaintPosition`](#outpaintposition-v170) | `"center"` | Where source video sits within target canvas |
| `duration` | float | required | `0 < x ≤ 30` seconds — output duration |
| `fps` | float | required | `0 < x ≤ 60` |
| `seed` | int | `0` | `≥ 0`; `0` → server picks a random 32-bit uint |
| `enhance_prompt` | bool | `false` | Gemma prompt-rewriter. Requires `GEMMA_VARIANT` to point at an instruction-tuned snapshot (e.g. `gemma-3-12b-it-nvfp4`, v1.16.0+); falls back to the raw prompt with a WARN log otherwise (v1.15.3 safe-fallback). |
| `lora` | [`LoRAInput`](#lorainput) \| null | `null` | `null` → defaults to `{"id": "ic-lora-outpaint", "strength": 1.0}`. Can point at any registered LTX IC-LoRA. |
| `conditioning_strength` | float | `1.0` | `0.0 ≤ x ≤ 1.0` — scalar attention weight on the IC-LoRA conditioning; values < 1 loosen fidelity to the source |
| `skip_stage_2` | bool | `false` | `true` → faster preview at half target resolution, lower quality |

**Response:** `202` + submission envelope. Output is **silent MP4** (no audio passthrough — deferred to v1.7.x).

**Latency** (measured):
- `skip_stage_2=true`: ~20 s for 3 s output at 1920×1080.
- Full 2-stage, 5 s output at 1920×1080: ~35–45 s.

**Known limitations:**
- **Dark-content gotcha:** the LoRA treats pure black (RGB 0,0,0) as the fill sentinel. Very dark source content (night scenes, deep shadows) can confuse this heuristic — the model may "fill" content where you wanted the source preserved. Workaround: gamma 2.0 on the source before upload, gamma 0.5 on the output after decode.
- **Default `reference_downscale_factor` is 1** — the reference latent stays at stage-1 resolution. LoRA metadata override is supported but not exposed on the request body.
- **Stage 2 keeps LoRA fused** — upstream ICLoraPipeline drops the LoRA for stage 2, but our cache-key design would cost ~30 s to reload. Accepted deviation; image quality matches upstream closely.
- `lora` must resolve in the LTX registry — `404 "LoRA not found: <id>"` if missing. If the default id `ic-lora-outpaint` isn't registered: `500 "outpaint LoRA resolve returned None — registry misconfigured"`.

**Example:**

```json
{
  "video_uri": "storage://abc-123",
  "prompt": "extend the scene naturally, matching lighting and style",
  "target_resolution": "1920x1080",
  "position": "center",
  "duration": 5.0,
  "fps": 24,
  "seed": 42,
  "enhance_prompt": false,
  "lora": null,
  "conditioning_strength": 1.0,
  "skip_stage_2": false
}
```

---

### `POST /v2/video-hdr`

**NEW in v1.14.0.** Promotes an LDR source clip to expanded dynamic range using the **IC-LoRA HDR** LoRA (default id `ic-lora-hdr`, repo `Lightricks/LTX-2.3-22b-IC-LoRA-HDR`). Async only.

Architecturally piggybacks on the outpaint pipeline with `target == source` and `position="center"` — the canvas is preserved, only pixel values change. The backend probes source video dims via PyAV and snaps to the nearest /64 multiple automatically; the client only supplies `duration` + `fps`.

**Body:** `VideoHdrRequest`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `video_uri` | string | required | `storage://<uuid>` — source LDR video |
| `prompt` | string | required | ≤ 10 000 chars — describes the HDR look (e.g. "preserve natural skin tones, expand highlights") |
| `duration` | float | required | `0 < x ≤ 30` seconds |
| `fps` | float | required | `0 < x ≤ 60` |
| `seed` | int | `0` | `≥ 0`; `0` → server picks a random 32-bit uint |
| `enhance_prompt` | bool | `false` | Gemma prompt-rewriter. Requires `GEMMA_VARIANT` to point at an instruction-tuned snapshot (e.g. `gemma-3-12b-it-nvfp4`, v1.16.0+); falls back to the raw prompt with a WARN log otherwise (v1.15.3 safe-fallback). |
| `lora` | [`LoRAInput`](#lorainput) \| null | `null` | `null` → defaults to `{"id": "ic-lora-hdr", "strength": 1.0}`. Override only when experimenting with alternate IC-LoRAs. |
| `conditioning_strength` | float | `1.0` | `0.0 ≤ x ≤ 1.0` — scalar attention weight; lower → more LoRA-driven, less faithful to source |
| `skip_stage_2` | bool | `false` | `true` → fast half-resolution preview; **also matches upstream's reference LoRA-active stage-1 behavior most closely** |

**Response:** `202` + submission envelope. Output is **silent MP4** (no audio passthrough).

**Server-side dim handling**

The backend reads `(src_w, src_h)` from the MP4's video stream then snaps each axis to the nearest 64-multiple via `_snap64(x) = max(64, ((x+32)//64)*64)`. The output canvas is the snapped pair. Diff vs source is ≤32 px per axis — visually negligible. If you need exact source dims preserved, pre-encode the source at /64-aligned resolutions.

**Known limitations**

- **Stage-2 LoRA fusion deviation** — same as `/v2/video-outpaint`. The LoRA stays fused through stage 2, whereas upstream `ICLoraPipeline` drops it. The HDR LoRA was likely trained against upstream behavior; for the closest-to-canonical output, set `skip_stage_2: true` (returns at half-resolution).
- **Output is SDR-encoded H.264** with expanded dynamic range baked into the pixel values. True HDR10 / PQ / BT.2020 metadata is **not** added to the MP4 — that's a separate ffmpeg post-step (out of scope).
- **Audio passthrough not supported** — output is silent.
- `lora` must resolve in the LTX registry — `404 "LoRA not found: <id>"` if missing. If the default id `ic-lora-hdr` isn't registered: `500 "HDR LoRA resolve returned None — registry misconfigured"` — run `bash scripts/register_hdr_lora.sh` on the host.
- 422 `video_probe_failed` if the source MP4 has no video stream or is corrupted.

**Example:**

```json
{
  "video_uri": "storage://abc-123",
  "prompt": "expand highlights and shadow detail, preserve natural skin tones",
  "duration": 6.0,
  "fps": 24,
  "seed": 0,
  "enhance_prompt": false,
  "lora": null,
  "conditioning_strength": 1.0,
  "skip_stage_2": false
}
```

**Curl smoke**

```bash
JOB=$(curl -s -X POST "$API/v2/video-hdr" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"video_uri":"storage://abc","prompt":"expand highlights","duration":6,"fps":24}' | jq -r .job_id)
curl -s "$API/v2/jobs/$JOB/stream?token=$(curl -sX POST $API/v1/sse-token -H "Authorization: Bearer $KEY" | jq -r .token)"
```

---

## Jobs lifecycle

### `GET /v2/jobs/{job_id}`

Status poll snapshot.

```json
{
  "job_id": "job_...",
  "status": "queued" | "processing" | "completed" | "failed" | "cancelled",
  "type": "text-to-video" | "image-to-video" | "audio-to-video" | "retake" | "video-outpaint" | "video-hdr" | "text-to-image" | "image-to-image" | "image-edit" | "music-generation" | "export-composition",
  "progress": 0.42,
  "phase": "denoising" | "decoding" | "encoding" | "saving" | "generating" | null,
  "queue_position": 3,
  "error": {"code": "generation_failed", "message": "..."} | null,
  "result_url": "/v2/jobs/job_.../result" | null,
  "result_storage_uri": "storage://..." | null,
  "result_media_type": "video/mp4" | "image/webp" | "audio/mpeg" | null
}
```

**Field semantics:**
- `progress` — only populated while `processing` (range `0.0–1.0`). Denoising callbacks cap at **0.90**; the top 10% is reserved for post-denoise phases. Completed jobs report `1.0`.
- `phase` — only populated while `processing`. LTX sequence: `denoising` → `decoding` (VAE) → `encoding` (ffmpeg) → `saving` (upload_store write). Flux: `denoising` → `encoding` (WEBP) → `saving`. Music jobs use `phase="generating"` for the entire ACE call. JoyAI jobs emit `phase="encoding"` for the entire ~78 s sidecar call (opaque to per-step callbacks) — render a spinner, not a moving percentage.
- `queue_position` — only populated while `queued`.
- `result_*` — only populated when `completed`.
- `error.code` — see [Error taxonomy](#error-taxonomy) for the full set.

**Errors:** `404 "Job not found"`.

### `GET /v2/jobs/{job_id}/stream`

**Server-Sent Events stream for live job state. Use this instead of polling.** One long-lived connection replaces the ~240 GETs per video job.

**Auth:** bearer `Authorization` header (programmatic clients) OR `?token=<sse-token>` query param (browsers — issue one via [`POST /v1/sse-token`](#post-v1sse-token)).

**Event format:** each `data:` line contains the same JSON snapshot as `GET /v2/jobs/{id}`. Keepalive comments (`: keepalive`) every 15 s during idle periods. Terminal states (completed / failed / cancelled) emit one final event and close the stream. If the job is evicted mid-stream: `event: error\ndata: {"error": "job_expired"}\n\n`, then close.

**Delivery semantics:**
- Emits one event immediately on connect with the current state.
- Emits again when `(status, progress rounded to 3 decimals, phase, error_code)` changes.
- Poll loop runs at 250 ms; disconnects are detected via `request.is_disconnected()`.

**Status codes:**
- `200 text/event-stream` — stream opened.
- `401 "Missing API key"` — neither bearer nor valid token.
- `404 "Job not found"` — unknown job id (before stream opens).

**Browser example:**

```js
const { job_id } = await fetch("/v2/text-to-video", { /* ... */ }).then(r => r.json());
const { token } = await fetch("/v1/sse-token", {
  method: "POST",
  headers: { Authorization: `Bearer ${KEY}` }
}).then(r => r.json());

const es = new EventSource(`/v2/jobs/${job_id}/stream?token=${token}`);
es.onmessage = (ev) => {
  const { progress, phase, status, result_url } = JSON.parse(ev.data);
  setProgress(progress);
  setPhase(phase);
  if (status === "completed") {
    fetch(result_url, { headers: { Authorization: `Bearer ${KEY}` } })
      .then(r => r.blob()).then(showResult);
    es.close();
  } else if (status === "failed" || status === "cancelled") {
    es.close();
  }
};
es.addEventListener("error", () => { /* connection dropped or server error event */ });
```

**curl example:**
```bash
curl -N -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JID/stream"
```

### `GET /v2/jobs/{job_id}/preview`

Returns a low-res preview JPEG through a 4-path decision:

| State | Response |
|---|---|
| `job.preview_bytes` cached (Flux step-end callback or worker backfill) | `200 image/jpeg` |
| Completed result; on-disk thumbnail from `history.save()` exists | `200 image/jpeg` (zero-copy `FileResponse`) |
| Completed video job; no on-disk thumbnail; lazy PyAV first-frame extraction succeeds | `200 image/jpeg` (cached on job for subsequent polls) |
| Queued / processing / no preview data | `204 No Content` — **not** `404`. Keep polling. |
| Unknown job id | `404 "Job not found"` |

Frontends must treat `204` as "no preview yet, keep polling". `404` shows up red in dev tools and confuses users.

### `GET /v2/jobs/{job_id}/result`

Download final media. Returns zero-copy `FileResponse` with `Cache-Control: no-store`.

- `200` — `Content-Type` matches `job.result_media_type`.
- `404 "Job not found"` — unknown id.
- `409 "Job result not ready"` — job not `completed`.
- `404 "Result file expired or not found"` — on-disk file missing (TTL eviction).

### `DELETE /v2/jobs/{job_id}`

Cancel a queued or processing job.

- `200 {"job_id": "...", "status": "cancelled"}` — cancel accepted.
- `404 "Job not found"` — unknown id.
- `409 "Cannot cancel a finished job"` — already completed/failed/cancelled.

Cancellation of a `processing` job is best-effort: the denoiser checks a cancel flag between steps, and post-denoise phases are cooperative.

---

## Batch scheduler

Submit multiple generation jobs as a single batch. Items execute sequentially (or 2-at-a-time in turbo mode), sorted to minimize GPU swaps — all images before all videos, Klein before Dev within images. Auto-turbo may engage if cuda:1 has been idle ≥ `AUTO_TURBO_IDLE_MINUTES` (default 15) and the batch has ≥ 2 items.

### `POST /v2/batch`

**Body:** `BatchRequest`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `items` | list\<`BatchItem`\> | required | `1 ≤ length ≤ MAX_BATCH_ITEMS (50)` |
| `items[].type` | enum | required | `"text-to-image"` \| `"image-to-image"` \| `"image-edit"` \| `"text-to-video"` \| `"image-to-video"` (note: a2v, retake, outpaint, music NOT supported in batch) |
| `items[].params` | object | required | Must validate against the matching Pydantic request model for the type |
| `priority` | enum | `"normal"` | `"normal"` \| `"high"` |
| `callback_url` | string \| null | `null` | Optional webhook; POSTed on completion with 3 retries (delays `[1, 5, 15]` s) |

**Response:** `202`
```json
{
  "batch_id": "batch_...",
  "status": "queued",
  "total": 2,
  "queue_position": 0
}
```

**Errors:**
- `503 {"error": "system_paused"}` + `Retry-After: 300`
- `429 {"error": "batch_queue_full"}` + `Retry-After: 30` (depth ≥ `MAX_BATCH_QUEUE_DEPTH = 5`)
- `400 "Invalid item type at index N: ..."`
- `422 "Validation failed at item N (<type>): <detail>"`

### `GET /v2/batch/{batch_id}`

Poll batch status + partial results.

```json
{
  "batch_id": "batch_...",
  "status": "queued" | "processing" | "completed" | "partial" | "failed" | "cancelled",
  "total": 5,
  "completed_count": 3,
  "failed_count": 0,
  "current_index": 3,
  "turbo": false,
  "results": [
    {
      "index": 0,
      "type": "text-to-image",
      "status": "completed" | "failed" | "cancelled",
      "result_uri": "storage://...",
      "result_url": "/v2/batch/batch_.../result/0",
      "media_type": "image/webp",
      "error": null,
      "elapsed_s": 3.2
    }
  ],
  "created_at": 1712345678.9,
  "started_at": 1712345679.0,
  "completed_at": null
}
```

**Status semantics:**
- `completed` — all items succeeded.
- `partial` — some items succeeded, some failed.
- `failed` — all items failed.

**Errors:** `404 "Batch not found"`.

### `GET /v2/batch/{batch_id}/result/{index}`

Download the result file for a completed batch item.

- `200 <media_type>` — `Cache-Control: no-store`, zero-copy `FileResponse`.
- `404 "Batch not found"` — unknown batch id.
- `404 "Batch item not found or not completed"` — bad index or item not yet complete.
- `404 "Result file expired or not found"` — on-disk file missing.

Result files retained for `BATCH_RESULT_TTL_SECONDS = 1800` (30 min) after batch completion; cleanup loop runs every 60 s.

### `DELETE /v2/batch/{batch_id}`

Cancel remaining items. Currently-running item (if any) will finish.

- `200 {"batch_id": "...", "status": "cancelled", "completed_count": 2, "cancelled_count": 3}`
- `404 "Batch not found"`
- `409 {"error": "batch_already_finished", "batch_id": "...", "status": "<terminal>"}` — already completed/failed/cancelled.

---

## History

Per-API-key history of completed v2 jobs, keyed by SHA-256 of the raw bearer key (keys are never stored). Thumbnails at `THUMBNAIL_DIR/thumb_<uuid>` (JPEG 256 px wide). 30-day retention. SQLite WAL mode.

### `GET /v2/history`

Query params:
- `limit` — default 50, clamped to 200.
- `offset` — default 0.
- `type` — optional filter:
  - `"image"` — any `*-image*` type
  - `"video"` — any `*-video*` or `retake` type
  - explicit JobType string, e.g. `"text-to-video"`
  - unset → all

**Response shape** (list — slimmer than the single-record shape; no `params`/`gen_config`/`seed`/`enhanced_prompt`):

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

### `GET /v2/history/{generation_id}`

Full record — includes raw request body (`params`), gen-config snapshot (`gen_config`), resolved `seed`, and `enhanced_prompt` text.

```json
{
  "id": "job_...",
  "job_type": "text-to-video",
  "prompt": "...",
  "enhanced_prompt": "A bioluminescent jellyfish..." | null,
  "model": "ltx-2-3-pro",
  "width": 1920,
  "height": 1080,
  "seed": 845210937,
  "turbo": false,
  "status": "completed",
  "created_at": 1712345678.9,
  "completed_at": 1712345745.2,
  "error": null,
  "result_url": "/v2/history/job_.../image" | null,
  "thumbnail_url": "/v2/history/job_.../thumbnail" | null,
  "params": { "...": "raw Pydantic body" },
  "gen_config": { "...": "LTX _gen_config snapshot OR Flux turbo snapshot OR null" }
}
```

**Field semantics:**
- `seed` — integer. If the client omitted it, the server auto-generated one; the stored value is what was actually used.
- `enhanced_prompt` — text of the LTX-rewritten prompt when `enhance_prompt=true`. `null` for Flux/ERNIE/JoyAI/retake/outpaint and for any LTX request where `enhance_prompt=false`.
- `params` — raw request body from `body.model_dump(mode="json")`. Preserves `storage://` URIs, `Resolution` enum string, `LoRAInput` `{id, strength}` shape, keyframe symbolic indices. Music jobs pass params through `_sanitize_params_for_history` to rewrite staged `/tmp/*` paths back to `storage://`.
- `gen_config`:
  - LTX jobs (`text-to-video`, `image-to-video`, `audio-to-video`, `retake`, `video-outpaint`, `video-hdr`) → snapshot of `_gen_config` at dispatch time (13 keys: sampler, etas, step counts, scheduler shifts, CFG/STG/rescale/modality scales, stg_blocks, stage2_sigmas).
  - Flux turbo jobs (`text-to-image`, `image-to-image`, `image-edit` with `turbo=true`) → `{"turbo_steps": 8, "turbo_guidance": 2.5}`.
  - Non-turbo Flux / ERNIE / JoyAI → `null`. Tunables live in `params`.

**Errors:** `401 "Missing API key"`; `404 "Not found"` (also returned when the entry belongs to another key — IDs can't be probed).

### `GET /v2/history/{generation_id}/image`

Full-size result. Content-Type: `video/mp4` for video types, `image/webp` otherwise.

- `200` — `FileResponse`.
- `401` — no API key.
- `404 "Not found"` — unknown entry or owned by another key.
- `404 "Result file not found"` — on-disk file missing.

### `GET /v2/history/{generation_id}/thumbnail`

`200 image/jpeg` — 256 px wide. For video jobs the thumbnail is the first frame extracted via PyAV.

- `401` — no API key.
- `404 "Not found"` — unknown entry / owned by another key / no thumbnail stored.
- `404 "Thumbnail not found"` — thumbnail file missing on disk.

### `DELETE /v2/history/{generation_id}`

Removes the history entry and its result/thumbnail files. Scoped to caller's API key.

- `200 {"ok": true}`.
- `401` — no API key.
- `404 "Not found"` — unknown OR owned by another key.

---

## Chat & vision

### `POST /v1/chat/completions`

Proxies to the external llama-swap server. OpenAI-compatible shape.

**Body:** `ChatCompletionRequest`

| Field | Type | Default | Constraint |
|---|---|---|---|
| `model` | string | `"gemma-3-12b-nvfp4"` | `llama-swap` model id |
| `messages` | list\<`ChatMessage`\> | required | Non-empty |
| `messages[].role` | string | required | `"system"` \| `"user"` \| `"assistant"` |
| `messages[].content` | string \| list | required | Plain text string, OR OpenAI multimodal array `[{"type": "text", ...}, {"type": "image_url", ...}]` |
| `temperature` | float | `0.7` | `0 ≤ x ≤ 2.0` |
| `max_tokens` | int | `512` | `1 ≤ x ≤ 8192` |

**Response:** raw OpenAI chat-completion JSON, passed through from upstream.

**Errors:**
- `500 "Chat model not loaded"` — proxy not configured.
- `422 "Messages list cannot be empty"`.
- `500 <sanitized>` on upstream failure.

### `POST /v2/char/rank`

Two-image comparison routed to the Gemma 4 31B vision model (via `CHAR_VISION_MODEL` override on llama-swap). Used by noodle-i character mode.

**Body:** `CharRankRequest`

| Field | Type | Constraint |
|---|---|---|
| `rank_image_uri` | string | Reference character image, `storage://<uuid>` |
| `generated_image_uri` | string | Generated image to compare, `storage://<uuid>` |
| `prompt` | string | Original generation prompt, ≤ 10 000 chars |

**Response:** strict-JSON body produced by the vision model:

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

Scale: 1-3 poor, 4-6 some resemblance, 7-8 good, 9-10 excellent. The system prompt (`CHAR_RANKING_PROMPT` in `server.py:3204`) enforces the JSON schema and the "only suggest edits if score < 9" rule.

**Errors:**
- `401 "Missing API key"`
- `404 <FileNotFoundError message>` — either `storage://` URI unresolvable.
- `500 "Chat model not loaded"` — llama-swap proxy not ready.
- `500 "Vision model did not return valid JSON"` — response couldn't be parsed.
- `500 "Failed to parse vision model response"` — JSON parse error in the regex-extracted block.

---

## Approved images

A per-API-key "approved feed" — noodle-i approves an image, noodle-v watches the feed.

### `POST /v1/approved-images`

**Body:**
```json
{
  "image_uri": "storage://...",
  "prompt": "...",
  "model": "flux2-dev",
  "width": 1024,
  "height": 1024
}
```

**Response:** `201 {"id": "<16 hex>", "status": "approved"}`.
**Errors:** `401 "Missing API key"`; `400 "Missing image_uri"`.

### `GET /v1/approved-images`

Query params: `limit` (default 50), `offset` (default 0).

**Response:** JSON array, per-API-key scoped (`api_key_hash` is stripped from responses).

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

### `GET /v1/approved-images/events`

**Auth:** none at the middleware layer; the handler accepts `?token=<sse-token>` OR a bearer header.

SSE stream of newly-added entries (scoped to caller's key). Heartbeat polls the manifest every 2 s via `mtime`.

```
data: {"id": "...", "image_uri": "...", "image_url": "/v1/approved-images/.../file", ...}

```

Closes on client disconnect.
**Errors:** `401 "Missing API key"` if neither token nor bearer resolves.

### `GET /v1/approved-images/{image_id}/file`

Returns the referenced file.

- `200 image/webp` (or the underlying file's mime).
- `401 "Missing API key"`.
- `404 "Not found"` — unknown id or owned by another key.
- `404 "Image file not found"` — manifest entry exists but the file was evicted.

---

## Compositions

Multi-clip composition timelines (noodle-v export pipeline). Shape is owned by `composition_store`.

### `POST /v2/compositions`

**Status code on success:** `201`.
**Body:**
```json
{
  "name": "My Cut",
  "clips": [...],
  "transitions": [...],
  "audio_uri": "storage://<upload_id>"
}
```

**Response:** the created row.
**Errors:** `401 "Missing API key"`; `400 "body_must_be_object"`.

**Persisted shape (v1.9.5):** every top-level body field except `name` is stored verbatim under `data`. `audio_uri` (MusicVideo mode) survives save → load → export. `clips` / `transitions` default to `[]` when absent. Future frontend-added fields do not require server changes.

### `GET /v2/compositions`

Query params: `limit` (default 50, clamped to 200), `offset` (default 0). Returns array scoped to caller's API key.
**Errors:** `401 "Missing API key"`.

### `GET /v2/compositions/{comp_id}`

- `200 <row>`.
- `401 "Missing API key"`.
- `404 "Composition not found"` — unknown id or owned by another key.

### `PUT /v2/compositions/{comp_id}`

**Body:** same as create.
**Response:** `{"status": "updated"}`.
**Errors:** `401 "Missing API key"`; `404 "Composition not found"`.

### `DELETE /v2/compositions/{comp_id}`

**Response:** `{"status": "deleted"}`.
**Errors:** `401 "Missing API key"`; `404 "Composition not found"`.

### `POST /v2/compositions/{comp_id}/export`

Enqueues a `JobType.EXPORT_COMPOSITION` job; returns the same `202` envelope as any v2 submit. Poll via [`GET /v2/jobs/{id}`](#get-v2jobsjob_id).

**Body (optional, v1.9.0):** `{"audio_uri": "storage://<upload_id>"}`

**Precedence (v1.9.5):** request body `audio_uri` > stored composition `audio_uri`. The export route falls back to the stored value when the body omits it, so reloading a MusicVideo composition and hitting export with no body produces the same MP4 as the original export. Pass a body `audio_uri` to override ad-hoc.

**Encoder quality knobs (v1.16.2, optional):** the export body accepts six new fields that override the encoder defaults. All fields are optional — empty body still works (and produces higher quality output than v1.16.1 thanks to the new defaults).

| Field | Type | Default | Notes |
|---|---|---|---|
| `output_encoder` | enum | `"libx264"` | One of `"libx264"`, `"libx265"`, `"libopenh264"`. Auto-falls-back to `libopenh264` only if the local ffmpeg lacks libx264/libx265. |
| `output_crf` | int 0–51 | `18` (libx264) / `22` (libx265) | Constant-rate-factor; lower = higher quality. CRF 18 is "visually transparent" for libx264. Ignored by `libopenh264` (uses bitrate instead). |
| `output_preset` | enum | `"medium"` | x264-style preset: `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow`, `placebo`. Slower = better compression at same CRF. |
| `output_profile` | enum | `"high"` | H.264 profile: `baseline`, `main`, `high`, `high10`, `high422`, `high444`. Only applies to `libx264`. |
| `output_video_bitrate` | str | unset | E.g. `"12M"`, `"8000k"`, `"5000000"`. Matches `^\d+[kMG]?$`. **Setting this on a CRF encoder (libx264 / libx265) switches the export to 1-pass ABR** (`-b:v` + `-maxrate` + `-bufsize 24M`); the CRF flag is dropped. Required for `libopenh264` since it doesn't honor CRF. |
| `output_audio_bitrate` | str | `"256k"` | Same regex as `output_video_bitrate`. v1.16.1 default was `192k`. |

Validation runs before the job is enqueued. Malformed values return `422` with a clear message (e.g. `"output_crf must be an int in [0, 51]"`).

Defaults rationale: `libx264 + CRF 18 + preset=medium + profile=high + yuv420p` is the standard "visually transparent" operating point for H.264. v1.16.1's hardcoded `libopenh264` with no flags emitted ~4-8 Mbps which produced visible blocking on 1080p+ output. The filter graph (trim / setpts / concat / xfade / atrim / `force_key_frames`) is byte-identical to v1.16.1; only the codec args changed.

Example (CLI, full quality override):
```bash
curl -X POST "https://api.example.com/v2/compositions/$COMP/export" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_uri": "storage://abc",
    "output_encoder": "libx264",
    "output_crf": 14,
    "output_preset": "slow",
    "output_profile": "high10",
    "output_audio_bitrate": "320k"
  }'
```

When `audio_uri` is a valid `storage://` URI pointing at an audio file, the exporter adds it as an extra ffmpeg input, maps it as the output audio track, and truncates to the video length via `-shortest`. Leave empty / omit body and the composition has no stored `audio_uri` for video-only export (pre-v1.9.0 behavior, unchanged).

- The audio file must already exist as a user upload (`PUT /uploads/put/{id}` then reference via `storage://<id>`). Any format ffmpeg supports (WAV, MP3, AAC, FLAC, OGG) works.
- Single-clip + audio works — the single clip is piped through a null video filter and muxed with the audio track.
- Audio longer than the video is trimmed (`-shortest`). Audio shorter than the video will truncate the output video to the audio length.

**Per-clip audio segmentation (v1.9.3, MusicVideo mode):** when every clip in the stored composition carries a numeric `audioStart` field (seconds into the source song where that clip's audio window begins), the exporter slices the song via `atrim` per clip and concatenates the slices 1:1 with the video — so audio stays beat-aligned even across LTX's `8k+1` frame-count quantization. Trigger requires: `audio_uri` set, **every** clip has `audioStart: <number>` (bools rejected), and the composition has no xfade transitions. Falls back to the legacy full-song overlay otherwise (no behavior change for timeline-mode compositions).

Clip shape (stored in `POST /v2/compositions` body, read back at export time):
```jsonc
{
  "historyId":        "h_abc",
  "sequenceIndex":    0,
  "duration":         2.042,
  "audioStart":       0.0,
  "tailTrimFrames":   9,                   // v1.12 segment chain: 9 on non-final clips
  "audioDurationSec": 2.0,                 // v1.11.2 (unchanged)
  "segmentUri":       "storage://...",     // v1.12 NEW: segment extracted from this clip's tail
  "fps":              24
}
```

Composition-root field (applies to the whole composition, not per-clip):

```jsonc
{ "chainMode": "seamless-segment" }        // v1.12 flag; new valid value added
```

Valid `chainMode` values (v1.12):
- `"hardcut"` — no chain conditioning; full-clip cuts. Byte-identical export to v1.9.9. Unchanged.
- `"seamless"` — v1.11.5 legacy 3-PNG-keyframes path. Fully supported — keep loading/saving existing compositions verbatim.
- `"seamless-segment"` *(v1.12, experimental)* — segment chain conditioning. New compositions should emit this once the `flags.v112_seamless_segment` FE flag is on.

Backend does NOT validate `chainMode` at write time — `POST /v2/compositions` passes the whole body through `data`. The value is inspected only by the FE during rechain / re-submit flows. Export (ffmpeg pipeline in `export_handler.py`) is abstract over `chainMode` — it sums `tailTrimFrames` per clip regardless of the flag value.

Per-clip field semantics:

- `duration` — used verbatim as the `atrim duration=` for beat-gap audio slicing **only when `audioDurationSec` is absent**. Keep it in sync with the actual generated clip length (LTX outputs 8k+1 frames).
- `audioStart` — unvalidated (numeric range is the frontend's contract). Negative values or values past the song length will fail ffmpeg with a normal filter-graph error.
- `tailTrimFrames` (v1.10.0+, default 0) — number of frames to drop from the END of this clip at export time. The right value depends on which chain mode you're using:

  | Mode | Recommended `tailTrimFrames` (non-final clips) | Why |
  |---|---|---|
  | `"hardcut"` | `0` | No chain conditioning; every frame is shown. |
  | `"seamless"` (v1.11.5, keyframes) with `audioDurationSec` | `6` | Drops safe tail [N-6..N-4] + unsafe tail [N-3..N-1]. The 3 safe-tail frames are regenerated as the follower's head keyframes. `audioDurationSec` prevents the 208 ms audio dropout that tail=6 would cause with the v1.11.1 clamp. |
  | `"seamless"` (v1.11.5, keyframes) without `audioDurationSec` | `3` | v1.11.1 legacy — the audio atrim clamps to `effective_duration`, so tail=6 would drop 208 ms of song; tail=3 splits the artifact 83 ms/83 ms. |
  | `"seamless-segment"` *(v1.12)* | `9` | The follower regenerates pixel frames [N-9..N-1] via segment conditioning. All 9 frames are cleanly replaced in playback; no repeat, no backward jump. Requires `audioDurationSec` for the matching full-song-continuity export. |

  Ignored on the final clip, single-clip exports, and compositions using xfade transitions. Over-trim (`tailTrimFrames >= declared_frames`) clamps to `declared_frames - 1` with a WARN. Backward compat: field omitted or `0` → byte-identical export to v1.9.9.

- `audioDurationSec` (v1.11.2+, optional) — explicit per-clip audio slice duration in seconds, decoupling the audio atrim from the video `effective_duration`. When present and `> 0`, the exporter uses it verbatim as the per-clip `atrim duration=`; when absent it falls back to `min(beat_gap, effective_duration)` (v1.11.1 clamp). Frontend computes this as `next.beatTime - this.beatTime` for non-final clips and `duration` for the final clip. Using this field with `tailTrimFrames=6` (v1.11.5) or `tailTrimFrames=9` (v1.12) gives seamless chain video + full-song audio; the trade-off is a progressive video-cut-before-beat drift of `audioDurationSec - effective_duration` per seam, accumulating over N clips. When audio duration exceeds video duration, the exporter omits `-shortest` so the last video frame freezes briefly while the song tail completes (preferred over truncating the song). v1.12 `tailTrimFrames=9` increases the per-seam drift vs. v1.11.5 tail=6 (~333 ms per seam on 49-frame clips vs. ~208 ms); for 3+ clip compositions prefer 97-frame (4.04 s) clips, which keep the ratio under control.

- `segmentUri` *(v1.12, experimental, optional)* — `storage://<uuid>` pointing at the MP4 segment that was extracted from THIS clip's tail (via [`POST /v2/video/extract-segment`](#post-v2videoextract-segment-v1120-experimental)) and passed to the **follower** clip's `segment_uri` for chain conditioning. Purely informational for audit / re-export — the backend does NOT re-extract from this field during export, and the export pipeline ignores it. FE uses it for re-chain operations or when re-exporting a saved composition from scratch. Absent on the final clip (no follower) and on `"hardcut"` / `"seamless"` compositions.

- `fps` (v1.10.0+, default 24) — per-clip fps override for the `effective_durations` math above (backend cascades `tailTrimFrames / fps` into beat-gap atrim + force-IDR seam timestamps). LTX is 24 today; include this for future models at different rates.

**Errors** (surfaced as the job's terminal `error` field):
- `"Audio not found: storage://..."` — the URI doesn't resolve to a file on disk (UploadStore raises `FileNotFoundError`)
- `"Invalid storage URI: <...>"` — URI doesn't start with `storage://`
- `"FFmpeg failed: ..."` — ffmpeg exit non-zero (last 5 lines of stderr echoed, truncated to 300 chars)

---

## Video utilities

### `POST /v2/video/extract-frames` (v1.10.0)

> **v1.12 note:** this endpoint is the *legacy* chain helper — it extracts per-frame PNGs for the v1.11.5 3-keyframes path. New compositions should prefer [`POST /v2/video/extract-segment`](#post-v2videoextract-segment-v1120-experimental), which produces a single MP4 segment that conditions 9 consecutive target pixel frames (vs. pinning only pixel frame 0 with the PNG path).

Server-side PyAV helper that extracts specific frame indices from a stored MP4 and re-saves each as a lossless PNG upload. Designed to power the multi-frame chain conditioning flow in noodle-v's MusicVideo export (the last N frames of clip i become the head keyframes of clip i+1, eliminating visible seams at composition boundaries).

**Auth:** bearer required.

**Body:**
```json
{
  "video_uri": "storage://<32 hex>",
  "frame_indices": [43, 44, 45]
}
```

- `video_uri` — must match `^storage://[0-9a-f]{32}$` (Pydantic enforced).
- `frame_indices` — 1..16 non-negative integers. Server sorts + dedupes before decode.

**Response (200):**
```json
{
  "frames": [
    {"frame_index": 43, "storage_uri": "storage://...", "width": 1920, "height": 1080},
    {"frame_index": 44, "storage_uri": "storage://...", "width": 1920, "height": 1080},
    {"frame_index": 45, "storage_uri": "storage://...", "width": 1920, "height": 1080}
  ]
}
```

Frames are returned in sorted-index order. The `storage_uri` values are freshly-minted uploads — fetch them via `GET /uploads/get/{id}` (v1.9.1), feed them back as `KeyframeInput.image_uri` on the next clip's generate request.

**Output format:** PNG (lossless, `compress_level=6`). JPEG would inject compression artifacts that then propagate through the next clip's VAE encoding; PNG keeps the chain reference pixel-accurate.

**Errors:**
- `404 video_not_found` — `video_uri` doesn't resolve to a file (unknown id or evicted).
- `422 frame_index_out_of_range: [...]` — one or more requested indices exceed the stream length.
- `504 pyav_timeout` — decode exceeded the 30 s timeout (malformed / very long file).
- `500 extract_failed` — PyAV raised during decode, or no video stream present.
- `429 upload_quota_exceeded` — emitted PNG bytes would push the caller past `PER_KEY_UPLOAD_BYTES_PER_DAY`.
- `401 "Invalid or missing API key"` — middleware.

**Quota:** Total PNG bytes count against the per-key 24h rolling upload byte budget (same pool as `PUT /uploads/put/{upload_id}` and `POST /v1/loras`).

**Concurrency:** A dedicated `asyncio.Semaphore(2)` caps simultaneous extracts (separate instance from the preview-extract pool, so long extracts can't starve preview polling on running jobs). Extra callers queue until a slot is free.

**Security model:** Capability URL — bearer unlocks the endpoint and attributes quota, but the returned `storage_uri` values are not scoped to the caller's key. Any bearer holder who knows the upload id can fetch it via `GET /uploads/get/{id}`. Matches the v1.9.1 upload-get pattern. The 32-hex uuid4 id is the capability.

**Use case (v1.11.5 legacy chain conditioning):** After clip N finishes, extract indices `[N_frames-6, N_frames-5, N_frames-4]` (safe zone, well clear of stage-2 tail-artifact region). Submit clip N+1 with `keyframes=[{image_uri:f0, frame_index:0, strength:1.0}, {image_uri:f1, frame_index:1, strength:1.0}, {image_uri:f2, frame_index:2, strength:1.0}]`. Note that LTX only hard-pins pixel frame 0 via `VideoConditionByLatentIndex` — frames 1 and 2 are soft-guided context tokens, which drives the subject drift the v1.12 segment path addresses. See [`docs/handover-frontend-v1.10-chain.md`](./handover-frontend-v1.10-chain.md) v1.11.2 section for the legacy flow, or the v1.12 top section for the recommended segment flow.

### `POST /v2/video/extract-segment` (v1.12.0, experimental)

> **Experimental v1.12 feature.** See the [top-of-file callout](#experimental--v112-segment-chain-conditioning). Opt in via `flags.v112_seamless_segment`. The legacy [`POST /v2/video/extract-frames`](#post-v2videoextract-frames-v1100) stays fully supported.

Server-side PyAV helper that extracts a contiguous range of pixel frames from a stored MP4 and re-encodes the range as a **standalone small H.264 MP4 upload**. Designed as the v1.12 chain-conditioning input: instead of three PNG frames (which only hard-pin pixel frame 0), a single segment MP4 is VAE-encoded as a multi-latent-frame tensor that hard-pins 9 consecutive target pixel frames at every sigma step.

**Auth:** bearer required. Same capability-URL security model as [`/v2/video/extract-frames`](#post-v2videoextract-frames-v1100) and [`/uploads/get/{id}`](#get-uploadsgetupload_id-v191) — the 32-hex uuid4 in `segment_uri` is the capability; any bearer holder who knows the id can fetch it.

**Body:**
```json
{
  "video_uri": "storage://<32 hex>",
  "start_frame": 40,
  "num_frames": 9
}
```

| Field | Type | Default | Constraint |
|---|---|---|---|
| `video_uri` | string | required | `^storage://[0-9a-f]{32}$` (Pydantic enforced) |
| `start_frame` | int | required | `>= 0`. `start_frame + num_frames` must fit within the source video's frame count |
| `num_frames` | int | `9` | **Must be one of `{9, 17, 25, 33}`** — i.e. `8k+1` for `k ∈ {1, 2, 3, 4}`. Any other value → `422 "num_frames must be one of {9, 17, 25, 33} (8k+1 for k in 1..4)"` |

**Why `num_frames=9` is the v1.12 default.** LTX's causal VAE scheme produces `k+1` latent frames from an `8k+1`-pixel-frame segment. At `num_frames=9` (k=1), the segment encodes to 2 latent frames — the minimum needed for a "real" multi-frame hard-pin (vs. a single-latent pin which collapses to single-frame semantics). Latent 0 pins target pixel 0; latent 1 pins target pixel frames 1–8, giving 9 hard-pinned pixel frames total. Larger values (17 / 25 / 33) pin proportionally more frames at the cost of wider audio–video drift per seam and larger segment uploads. 9 is the sweet spot for 49-frame (2.04 s at 24 fps) and 97-frame (4.04 s) LTX clips — larger values will re-use a dispropriately large fraction of the prior clip.

**Response (200):**
```json
{
  "segment_uri": "storage://<32 hex>",
  "width": 1920,
  "height": 1080,
  "num_frames": 9,
  "fps": 24.0
}
```

- `segment_uri` — freshly-minted upload; feed it into the next clip's `AudioToVideoRequest.segment_uri` or `ImageToVideoRequest.segment_uri`.
- `width` / `height` — pixel dims of the encoded segment (matches source video).
- `num_frames` — echoed back so clients don't need to track it separately.
- `fps` — source MP4's stream fps as a float.

**Output format:** H.264 MP4 (video-only, no audio track), encoded in a single PyAV pass. Typical size: 500 KB – 1.5 MB for a 9-frame 1920×1080 segment — considerably smaller than 9 lossless PNGs of the same content.

**Errors:**

| Status | Code / message | When |
|---|---|---|
| `404` | `"video_not_found"` | `video_uri` doesn't resolve (unknown id or evicted) |
| `422` | `"segment_out_of_range: <detail>"` | `start_frame + num_frames` exceeds source stream length (PyAV raised `IndexError` during decode) |
| `422` | `"num_frames must be one of {9, 17, 25, 33} ..."` | Pydantic validator rejection |
| `504` | `"pyav_timeout"` | Decode + encode exceeded 30 s (malformed / very long source file) |
| `500` | `"extract_failed"` | PyAV raised during decode or encode |
| `429` | `"upload_quota_exceeded"` | Emitted MP4 bytes would push the caller past `PER_KEY_UPLOAD_BYTES_PER_DAY` |
| `401` | `"Invalid or missing API key"` | Middleware |

**Quota:** Emitted MP4 bytes count against the per-key 24 h rolling upload-byte budget (same pool as `PUT /uploads/put/{upload_id}`, `POST /v1/loras`, and `POST /v2/video/extract-frames`).

**Concurrency:** Shares `_FRAME_EXTRACT_SEMAPHORE(2)` with `/v2/video/extract-frames`. Extra callers queue until a slot is free; both endpoints together cap at 2 concurrent decodes.

**Idempotence:** Each call mints a *fresh* `segment_uri`. Calling twice with identical inputs produces two distinct uploads that each count against quota. Clients should cache the URI they receive rather than re-extract.

**Worked example — canonical chain-tail extraction.** Clip N finished with `num_frames = 49` (2.04 s at 24 fps). Extract its tail 9 frames to seed clip N+1:

```bash
# Assume CLIP_N_STORAGE_URI = "storage://<clip N final MP4>"; NUM_FRAMES = 49.
START=$((49 - 9))   # 40

SEG=$(curl -s -X POST "$API/v2/video/extract-segment" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"video_uri\":\"$CLIP_N_STORAGE_URI\",\"start_frame\":$START,\"num_frames\":9}")

# SEG = {"segment_uri":"storage://abcd...","width":1920,"height":1080,"num_frames":9,"fps":24}
SEGMENT_URI=$(echo "$SEG" | jq -r '.segment_uri')

# Submit clip N+1, conditioning on the segment. Do NOT set image_uri or keyframes.
curl -X POST "$API/v2/audio-to-video" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "prompt": "...continues scene...",
  "audio_uri": "storage://<song-slice-or-full>",
  "segment_uri": "$SEGMENT_URI",
  "model": "ltx-2-3-fast",
  "resolution": "1920x1080",
  "duration": 2.04,
  "fps": 24
}
EOF
```

The resulting clip's first 9 pixel frames are hard-pinned to clip N's frames 40–48, visually seamless. On composition save, set `tailTrimFrames: 9` on clip N (non-final) so the export drops those 9 frames from clip N (they're regenerated as the head of clip N+1) and `chainMode: "seamless-segment"` at the composition root.

**Preconditions and postconditions:**
- **Precondition:** caller holds a valid bearer; `video_uri` resolves to a video MP4 in the upload store; `start_frame + num_frames ≤ source frame count`.
- **Postcondition on 200:** a new `storage://<id>` entry exists, attributable to the caller via quota accounting; the file is a valid H.264 MP4 with exactly `num_frames` frames.

---

## SSE session tokens

### `POST /v1/sse-token`

Short-lived disposable token for browser `EventSource` (which cannot send custom headers).

**Response:** `{"token": "<32 url-safe bytes>", "expires_in": 300}`

- Tokens expire after 5 minutes.
- Pruned on each `POST /v1/sse-token` call (expired entries deleted).
- Pass as `?token=...` on `/v2/jobs/{id}/stream` and `/v1/approved-images/events`.

**Errors:** `401 "Missing API key"` if no valid bearer.

---

## Error taxonomy

All error responses follow the shape described in [Error shape](#error-shape) (`{"error", "message", "detail"}` all equal). The table below lists every distinct error string that the server can emit, grouped by scenario.

### Universal middleware

| Status | Error string | When |
|---|---|---|
| `401` | `"Invalid or missing API key"` | Bearer missing or doesn't match any `.api_keys` line |
| `403` | `"admin_required"` | v1.8.2 — caller's bearer isn't in `.admin_keys` on one of the 12 admin-gated endpoints |

### System state

| Status | Error string / code | When |
|---|---|---|
| `503` | `"System is paused for maintenance"` (sync) / `"system_paused"` (v2/turbo) + `Retry-After: 300` | `_paused == True` |
| `503` | `"turbo_mode_active: ACE/JoyAI unavailable..."` + `Retry-After: 10` | Music v2 submitted while turbo is active |
| `503` | `"Music generation not enabled (LOAD_ACE=0)"` | `LOAD_ACE=0` |
| `503` | `"JoyAI not enabled (LOAD_JOYAI=0)"` / `"joyai_disabled: ..."` | `LOAD_JOYAI=0` or sidecar error |
| `503` | `"sidecar_unreachable"` / `"sidecar_timeout"` | JoyAI / ERNIE / LTX-sidecar HTTP failure |
| `503` | `"ernie_disabled: ..."` | `LOAD_ERNIE=0` or ERNIE sidecar error |
| `500` | `"Flux not enabled"` / `"Flux pipeline not loaded"` | `LOAD_FLUX=0` and request targets Flux |
| `500` | `"Chat model not loaded"` | `chat.is_ready == False` |

### System operations

| Status | Error string | When |
|---|---|---|
| `500` | `"pause_failed"` | `/v1/system/pause` failed |
| `500` | `"resume_failed"` | `/v1/system/resume` failed |
| `500` | `"flux_unload_failed"` / `"flux_reload_failed"` | Flux unload/reload failed |
| `500` | `"ltx_unload_failed"` / `"ltx_reload_failed"` | LTX unload/reload failed |
| `409` | `"already_enabled"` / `"already_disabled"` | `/v1/system/turbo` called with state already matching body |
| `500` | `"turbo_toggle_failed"` | Turbo transition failed |
| `400` | `"remote_sidecar_not_configured: set LTX_REMOTE_SIDECAR_URL in .env"` | Pool endpoint called with no sidecar |
| `500` | `"pool_scale_failed: ..."` | Pool resize failed |

### Queue / rate limit

| Status | Error code | When | Header |
|---|---|---|---|
| `429` | `"queue_full"` | `job_store.pending_count() >= MAX_QUEUE_DEPTH (30, v1.16.1; was 10)` | `Retry-After: 30` |
| `429` | `"music_queue_full"` | Music-specific depth ≥ `MAX_MUSIC_PENDING (5)` | `Retry-After: 30` |
| `429` | `"batch_queue_full"` | `batch_store.active_count() >= MAX_BATCH_QUEUE_DEPTH (5)` | `Retry-After: 30` |
| `429` | `"per_key_queue_full"` | v1.8.2 — single bearer has `PER_KEY_QUEUE_CAP` (15, v1.16.1; was 3) / `PER_KEY_MUSIC_CAP` (5, v1.16.1; was 2) / `PER_KEY_BATCH_CAP` (5, v1.16.1; was 2) jobs, music jobs, or batches already in flight. All four caps env-overridable; see [`docs/operator-tuning.md`](./operator-tuning.md). | `Retry-After: 30` |
| `429` | `"per_key_upload_quota_exceeded"` | v1.8.2 — single bearer has uploaded `PER_KEY_UPLOAD_BYTES_PER_DAY` (default 10 GiB) within the last 24h rolling window | `Retry-After: 3600` |
| `429` | `"per_key_lora_count_exceeded"` | v1.8.2 — single bearer has `PER_KEY_LORA_COUNT` (default 20) active LoRAs; DELETE some first | `Retry-After: 3600` |

### Uploads

| Status | Error string | When |
|---|---|---|
| `413` | `"Upload exceeds 1024MB limit"` | `PUT /uploads/put/{id}` body > `MAX_UPLOAD_BYTES` |
| `413` | `"File exceeds 1024MB limit"` | `POST /v1/loras` file > `MAX_LORA_SIZE_BYTES` |
| `422` | `"content_type_mismatch"` | v1.8.2 — `PUT /uploads/put/{id}` body's magic bytes don't match the declared `Content-Type` header. Lenient on `application/octet-stream` and unrecognized types (those pass) |
| `400` | `"Expected multipart/form-data"` | `POST /v1/loras` content-type mismatch |
| `400` | `"Missing 'file' field"` | `POST /v1/loras` no file |
| `400` | `"File must be a .safetensors file"` | `POST /v1/loras` wrong extension |
| `422` | `"Missing 'name' field"` | `POST /v1/loras` no name |
| `400` | `<registry validation error>` | `POST /v1/loras` duplicate id etc. |

### LoRA / asset resolution

| Status | Error string | When |
|---|---|---|
| `404` | `"LoRA not found: <id>"` | `lora.id` not in LTX registry |
| `404` | `"Flux LoRA not found: <id>"` | `lora.id` not in Flux registry |
| `404` | `"LoRA not found: <id>"` | `DELETE /v1/loras/{id}` with unknown id |
| `422` | `<FluxLoraError message>` | LoRA/model incompatibility on Flux path |
| `500` | `"outpaint LoRA resolve returned None — registry misconfigured"` | Default `ic-lora-outpaint` missing |

### Generation request validation

| Status | Error string | When |
|---|---|---|
| `422` | `"Specify at most one of: image_uri, keyframes, segment_uri"` | v1.12 — i2v/a2v with ≥ 2 of the three conditioning inputs set |
| `422` | `"Cannot specify image_strength together with segment_uri"` | v1.12 — i2v with explicit `image_strength` + `segment_uri` |
| `422` | `"Cannot specify image_strength together with keyframes or segment_uri"` | v1.12 — a2v with explicit `image_strength` + `keyframes` or `segment_uri` |
| `422` | `"keyframes list must not be empty"` | i2v empty keyframes array |
| `422` | `"At most 8 keyframes are allowed"` | i2v keyframes > 8 |
| `422` | `"Either image_uri or keyframes is required"` | i2v with none of `image_uri` / `keyframes` / `segment_uri` set |
| `422` | `"Resolved frame_index N is out of range for M frames"` | Keyframe bounds violation |
| `422` | `"Duplicate frame_index values after resolution"` | Two keyframes resolve to same index |
| `422` | `"num_frames must be one of {9, 17, 25, 33} (8k+1 for k in 1..4)"` | v1.12 — `POST /v2/video/extract-segment` with invalid `num_frames` |
| `404` | `"segment_uri not found"` | v1.12 — i2v/a2v references a `segment_uri` that doesn't resolve |
| `404` | `"video_not_found"` | `POST /v2/video/{extract-frames,extract-segment}` source MP4 unresolvable |
| `422` | `"segment_out_of_range: <detail>"` | `POST /v2/video/extract-segment` — `start_frame + num_frames` exceeds source length |
| `422` | `"frame_index_out_of_range: <detail>"` | `POST /v2/video/extract-frames` — requested index exceeds source length |
| `504` | `"pyav_timeout"` | Extract-frames or extract-segment decode exceeded 30 s |
| `500` | `"extract_failed"` | Extract-frames or extract-segment decode/encode raised |
| `422` | `"joyai-edit requires exactly one image_uri"` | `joyai-edit` with `image_uris.length != 1` |
| `422` | `"task_type '<t>' requires source_audio_uri"` | Music task-type missing `source_audio_uri` |
| `422` | `"task_type '<t>' requires track_name"` | extract/lego/complete missing `track_name` |
| `422` | `"Content rejected or generation failed: <detail>"` | Retake generation failure |
| `422` | `"Messages list cannot be empty"` | Chat completions no messages |

### URI / file resolution

| Status | Error string | When |
|---|---|---|
| `404` | `<FileNotFoundError message>` | `uploads.resolve()` fails on any `storage://` |
| `404` | `"source_audio_uri not found"` | Music `source_audio_uri` fails resolve |
| `404` | `"reference_audio_uri not found"` | Music `reference_audio_uri` fails resolve |

### Jobs / batch lifecycle

| Status | Error string | When |
|---|---|---|
| `404` | `"Job not found"` | Unknown job id on `/v2/jobs/*` |
| `409` | `"Job result not ready"` | `GET /v2/jobs/{id}/result` on non-completed job |
| `404` | `"Result file expired or not found"` | Job complete but file missing (TTL evicted) |
| `409` | `"Cannot cancel a finished job"` | `DELETE /v2/jobs/{id}` on terminal-state job |
| `404` | `"Batch not found"` | Unknown batch id |
| `404` | `"Batch item not found or not completed"` | Bad batch index or not yet complete |
| `409` | `"batch_already_finished"` | `DELETE /v2/batch/{id}` on terminal batch |
| `400` | `"Invalid item type at index N: ..."` | Batch item has unsupported type |
| `422` | `"Validation failed at item N (<type>): <detail>"` | Batch item params fail Pydantic validation |

### Job `error.code` values (v2 status)

On failure, `GET /v2/jobs/{id}` returns an `error: {code, message}` block. Codes emitted by the worker:

| Code | Source | Meaning |
|---|---|---|
| `"generation_failed"` | `job_queue.py:300` default | Generic failure during generate |
| `"cuda_oom"` | `job_queue.py:300` | `"out of memory"` substring in exception |
| `"ace_error"` | `server.py:282` | Music dispatch raised `AceError` |

### Approved images / history / compositions

| Status | Error string | When |
|---|---|---|
| `401` | `"Missing API key"` | Auth layer not satisfied |
| `400` | `"Missing image_uri"` | `POST /v1/approved-images` no body field |
| `404` | `"Not found"` | Unknown approved image id OR owned by another key |
| `404` | `"Image file not found"` | Manifest entry exists, file evicted |
| `404` | `"Not found"` (history) | Unknown generation id OR owned by another key |
| `404` | `"Result file not found"` (history image) | On-disk result missing |
| `404` | `"Thumbnail not found"` | On-disk thumbnail missing |
| `404` | `"Composition not found"` | Unknown composition id OR owned by another key |

### Char rank

| Status | Error string | When |
|---|---|---|
| `500` | `"Vision model did not return valid JSON"` | No JSON regex match in model output |
| `502` | `"char_rank_schema_violation"` | v1.8.2 — model's JSON output failed Pydantic validation against `CharRankResponse` (missing keys, wrong types, `score` or `analysis.*` out of range). `detail` carries the pydantic ValidationError (≤500 chars) |

### Dashboard

| Status | Error string | When |
|---|---|---|
| `404` | `"dashboard.html not found"` | Static file missing |

### Path redaction

Any error message containing `/mnt/`, `/home/`, or `/tmp/` is truncated to 500 chars and then replaced with `"Internal server error"` (all three fields) before leaving the server.

---

## Endpoint index

The **MCP** column lists the corresponding tier-1 wrapper tool from [docs/MCP.md](MCP.md). `submit_job` covers any v2 generation POST returning a `SubmissionEnvelope`; system-control endpoints (`/v1/system/*`, `/v1/{flux,ltx}/*`) are deliberately not wrapped — they stay on the dashboard.

| Method | Path | Auth | Purpose | MCP |
|---|---|---|---|---|
| GET | `/health` | no | Liveness + model status | — |
| GET | `/dashboard` | no | GPU management SPA | — |
| GET | `/v1/system/gpu` | no | `nvidia-smi` telemetry | — |
| POST | `/v1/system/pause` | yes | Evict all + cancel queued | — |
| POST | `/v1/system/resume` | yes | Reload all | — |
| POST | `/v1/flux/unload` | yes | Unload Flux | — |
| POST | `/v1/flux/reload` | yes | Reload Flux | — |
| POST | `/v1/ltx/unload` | yes | Unload LTX | — |
| POST | `/v1/ltx/reload` | yes | Reload LTX | — |
| POST | `/v1/system/turbo` | yes | Toggle turbo mode | — |
| GET | `/v1/system/pool` | yes | Remote-sidecar pool state (per-provider in v1.9.0) | — |
| POST | `/v1/system/pool/remote-workers` | yes | Set target remote worker counts (legacy `{count}` or per-provider dict) | — |
| POST | `/v1/system/pool/remote-workers/{provider}` | yes | Set target worker count for one provider (v1.9.0) | — |
| GET | `/v1/system/sampler` | yes | Sampler subset of gen config | — |
| POST | `/v1/system/sampler` | yes | Toggle CFG++ / Euler | — |
| GET | `/v1/system/config` | yes | Full LTX gen config | — |
| POST | `/v1/system/config` | yes | Merge-update LTX gen config | — |
| POST | `/v1/system/config/reset` | yes | Reset LTX gen config | — |
| GET | `/v1/system/flux-config` | yes | Full Flux gen config | — |
| POST | `/v1/system/flux-config` | yes | Merge-update Flux gen config | — |
| POST | `/v1/system/flux-config/reset` | yes | Reset Flux gen config | — |
| POST | `/v1/text-to-video` | yes | Sync t2v | — |
| POST | `/v1/image-to-video` | yes | Sync i2v / keyframe | — |
| POST | `/v1/audio-to-video` | yes | Sync a2v | — |
| POST | `/v1/retake` | yes | Sync retake | — |
| POST | `/v1/text-to-image` | yes | Sync Flux t2i / ERNIE t2i | — |
| POST | `/v1/image-to-image` | yes | Sync Flux i2i | — |
| POST | `/v1/image-edit` | yes | Sync edit (Flux multi / JoyAI single) | — |
| POST | `/v1/music` | yes | Sync music (ACE) | — |
| POST | `/v1/chat/completions` | yes | llama-swap proxy | — |
| POST | `/v1/upload` | yes | Get upload slot | `upload_file` |
| PUT | `/uploads/put/{upload_id}` | yes | Upload bytes | `upload_file` |
| GET | `/uploads/get/{upload_id}` | yes | Read back upload (v1.9.1) | `download_storage_uri` |
| GET | `/v1/loras` | yes | List LTX LoRAs | — |
| POST | `/v1/loras` | yes | Upload LTX LoRA (multipart) | — |
| DELETE | `/v1/loras/{lora_id}` | yes | Delete LTX LoRA | — |
| GET | `/v1/flux-loras` | yes | List Flux LoRAs | — |
| POST | `/v1/flux-loras/rescan` | yes | Re-scan Flux LoRA folder | — |
| POST | `/v1/sse-token` | yes | Issue 5-min SSE token | — |
| POST | `/v1/approved-images` | yes | Approve an image | — |
| GET | `/v1/approved-images` | yes | List approved images | — |
| GET | `/v1/approved-images/events` | no | SSE feed (bearer OR `?token=`) | — |
| GET | `/v1/approved-images/{id}/file` | yes | Fetch approved file | — |
| POST | `/v2/text-to-video` | yes | Async t2v | `submit_job` |
| POST | `/v2/image-to-video` | yes | Async i2v | `submit_job` |
| POST | `/v2/audio-to-video` | yes | Async a2v | `submit_job` / `cut_music_video` |
| POST | `/v2/retake` | yes | Async retake | `submit_job` |
| POST | `/v2/video-outpaint` | yes | **v1.7.0** Async IC-LoRA outpaint | `submit_job` |
| POST | `/v2/video-hdr` | yes | **v1.14.0** Async IC-LoRA HDR (canvas preserved) | `submit_job` |
| POST | `/v2/video/extract-frames` | yes | **v1.10.0** PyAV frame extractor (v1.11.5 legacy chain conditioning) | — |
| POST | `/v2/video/extract-segment` | yes | **v1.12.0** PyAV segment extractor (experimental v1.12 chain conditioning) | `extract_segment` |
| POST | `/v2/text-to-image` | yes | Async t2i | `submit_job` |
| POST | `/v2/image-to-image` | yes | Async i2i | `submit_job` |
| POST | `/v2/image-edit` | yes | Async edit | `submit_job` |
| POST | `/v2/music` | yes | Async music | `submit_job` / `cut_music_video` |
| GET | `/v2/jobs/{id}` | yes | Poll status | `get_job` / `wait_for_job` |
| GET | `/v2/jobs/{id}/preview` | yes | Preview JPEG (204 when empty) | — |
| GET | `/v2/jobs/{id}/result` | yes | Download final media | `download_job_result` |
| GET | `/v2/jobs/{id}/stream` | no (bearer OR `?token=`) | SSE live status | — |
| DELETE | `/v2/jobs/{id}` | yes | Cancel job | `cancel_job` |
| POST | `/v2/batch` | yes | Submit batch | — |
| GET | `/v2/batch/{id}` | yes | Poll batch status | — |
| GET | `/v2/batch/{id}/result/{index}` | yes | Download batch item result | — |
| DELETE | `/v2/batch/{id}` | yes | Cancel batch | — |
| GET | `/v2/history` | yes | Per-key history list | — |
| GET | `/v2/history/{id}` | yes | Full record with params + gen_config | — |
| GET | `/v2/history/{id}/image` | yes | Full-size history media | — |
| GET | `/v2/history/{id}/thumbnail` | yes | History thumbnail | — |
| DELETE | `/v2/history/{id}` | yes | Delete history entry | — |
| POST | `/v2/char/rank` | yes | Vision character consistency rank | — |
| POST | `/v2/compositions` | yes | Create composition | `create_composition` |
| GET | `/v2/compositions` | yes | List compositions | — |
| GET | `/v2/compositions/{id}` | yes | Get composition | — |
| PUT | `/v2/compositions/{id}` | yes | Update composition | — |
| DELETE | `/v2/compositions/{id}` | yes | Delete composition | — |
| POST | `/v2/compositions/{id}/export` | yes | Enqueue export job | `export_composition` |

**Total: 73 routes.**

---

## Curl examples

Set `API` and `KEY` env vars first:

```bash
export API="http://localhost:8090"
export KEY="your-api-key"
```

### Upload + sync t2i

```bash
# 1. Upload only needed for i2i/i2v/a2v/retake/outpaint/image-edit
SLOT=$(curl -s -X POST "$API/v1/upload" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d '{}')
UPLOAD_URL=$(echo $SLOT | jq -r '.upload_url')
STORAGE_URI=$(echo $SLOT | jq -r '.storage_uri')
curl -X PUT "$UPLOAD_URL" \
  -H "Authorization: Bearer $KEY" \
  --data-binary @input.jpg

# 2. Straight text-to-image
curl -X POST "$API/v1/text-to-image" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"cinematic portrait","model":"flux2-dev","width":1024,"height":1024}' \
  --output out.webp
```

### Async video job → poll → fetch

```bash
JOB=$(curl -s -X POST "$API/v2/text-to-video" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"a cat","model":"ltx-2-3-fast","resolution":"1920x1080","duration":4,"fps":24}' \
  | jq -r '.job_id')

while true; do
  STATUS=$(curl -s -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JOB" | jq -r '.status')
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && echo "failed" && exit 1
  sleep 2
done

curl -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JOB/result" --output out.mp4
```

### Async video job via SSE

```bash
# Open a long-lived SSE connection; close when terminal state arrives.
JOB=$(curl -s -X POST "$API/v2/text-to-video" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"a cat","model":"ltx-2-3-fast","resolution":"1920x1080","duration":4,"fps":24}' \
  | jq -r '.job_id')

curl -N -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JOB/stream"
```

### Video outpaint (v1.7.0)

```bash
# 1. Upload source video → $STORAGE_URI
# 2. Submit async outpaint (silent MP4 output)
JOB=$(curl -s -X POST "$API/v2/video-outpaint" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d @- <<EOF | jq -r '.job_id'
{
  "video_uri": "$STORAGE_URI",
  "prompt": "extend the scene naturally, matching lighting and style",
  "target_resolution": "1920x1080",
  "position": "center",
  "duration": 5.0,
  "fps": 24,
  "skip_stage_2": false
}
EOF
)

# 3. Stream status until completion
curl -N -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JOB/stream"

# 4. Download
curl -H "Authorization: Bearer $KEY" "$API/v2/jobs/$JOB/result" --output outpaint.mp4
```

### Batch submit

```bash
curl -X POST "$API/v2/batch" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"type":"text-to-image","params":{"prompt":"cat","model":"flux2-klein","width":1024,"height":1024,"num_inference_steps":4}},
      {"type":"text-to-video","params":{"prompt":"dog","model":"ltx-2-3-fast","resolution":"1280x720","duration":4,"fps":24}}
    ]
  }'
```

---

## Changelog

- **v1.12.0** (2026-04-20)
  - **NEW (experimental):** `POST /v2/video/extract-segment` — extract a contiguous `8k+1` pixel-frame range (`k ∈ {1..4}`; default 9) from a stored MP4 and re-encode as a standalone H.264 MP4 upload. Same capability-URL / quota / semaphore pattern as `/v2/video/extract-frames`.
  - **NEW (experimental):** `segment_uri: string | null` field on `AudioToVideoRequest` and `ImageToVideoRequest`. 3-way mutually exclusive with `image_uri` and `keyframes` (422 on mixed specifications). Backend VAE-encodes the segment as a multi-latent-frame tensor and hard-pins 9 consecutive target pixel frames via a single `VideoConditionByLatentIndex` — replacing v1.11.5's single-pixel-frame-0 hard-pin, which drifted past seam 2.
  - **Composition:** new `chainMode` value `"seamless-segment"` (in addition to `"seamless"` / `"hardcut"`); new optional per-clip `segmentUri` audit field; recommended `tailTrimFrames: 9` on non-final clips in seamless-segment mode.
  - Opt in behind FE flag `flags.v112_seamless_segment`. Legacy `"seamless"` (v1.11.5 keyframes) and `"hardcut"` paths remain byte-identical.
  - See [Experimental callout](#experimental--v112-segment-chain-conditioning) and [`docs/handover-frontend-v1.10-chain.md`](./handover-frontend-v1.10-chain.md) v1.12 section.
- **v1.11.5** (2026-04-20) — Revert v1.11.3's chain-keyframes routing (was pinning at latent-frame granularity, produced visible "slideshow" through pixel frames 1–16). Unconditional delegation to `combined_image_conditionings`. Promoted the `a2v keyframes=` diagnostic to `WARNING` so it actually emits.
- **v1.11.4** (2026-04-20) — `ImageConditioningInput(..., crf=0)` so user-provided reference images are no longer silently H.264-CRF33-compressed before VAE encode. Promoted dispatch diagnostic log level.
- **v1.11.3** (2026-04-20) — Attempted fix for multi-keyframe chain conditioning by routing through `image_conditionings_by_replacing_latent`. Wrong granularity (latent-frame vs pixel-frame) — reverted in v1.11.5.
- **v1.11.2** (2026-04-20) — Composition clip gains optional `audioDurationSec: float | null` — decouples per-clip audio atrim from video `effective_duration`. Enables `tailTrimFrames=6` with zero visual seam AND full-song audio continuity. Frontend-preferred export path.
- **v1.11.1** (2026-04-20) — Reverts the v1.11.0 `tailTrimFrames=6` recommendation back to `tailTrimFrames=3` as the minimum-total-perception config under the pre-v1.11.2 audio clamp. Both 3 and 6 continue to work.
- **v1.11.0** (2026-04-19) — Briefly recommended `tailTrimFrames=6` to eliminate a visual stutter; traded for 208 ms of audible per-seam audio dropout. Superseded by v1.11.2's `audioDurationSec` decoupling.
- **v1.10.1** (2026-04-18) — `AudioToVideoRequest.image_strength` default changed from `0.85` to `1.0`. v1.9.5 added the field to the model but v1.9.5–v1.9.9 silently dropped it in `_submit_job` (hardcoded 1.0 in `_run_a2v`). v1.10.0 wired the field end-to-end but kept the 0.85 default, regressing default-path a2v. v1.10.1 restores 1.0 so default behavior matches pre-v1.10.0.
- **v1.10.0** (2026-04-18)
  - `POST /v2/video/extract-frames` — PyAV multi-frame extractor, lossless PNG output, bounded concurrency via `_FRAME_EXTRACT_SEMAPHORE(2)`, 30 s timeout, per-key quota accounting.
  - `AudioToVideoRequest.keyframes: list[KeyframeInput] | None` — multi-keyframe support now matches i2v. Legacy `image_uri` + `image_strength` single-keyframe path is unchanged.
  - Composition clip schema: optional `tailTrimFrames: int` (default 0), optional per-clip `fps` override.
- **v1.9.x** — Per-provider remote pool (Modal + RunPod dict-keyed), `POST /v1/system/pool/remote-workers/{provider}`, `GET /uploads/get/{id}`, composition `audio_uri` persistence, per-clip `audioStart` beat-synced audio, per-key queue/upload/LoRA caps.
- **v1.8.x** — Admin gate on mutation endpoints (`.admin_keys`, `403 admin_required`); tenancy enforcement on `/v2/jobs/*` and `/v2/batch/*`; identity preservation hooks on Klein edit (`preserve_identity`, `identity_strength`, `identity_mode`).
- **v1.7.0** (2026-04-17)
  - `POST /v2/video-outpaint` — IC-LoRA Outpaint (`oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint`, registered id `ic-lora-outpaint`). Silent MP4 output; black-sentinel LoRA fills padded regions; `skip_stage_2` fast path.
  - `POST /v2/video-hdr` — IC-LoRA HDR (`Lightricks/LTX-2.3-22b-IC-LoRA-HDR`, registered id `ic-lora-hdr`). Silent MP4 output; LDR→HDR transform with canvas preserved; reuses outpaint pipeline with `target == source, position="center"`; `skip_stage_2` produces the closest match to upstream's reference behavior.
  - `JobType.VIDEO_OUTPAINT` added.
  - New [`OutpaintPosition`](#outpaintposition-v170) enum (9 values).
- **v1.6** (2026-04-15)
  - LTX remote-sidecar pool: `GET /v1/system/pool`, `POST /v1/system/pool/remote-workers`, `POST /v1/system/pool/remote-workers/{provider}`. Turbo + up to 4 Modal + 2 RunPod workers for 8-way concurrent video (v1.9.0).
  - `LTX_REMOTE_SIDECAR_URL`, `LTX_REMOTE_SIDECAR_TOKEN`, `LTX_REMOTE_SIDECAR_MAX_WORKERS` env vars.
- **v1.5**
  - Turbo entry hardening: cuda:1 drain verification via `nvidia-smi` bus-id matching.
  - `_systemctl_unit` helper with proper error surfacing.
- **v1.4**
  - Auto-turbo: batch worker engages turbo when cuda:1 idle ≥ `AUTO_TURBO_IDLE_MINUTES` and batch has ≥ 2 items.
- **v1.3** (2026-04-13)
  - Upstream LTX-2 migration: `SingleGPUModelBuilder`, `CachingModelFactory`, new Denoiser classes.
  - `GET/POST /v1/system/config`, `POST /v1/system/config/reset` — 13 tunable LTX params persisted to `.gen_config.json`.
  - `GET/POST /v1/system/sampler` — CFG++ default; Euler fallback.
  - `GET /v2/history/{id}` — full record with `params`, `gen_config`, `seed`, `enhanced_prompt`.
  - `GET /v2/batch/{id}/result/{index}` — individual batch-item download.
  - BatchSplitAdapter, bf16 precision fix, TilingConfig.default() for VAE decode.
  - `DUAL_GPU_LTX` env flag, `torch.compile` flag (default OFF).
- **v1.2** (2026-04-11)
  - Dual-GPU layout: cuda:0 = LTX ↔ Flux swap; cuda:1 = ACE + JoyAI (or ERNIE) coexist.
  - Music: `POST /v1/music` + `POST /v2/music` (ACE Step). Gated by `LOAD_ACE`.
  - Batch scheduler: `POST /v2/batch`, `GET /v2/batch/{id}`, `DELETE /v2/batch/{id}`.
  - Turbo mode: `POST /v1/system/turbo` — 2 concurrent LTX workers.
  - `GET /dashboard` SPA, `GET /v1/system/gpu` telemetry.
  - `/health` adds `ace` field.
- **v1.1.8** (2026-04-11)
  - `model: "joyai-edit"` on `/v1/image-edit` + `/v2/image-edit` — single-image editing via `JoyAI-Image-Edit-Diffusers` sidecar on 127.0.0.1:8092.
  - Fixed `/v1/image-edit` silently ignoring `model: "flux2-dev"`.
  - `phase="encoding"` during opaque sidecar calls.
- **v1.1.7** (2026-04-11)
  - `GET /v2/jobs/{id}/stream` — SSE endpoint (bearer OR `?token=`).
- **v1.1.6** (2026-04-11)
  - `phase` field on `/v2/jobs/{id}` (`denoising` / `decoding` / `encoding` / `saving`). Denoising caps at 0.90.
  - `/v2/jobs/{id}/preview` zero-copy from history thumbnail; async fallback via PyAV.
  - SQLite WAL mode; `history.save()` off the worker's event loop.
- **v1.1.5** — `/v2/jobs/{id}/preview` returns `204` (not `404`) while empty; video-first-frame via PyAV.
- **v1.1.4** — Single-GPU swap mode; `_ensure_ltx_resident` / `_ensure_flux_ready`; `evict_all` leak fix (reclaims 99% of LTX VRAM).
- **v1.1.3** — VAE force-upcast fix (pre-hook + fp32 params).
- **v1.1.2** — Lossless VP8L WEBP for Flux output.
- **v1.1.1** — Dropped FP8 layerwise casting on Flux 2 Dev; adapter-mode LoRA (free strength changes).
- **v1.1** — Flux LoRA folder-drop; first/middle/last keyframes; char rank; gen history.
