# Changelog

All notable changes to taco-backend. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## v1.9.2 — 2026-04-19

### Fix: concurrent PyAV encode race on `avcodec_open2(aac)`

Under turbo mode (2 local workers), two a2v jobs hitting `_video_to_bytes` around the same time intermittently failed with `av.error.ValueError: [Errno 22] Invalid argument: 'avcodec_open2(aac)'` during `container.mux()` / `start_encoding()`. FFmpeg's muxer initialization (opening the libx264 + AAC encoders) is not fully thread-safe across concurrent output containers in the same process. Single-threaded repro of the exact failing file works byte-for-byte.

Fix: module-level `_ENCODE_LOCK = threading.Lock()` in `split_model_manager.py` wraps every `encode_video(...)` call site (the single funnel `_video_to_bytes`). Denoising still runs in parallel under turbo (the expensive 10-60s part); only the final ~1-2s MP4 encode tail serializes. Throughput impact ≤ 5% on typical 5-30s videos. Applies to every video job type (t2v, i2v, a2v, retake, outpaint).

## v1.9.1 — 2026-04-19

### `GET /uploads/get/{upload_id}` — serve uploads back

Routing gap fix. The upload store has always been disk-backed + persistent, but there was no GET route to read files back. Frontend consumers (noodle-m MusicVideo tab reloading an uploaded song, composition-export audio preview) hit 404s.

- **New**: `GET /uploads/get/{upload_id}` returns the raw bytes with an inferred `Content-Type` (`image/jpeg`, `image/png`, `image/webp`, `audio/wav`, `audio/mpeg`, `audio/flac`, `audio/ogg`, `video/mp4`; fallback `application/octet-stream`). Auth via global bearer middleware — the 128-bit `uuid4` hex ID is the capability.
- **Errors**: `400 invalid_upload_id` (malformed), `404 upload_not_found` (valid ID, no file), `500 upload_read_failed` (disk I/O).
- **Deferred by design**: no TTL, no per-key ownership enforcement, no signed URLs, no HTTP Range — add selectively if needed.
- New helper `_infer_media_type_from_magic(head) -> str` alongside the existing `_content_type_matches_magic` (v1.8.2).

## v1.9.0 — 2026-04-19

### Composition export: optional audio overlay

`POST /v2/compositions/{comp_id}/export` now accepts an optional body `{"audio_uri": "storage://<id>"}`. When set, ffmpeg muxes the referenced audio file onto the stitched video output (AAC @ 192kbps, `-shortest`). No body / empty body preserves pre-v1.9 video-only behavior. Single-clip + audio works via an inserted `[0:v]null[vout]` pass-through filter (prior single-clip short-circuit that returned raw bytes is bypassed when audio is present).

### RunPod as a second remote-sidecar provider alongside Modal

v1.6's single remote-sidecar pool becomes a **multi-provider** pool. Modal and RunPod run side-by-side, each with independent target/active/max worker counts. Operators can burst to both simultaneously (up to `2 local + 4 modal + 2 runpod = 8` concurrent video workers) or pick whichever is cheapest / has availability. No Modal retirement — existing deployments keep working unchanged.

**Why**: RunPod RTX PRO 6000 Blackwell serverless is cheaper than Modal (~$2.66/hr active vs ~$3.03/hr), the RunPod account has free credits, and two providers = redundancy against single-vendor outages.

#### Backwards compatibility

All pre-v1.9 deployments keep working. `LTX_REMOTE_SIDECAR_URL` / `LTX_REMOTE_SIDECAR_TOKEN` / `LTX_REMOTE_SIDECAR_MAX_WORKERS` are still honored and transparently aliased to the `modal` provider. The legacy `POST /v1/system/pool/remote-workers {"count": N}` body shape still scales modal only. Legacy flat fields (`remote_sidecar_configured`, `remote_worker_target/active/max`, `remote_sidecar_url`) stay in the `GET /v1/system/pool` response as aliases to the modal provider.

#### API — breaking additions (response shape expanded, not removed)

- **`GET /v1/system/pool`** adds `providers: {modal: {configured, url, target, active, max}, runpod: {...}}`. Legacy flat fields preserved.
- **`POST /v1/system/pool/remote-workers`** now accepts `{"modal": N, "runpod": M}` alongside the legacy `{"count": N}`. Response returns the same shape as `GET /v1/system/pool` plus `applied_now`.
- **`POST /v1/system/pool/remote-workers/{provider}`** (new) — cleanest RESTful per-provider scale, `{provider}` ∈ `{"modal", "runpod"}`.

#### Config

- New env vars: `LTX_MODAL_SIDECAR_URL/TOKEN/MAX_WORKERS`, `LTX_RUNPOD_SIDECAR_URL/TOKEN/MAX_WORKERS`. Modal vars fall back to `LTX_REMOTE_SIDECAR_*` when unset.
- New `config.LTX_PROVIDER_LORAS_MOUNT` maps provider → LoRA mount path for outpaint LoRA rewrites (Modal `/mnt/nvme-1/huggingface/loras/`, RunPod `/runpod-volume/loras/`).
- `LTX_RUNPOD_MAX_WORKERS` defaults to 2 (matches the endpoint's `workers.max` in `endpoint.yaml`).

#### Internal refactors

- `ltx_sidecar_client.ltx_remote_sidecars: dict[str, LtxSidecarClient]` replaces the single `ltx_remote_sidecar` module-level. The singular name is kept as a backwards-compat alias pointing at the modal entry.
- `server.py` pool state becomes per-provider dicts: `_remote_worker_targets` + `_remote_worker_tasks` keyed by provider. `_PROVIDERS = ("modal", "runpod")`.
- `_dispatch_job_turbo_remote(job, *, provider: str)` gains a provider kwarg. `_scale_remote_pool()` uses `functools.partial` to bind each worker task to its provider at spawn time.
- Dashboard grows a second row ("RunPod Pool") mirroring the Modal row. JS walks `data.providers` with fallback to legacy flat fields.

#### New repo: `/mnt/nvme-1/servers/ltx-sidecar-runpod/`

- `runpod_app.py` — FastAPI app mirroring `modal_app.py::fastapi_app`. `/ping` health probe, `/health`, `/load`, `/unload`, `/generate` all share the Modal client contract.
- `Dockerfile` — `runpod/pytorch:2.11.0-py3.12-cuda13.0-ubuntu24.04` base + LTX-2 editable install + minimal taco-inference-deps.
- `download_weights.py` — one-shot script to populate the RunPod Network Volume (~80 GB).
- `endpoint.yaml` — RunPod Load-Balancing Serverless config (GPU `RTX_PRO_6000`, `min_workers=1`, `max_workers=2`).

#### Migration

- No action required for existing single-provider Modal users — old env vars still work.
- To add RunPod: build + push the image at `/mnt/nvme-1/servers/ltx-sidecar-runpod/`, create the Serverless endpoint + Network Volume, populate weights, add `LTX_RUNPOD_SIDECAR_URL` + `LTX_RUNPOD_SIDECAR_TOKEN` to `.env`, restart backend.
- **Security note**: rotate any RunPod API keys that were shared in chat/transcripts during planning. The `SIDECAR_AUTH_TOKEN` secret on the RunPod endpoint is independent of the user-account API key.

## v1.8.2 — 2026-04-18

### server.py security sweep — admin gate, quotas, validation, timing hardening

Eight SEC findings from the v1.8.1 audit, all landing in `server.py`. No new dependencies. No endpoint URLs change. The one behavioural change that matters to clients is the admin gate on 12 mutation endpoints — see migration note below.

- **SEC P0-2 — Admin gate on 12 mutation endpoints.** `POST /v1/system/{pause,resume,turbo,config,config/reset,flux-config,flux-config/reset,sampler,pool/remote-workers}` and `POST /v1/{flux,ltx}/{unload,reload}` now require the caller's bearer to appear in a new `.admin_keys` file (or `TACO_ADMIN_KEY` env var). On mismatch: `403 admin_required`. Read endpoints (`GET /v1/system/{pool,config,flux-config,sampler}`) stay user-level. **Backwards-compat bridge**: when `.admin_keys` is empty, every entry in `.api_keys` is treated as admin (preserves pre-v1.8.2 behaviour), and a WARN is logged at boot so ops notices the degraded posture. When `.api_keys` is also empty, auth is globally off and the gate is a no-op.
- **SEC P1-3 — Per-API-key queue caps.** New `PER_KEY_QUEUE_CAP` (default 3), `PER_KEY_MUSIC_CAP` (2), `PER_KEY_BATCH_CAP` (2). Enforced BEFORE the global `MAX_QUEUE_DEPTH` / `MAX_MUSIC_PENDING` / `MAX_BATCH_QUEUE_DEPTH` caps, so one bearer can't claim the whole queue. Breach returns `429 per_key_queue_full` + `Retry-After: 30`. Counters keyed by `sha256(api_key)` — raw bearers never land in the map. Decremented from `worker_loop`'s `finally` (via new `on_complete` callback on `job_queue.py::worker_loop`), from `_run_music_job`'s `finally`, and from `batch_worker` completion.
- **SEC P1-5 — CharRankResponse validation.** `/v2/char/rank` previously parsed whatever JSON the vision model emitted and echoed it to the client. Now validates against a new `CharRankResponse` Pydantic model (`score` 0–10, `analysis.{face_match,eyes,proportions,overall_likeness}` 1–10, `edits.{add,remove,modify}`). Failures return `502 char_rank_schema_violation` with the Pydantic detail truncated to ≤500 chars.
- **SEC P2-1 — Constant-time bearer compare.** Middleware `any(compare_digest(...) for key in API_KEYS)` short-circuited on first match, leaking set membership via wall-clock timing. Replaced with full-iteration compare at both the middleware and inside `_require_admin`.
- **SEC P2-3 — Per-key upload byte quota.** New `PER_KEY_UPLOAD_BYTES_PER_DAY` (default 10 GiB). Rolling 24h window keyed by `sha256(api_key)`. Applied in `PUT /uploads/put/{id}` (early peek via `Content-Length`, final check after body read) and `POST /v1/loras`. Breach returns `429 per_key_upload_quota_exceeded` + `Retry-After: 3600`.
- **SEC P2-4 — Per-key active-LoRA count.** New `PER_KEY_LORA_COUNT` (default 20). Breach returns `429 per_key_lora_count_exceeded`. Decremented on `DELETE /v1/loras/{id}`.
- **SEC P2-7 — Magic-byte upload Content-Type check.** `PUT /uploads/put/{id}` now peeks the first 16 bytes of the body and rejects with `422 content_type_mismatch` when the declared `Content-Type` doesn't match the file's magic (JPEG `FF D8`, PNG `89 50 4E 47`, WebP `RIFF..WEBP`, MP4 `ftyp` at offset 4, MP3 `ID3` / `FF FB`, WAV `RIFF..WAVE`, FLAC `fLaC`, Ogg `OggS`). Lenient on `application/octet-stream` and unrecognized/missing content-types — those pass through unchanged.
- **SEC P2-8 — Dedup auto-exit-turbo.** When the Modal sidecar flaps, every failed remote-turbo job previously spawned its own `_auto_exit_turbo_on_sidecar_failure` task. Added a module-level `_exit_turbo_scheduled` flag + `_schedule_auto_exit_turbo()` wrapper: one exit task max in flight, subsequent failures log a WARN and return.
- **SEC P2-10 — Manifest type guard.** All three `approved-images/manifest.json` load paths now verify `isinstance(manifest, list)` after parsing and reset to `[]` on mismatch (with a WARN).

### Migration notes for clients

- **Admin gate** is the one user-visible change. If you've been using a single bearer for both generation AND system operations, either:
  1. Create `/mnt/nvme-1/servers/taco-backend/.admin_keys` with the operator bearer(s), one per line — recommended.
  2. Do nothing. The backwards-compat bridge keeps every `API_KEYS` entry admin. A `logger.warning` at boot (`admin auth disabled: .admin_keys is empty`) tells you the gate is degraded.
- **Per-key quota 429s** are new error codes. Clients that already handle `queue_full` can treat `per_key_queue_full` identically (same `Retry-After: 30`). Bulk-upload clients should expect `per_key_upload_quota_exceeded` with `Retry-After: 3600` once a bearer crosses 10 GiB in a 24h window.
- **`422 content_type_mismatch`** on uploads means the declared `Content-Type` header doesn't match the file's magic bytes. Either correct the header or send `application/octet-stream` (explicitly exempt).
- **`502 char_rank_schema_violation`** replaces `500 "Failed to parse vision model response"` when the vision model emits malformed JSON.

### Non-server hardening

- **SEC P2-2 — /dev/shm size guard on MP4 tmpfile** (`split_model_manager.py`). Concurrent turbo encodes (2 local + up to 4 Modal workers) could each land several hundred MB of intermediate MP4 on `/dev/shm`, and when the tmpfs ceiling was hit we saw the ltx-sidecar freeze on `kmalloc`. Added `_pick_tmp_dir(estimated_bytes)` which queries `shutil.disk_usage(config.MP4_TMPDIR)` and falls back to `/tmp` (NVMe) with a WARN log when free bytes drop below `max(estimated * 3, 2 GB)`. Every `_run_*` call site now passes an estimate derived from `num_frames × width × height × 3 × 1.2`. `_video_to_bytes`'s legacy signature still works — `estimated_bytes` is an optional kwarg defaulting to a conservative 500 MB.
- **SEC P2-5 + P2-6 — History blob caps + WAL checkpoint cadence** (`history_store.py`). `params_json` is now capped at 100 KB and `gen_config_json` at 50 KB; over-limit blobs are replaced with a `{"__truncated__": true, "original_bytes": N, "preview": "..."}` sentinel (first 4 KB of the original). Prevents a single rogue request from inflating the history row to multi-MB. Added a write counter + automatic `PRAGMA wal_checkpoint(TRUNCATE)` every 500 rows via new `checkpoint_wal(mode="TRUNCATE")` method; logs at INFO if the WAL file was >1 GB immediately before the checkpoint.
- **SEC P2-11 — Bounded retry on `IdentityFeatureTransfer` blend failures** (`flux_identity.py`). The blend-exception path silently swallowed every failure and returned the raw attention output unmodified. A shape-mismatch regression could silently produce an identity-free image across all 6 hooks × N steps, leaving the client no signal. Added `self._consec_failures` on `IdentityFeatureTransfer`: first failure logs WARN with shape info, 5 consecutive failures re-raises so `identity_session`'s `try/finally` tears down the forward hooks and the job aborts cleanly.

### Files changed

| File | Change |
|------|--------|
| `server.py` | 11 new helpers (`_require_admin`, `_constant_time_match`, `_sha256_key`, per-key counter helpers, upload-window helpers, `_content_type_matches_magic`, `_schedule_auto_exit_turbo`, `_decr_queue_on_complete`); 12 admin-gated handlers gain `request: Request` + gate check; middleware compare fixed; `CharRankResponse` / `CharRankAnalysis` / `CharRankEdits` Pydantic models + `/v2/char/rank` validation; three manifest type guards; per-key increment+decrement wired in `_submit_job` / `v2_music` / `v2_batch_submit` / `_run_music_job` / `batch_worker`; startup-time admin-posture log |
| `config.py` | `ADMIN_KEYS` loader (from `.admin_keys` / `TACO_ADMIN_KEY`); `PER_KEY_QUEUE_CAP` / `PER_KEY_MUSIC_CAP` / `PER_KEY_BATCH_CAP` / `PER_KEY_UPLOAD_BYTES_PER_DAY` / `PER_KEY_LORA_COUNT` with env-var overrides |
| `job_queue.py` | `worker_loop(..., on_complete=None)` — optional terminal-state callback invoked from the `finally` block; exceptions inside the callback logged but don't crash the worker |
| `split_model_manager.py` | Added `_pick_tmp_dir()` + `_estimate_mp4_bytes()` module-level helpers, `_SHM_MIN_FREE_BYTES` + `_DEFAULT_ENCODE_ESTIMATE_BYTES` constants, and optional `estimated_bytes` kwarg on `_video_to_bytes`. All 7 call sites (`_run_t2v`, `_run_t2v_hq`, `_run_i2v`, `_run_a2v`, `_run_retake`, `_run_outpaint` ×2) pass an estimate |
| `history_store.py` | New `_truncate_json_blob()` helper + `_HISTORY_PARAMS_MAX_BYTES` / `_HISTORY_GEN_CONFIG_MAX_BYTES` / `_HISTORY_TRUNCATED_PREVIEW_BYTES` / `_HISTORY_WAL_CHECKPOINT_EVERY` / `_HISTORY_WAL_WARN_BYTES` constants; `HistoryStore.save()` truncates before INSERT and bumps `_write_count`; new `checkpoint_wal(mode)` method |
| `flux_identity.py` | `_MAX_BLEND_FAILURES` constant; `IdentityFeatureTransfer._consec_failures` counter; `_hook_fn` logs WARN on first failure, re-raises after 5 consecutive |
| `docs/API.md` | Admin gate noted under affected endpoints; Error taxonomy additions (`admin_required`, `per_key_queue_full`, `per_key_upload_quota_exceeded`, `per_key_lora_count_exceeded`, `content_type_mismatch`, `char_rank_schema_violation`) |

---

## v1.8.1 — 2026-04-18

### Security hardening + canonical public URL + frontend service persistence

**Public base URL is now `https://api.noodlefinger.io`.** `https://taco.noodlefinger.io` was retired at the same time — its DNS record was removed, so it no longer resolves. Hard cutover, not an alias overlap. If you see DNS failures on clients pointing at `taco.` that's the reason; repoint them at `api.` and they'll work unchanged (same Cloudflare Tunnel, same origin, same auth, same request shape).

**SEC P0-1 — IDOR ownership gate on `/v2/jobs/*` and `/v2/batch/*`** (`server.py`). Before this release, any authenticated bearer could fetch / cancel any other tenant's job or batch by guessing the 128-bit ID (or via any ID leak through logs, SSE `?token=` query params, screenshots, etc.). Added a `_require_owner(owner_key, request, *, sse_token=None)` helper and injected it into 8 handlers: `GET /v2/jobs/{id}`, `/preview`, `/result`, `/stream`, `DELETE /v2/jobs/{id}`, `GET /v2/batch/{id}`, `/result/{index}`, `DELETE /v2/batch/{id}`. Cross-tenant requests now return `404 Not found` — same shape as an unknown ID (no existence oracle). Constant-time compare via `hmac.compare_digest`. Legacy jobs/batches with empty `api_key` remain accessible (backwards-compat); history + approved-images endpoints were already SQL-scoped by `api_key_hash` so they were unaffected.

**SEC P1-1+P1-2 — Dashboard + GPU telemetry moved to a LAN-only admin companion** (`dashboard_server.py`, `taco-dashboard.service`). The previous `GET /dashboard` and `GET /v1/system/gpu` were in the public server's no-auth whitelist, exposing the ops SPA and live GPU state (model, memory, temperature, utilization, tenant info, gen_config) to the internet via `api.noodlefinger.io`. Both routes are now removed from the whitelist and 401 on the public host. A tiny FastAPI companion on `192.168.1.80:8099` (LAN-bound, not routed through Cloudflare Tunnel) serves `dashboard.html` and transparently proxies every other path to `127.0.0.1:8090` with the caller's `Authorization` header. SSE streams are passed through. Access from off-LAN requires an SSH tunnel (`ssh -L 8099:192.168.1.80:8099 ...`). `taco-dashboard.service` is systemd-user-managed, enabled for boot.

**Ops: noodle-i / noodle-v / noodle-mv frontend services persisted via systemd.** After the box crash earlier today, three Vite/Express frontends (`i.noodlefinger.io`, `v.noodlefinger.io`, `mv.noodlefinger.io`) didn't auto-restart because they were running from manual `pnpm dev` invocations. Created `noodle-i.service`, `noodle-v.service`, `noodle-mv.service` with `Type=simple`, `KillMode=control-group`, `Restart=on-failure`, enabled for boot. The Cloudflare Tunnel ingress map is unchanged (`i → :5173`, `v → :5174`, `t → :5175`, `mv → :5176`, `taco → :8090`). `run-dashboard.sh` / `dashboard_server.py` sit on the new LAN-only port 8099.

### Files changed

| File | Change |
|------|--------|
| `server.py` | New `_require_owner()` helper; 8 job/batch handlers gain `request: Request` + ownership check; middleware whitelist trimmed to `/health` + `/v1/approved-images/events` only; `/dashboard` route now 404 stub |
| `dashboard_server.py` *(new, ~125 LOC)* | FastAPI on `192.168.1.80:8099`: serves `dashboard.html` + transparent proxy of `/v1`, `/v2`, `/health` etc. to `127.0.0.1:8090` with `Authorization` forwarded. SSE passthrough. Not in OpenAPI |
| `README.md`, `docs/API.md`, `docs/QUICKSTART.md` | Base URL set to `api.noodlefinger.io` + retirement notice for `taco.noodlefinger.io` |
| `~/.config/systemd/user/{noodle-i,noodle-v,noodle-mv,taco-dashboard}.service` *(new)* | systemd user units for the three frontend apps and the new admin dashboard. All `Type=simple` + `KillMode=control-group`, all enabled for boot |

### Migration notes for clients

- **Required**: repoint from `taco.noodlefinger.io` to `api.noodlefinger.io`. The old DNS is gone — clients still pointed at `taco.` get NXDOMAIN and fail immediately.
- If you hit 404 on a job that used to work cross-key — that's the IDOR fix. Use the same bearer that submitted the job.
- If you hit 401 on `GET /dashboard` or `GET /v1/system/gpu` — intentional; use the LAN admin server on 8099 (SSH tunnel off-LAN).

---

## v1.8.0 — 2026-04-18

### Flux 2 Klein identity preservation on `/v2/image-edit`

Adds three optional fields to `ImageEditRequest` for subject-identity-preserving edits on Klein, ported from [`capitan01R/ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer#identity-preservation-nodes) (MIT). Fully additive — default behaviour is unchanged.

- **`preserve_identity: bool = false`** — master switch. `false` is a zero-cost no-op path. `true` is rejected with `422 preserve_identity_klein_only` for any `model` other than `flux2-klein` (hooks target the Klein KV transformer specifically).
- **`identity_strength: float = 0.5`** — overall dial ∈ [0, 1]. Scales both internal hooks proportionally. `0.5` reproduces the upstream plugin's recommended defaults; `1.0` is maximum. `0.0` is treated as a no-op even if `preserve_identity=true`.
- **`identity_mode: "balanced" | "faithful" | "loose"` = "balanced"`** — three curated presets. Each pairs one `IdentityGuidance` mode (latent-space pull) with one `IdentityFeatureTransfer` mode (attention-output steering):
  - `balanced` = `adaptive` + `cosine_pull` (plugin default)
  - `faithful` = `direct` + `topk_replace` (stronger lock)
  - `loose` = `channel_match` + `mean_transfer` (palette/lighting fidelity, flexible geometry)

Under the hood:
- **IdentityGuidance** runs in `callback_on_step_end`, pulls the denoised latent toward the VAE-encoded first reference inside the sampling window `[0, 0.5]`. Three modes implemented: `direct`, `adaptive` (cosine-weighted), `channel_match`.
- **IdentityFeatureTransfer** uses `torch.nn.Module.register_forward_hook` on `Flux2Attention` within the middle-plus 25–88% of the 8 double-stream blocks (`transformer_blocks[2..6]` on the current Klein 9B). Three modes implemented: `cosine_pull`, `topk_replace`, `mean_transfer`. Because Klein KV caches reference K/V after step 0 (ref tokens not in subsequent attention sequences), the hook is self-gating: it observes `T_img > expected_gen_tokens` before blending.
- Both hooks share the same per-request `reference_latent` derived from resizing `image_uris[0]` to target `(width, height)` then VAE-encoding once.
- Hook install + teardown lives in `flux_identity.identity_session()` — a strict `contextmanager` with `try/finally` hook removal, important because `FluxManager._pipe` is long-lived across requests; any leaked state would corrupt subsequent non-identity edits.

### Files changed

| File | Change |
|------|--------|
| `flux_identity.py` *(new, ~340 LOC)* | `IdentityGuidance` + `IdentityFeatureTransfer` + `_resolve_identity_preset` + `identity_session` context manager |
| `server.py` | `ImageEditRequest` gains 3 fields; `image_edit` (v1) + `v2_image_edit` (v2) validate Klein-only + forward new params through the dispatch params dict |
| `flux_manager.py` | `_edit()` accepts + forwards the 3 kwargs; when active, prepares reference latent via `pipe.vae.encode(resized first image)` and wraps the pipeline call in `identity_session` |
| `docs/API.md` | New "Identity preservation" subsection under `POST /v1/image-edit` documenting preset table, known limits, and timing delta |

### Client guidance

- Existing clients are unaffected — all three fields default to zero-cost off.
- For best results: portrait-style first reference, `balanced` preset, `identity_strength=0.5–0.7`. Bump to `faithful` when the edit prompt is radically different from the reference (e.g., "now as a statue"). Use `loose` for pose / scene changes where strict pixel lock would fight the prompt.
- No new endpoint — this continues to ship through `POST /v2/image-edit` (async) and `POST /v1/image-edit` (sync).

---

## v1.7.0 — 2026-04-17

### IC-LoRA video outpaint — new `/v2/video-outpaint` endpoint

Adds a new async endpoint that expands a source video's canvas to a larger target resolution by letterboxing with pure-black padding, then uses an IC-LoRA to fill the black regions with temporally coherent generated content. Backed by [`oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint`](https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint) (Apache 2.0).

Fully additive — no existing endpoints, request shapes, headers, or response semantics changed.

- **Endpoint**: `POST /v2/video-outpaint`. Returns 202 + submission envelope, same pattern as other v2 endpoints.
- **Request**: `VideoOutpaintRequest` with `video_uri`, `prompt`, `target_resolution` (reuses existing `Resolution` literal union), `position` (9-value enum: center + 4 edges + 4 corners), `duration`, `fps`, `seed`, `enhance_prompt`, `lora` (optional override; defaults to `id="ic-lora-outpaint"`), `conditioning_strength` ∈ [0, 1], `skip_stage_2` escape hatch.
- **Pipeline**: 2-stage distilled, patterned on `_run_t2v` fast branch with an IC-LoRA `VideoConditionByReferenceLatent` appended to stage 1 conditionings:
  - Stage 1 at half target res with the outpaint LoRA fused into the distilled transformer; letterboxed source is VAE-encoded and passed as `VideoConditionByReferenceLatent` (optionally wrapped in `ConditioningItemAttentionStrengthWrapper` when `conditioning_strength < 1.0`).
  - Stage 2 (if not skipped) upsamples 2x and refines at full target res. LoRA stays fused across both stages (accepted deviation from upstream `ltx_pipelines.ic_lora.ICLoraPipeline`, which drops LoRA for stage 2; reloading mid-request would cost ~30 s of fusion work — see plan notes for the tradeoff).
- **Letterbox**: scale source proportionally to fit target, pad remainder with -1 in normalized pixel space (= RGB 0,0,0 after VAE decode = the LoRA's training black sentinel). Temporal dim padded with black frames if source is shorter than `num_frames`. `reference_downscale_factor` read from LoRA safetensors metadata (default 1).
- **Output**: silent MP4 (no audio). Source audio passthrough deferred to v1.7.x.
- **Turbo + Modal parity**: outpaint works under turbo with local cuda:1 sidecar and via the Modal pool. Modal container has the outpaint LoRA pre-staged on the HF volume at `/mnt/nvme-1/huggingface/loras/ic-lora-outpaint.safetensors` (populated by `modal run modal_app.py::download_weights`); `_dispatch_job_turbo_remote` rewrites the local LoRA path to that volume path before calling remote. Custom IC-LoRAs over remote fall back to single-machine dispatch for v1.7.0.
- **LoRA registered** under id `ic-lora-outpaint` (strategy `ic_lora_outpaint`). Download + registration script: `scripts/register_outpaint_lora.sh` (idempotent).

### Files changed

| File | Change |
|------|--------|
| `server.py` | New `OutpaintPosition` + `VideoOutpaintRequest`; new `v2_video_outpaint` handler with default-LoRA substitution; `_dispatch_job` branch for `JobType.VIDEO_OUTPAINT`; `_dispatch_job_turbo` + `_dispatch_job_turbo_remote` pass outpaint extras; `_VIDEO_JOB_TYPES` includes new type |
| `job_queue.py` | `JobType.VIDEO_OUTPAINT` enum value + `_MEDIA_TYPES` mapping (`video/mp4`) |
| `split_model_manager.py` | New module-level helpers `_read_lora_reference_downscale_factor` + `_build_outpaint_reference_latent`; new `_run_outpaint` method (2-stage, IC-LoRA conditioning); new `generate_outpaint` async wrapper; added `VideoConditionByReferenceLatent` + `ConditioningItemAttentionStrengthWrapper` + `decode_video_by_frame` imports |
| `ltx_sidecar_client.py` | `generate()` accepts `position`, `conditioning_strength`, `skip_stage_2` kwargs; payload includes them when set |
| `loras/registry.json` | New entry: `id=ic-lora-outpaint`, name `IC-LoRA Outpaint`, strategy `ic_lora_outpaint` |
| `loras/ic-lora-outpaint.safetensors` | Symlink to downloaded `ltx-2.3-22b-ic-lora-outpaint.safetensors` (1.3 GB, 960 tensors, metadata `reference_downscale_factor=1`) |
| `docs/API.md` | New `POST /v2/video-outpaint` section with request shape, position values, known limitations (dark content + silent output) |
| `scripts/register_outpaint_lora.sh` (new) | Idempotent LoRA fetch + registry insert for cold-start installs |

Companion (ops trees, not in this repo):
- `ltx-sidecar/sidecar.py`: `GenerateRequest` gains `position` / `conditioning_strength` / `skip_stage_2` + "video-outpaint" in the `job_type` Literal; new match case routes to `manager.generate_outpaint(...)`.
- `ltx-sidecar-modal/modal_app.py`: same GenerateRequest + match case additions; `download_weights()` extended to fetch the outpaint LoRA into the HF volume at `/mnt/nvme-1/huggingface/loras/` + symlink to the canonical `ic-lora-outpaint.safetensors` ID.

---

## v1.6.1 — 2026-04-17

### Hot-fix: remote sidecar can't see taco-backend's `uploads/` filesystem

Reported symptom (live): `generate_failed: [Errno 2] No such file or directory: '/mnt/nvme-1/servers/taco-backend/uploads/<uuid>'` on every `a2v` / `i2v` / `retake` dispatched to the Modal remote pool. Text-to-video worked because it has no source-media path fields.

Root cause: `_dispatch_job_turbo_remote` was passing taco-backend's local absolute paths (`audio_path`, `image_path`, `video_path`, `keyframes[].image_path`) straight through to Modal's `/generate`. Modal's container has no mount of the local `uploads/` directory, so `av.open(path)` / `Path(p).read_bytes()` fail with `FileNotFoundError`.

Fix: inline the media as base64 in the request body.

- `ltx_sidecar_client.LtxSidecarClient.generate()` gained `audio_b64`, `image_b64`, `video_b64` kwargs. When set, they go into the JSON payload alongside (or instead of) the corresponding `*_path` fields.
- `_dispatch_job_turbo_remote` in `server.py`: before calling the remote, reads each local media file (`Path(p).read_bytes()`), base64-encodes, and passes as `*_b64` with the path field set to `None`. Keyframe images get the same treatment per-entry. Raises `ValueError("remote_dispatch: media file not found: ...")` if a path doesn't exist (fail fast, not mid-call).
- Modal `/generate` (in `/mnt/nvme-1/servers/ltx-sidecar-modal/modal_app.py`): `GenerateRequest` gains the three `*_b64` fields. On arrival, any present b64 is written to `tempfile.mkstemp(prefix="modal-sidecar-", suffix=".wav|.png|.mp4")`, and the resulting path is passed downstream to the pipeline. Staged files are removed in a `finally` block regardless of outcome.
- Local sidecar (`_dispatch_job_turbo`) path is unchanged — it has direct filesystem access to `uploads/` so it keeps using the `*_path` fields directly.

Payload size impact: base64 expands 4/3. Typical audio (3–10 s): 30–100 KB → 40–135 KB. Reference image: 500 KB–2 MB → 670 KB–2.7 MB. Retake source video (5–30 s at 1080p): 10–100 MB → 13–135 MB. All within reasonable HTTP body limits.

### Files changed

| File | Change |
|------|--------|
| `ltx_sidecar_client.py` | `generate()` accepts `audio_b64` / `image_b64` / `video_b64`; payload includes them when set |
| `server.py` | `_dispatch_job_turbo_remote` reads local media files and converts to base64 before calling remote |

Companion (ops tree, not in this repo): `modal_app.py::GenerateRequest` gained the `*_b64` fields; `/generate` materializes b64 → `/tmp` and cleans up in `finally`.

---

## v1.6 — 2026-04-17

### Remote-sidecar pool with dashboard controls (up to 4 Modal workers)

Evolution of v1.5's single-remote-sidecar addition. The pool now scales 0..N on demand with a dashboard slider, giving a total of **up to 6 concurrent video workers** (2 local — main cuda:0 in-process + local cuda:1 sidecar — plus up to 4 remote Modal containers).

- `config.LTX_REMOTE_SIDECAR_MAX_WORKERS` (default 4) caps the pool. Must not exceed Modal's `max_containers` or requests queue forever.
- Modal app (`/mnt/nvme-1/servers/ltx-sidecar-modal/modal_app.py`) bumped to `max_containers=4` to match.
- `server.py` now manages `_remote_worker_tasks: list[asyncio.Task]` + `_remote_worker_target: int` (persists across turbo toggles). `_scale_remote_pool()` reconciles the live workers to match target IF turbo is active (the pool is turbo-scoped because non-video jobs submitted while turbo is off would otherwise be stolen by remote workers that can only handle video).
- New endpoints:
  - `GET /v1/system/pool` — returns `{turbo_active, remote_sidecar_configured, remote_sidecar_url, remote_worker_target, remote_worker_active, remote_worker_max}`.
  - `POST /v1/system/pool/remote-workers {"count": N}` — sets target. Scales live if turbo is on; else just stores target for next turbo-on.
- Dashboard: new "Remote Pool" row under Controls, rendered with N+1 buttons (0..MAX). Active button is highlighted; status line reflects configured / target / active / turbo-pending states.
- Backward compat: v1.5 clients that relied on "turbo-on auto-spawns 1 remote worker" still get that default — `_remote_worker_target` initializes to 1 when `LTX_REMOTE_SIDECAR_URL` is set, 0 otherwise.

### Fix: Modal /unload no longer breaks the manager

v1.5's pool scale-to-0 path called `ltx_remote_sidecar.unload()` which triggered Modal's `/unload` endpoint. That endpoint ran `manager.evict_all()`, which clears `self._workers` on `SplitModelManager`. Because `@modal.enter` only fires on container boot and not on subsequent requests, any future `/generate` against a still-warm container then failed with `"No LTX workers available — call load_all() first"`.

Fixes:
1. `_scale_remote_pool`'s scale-to-0 path no longer calls `/unload` — Modal's 5-min `scaledown_window` reclaims the GPU authoritatively.
2. Modal's `/unload` endpoint now uses `worker.evict_transformer()` per worker (frees the ~46 GB transformer while keeping the worker registry intact) instead of `manager.evict_all()`.
3. Modal's `/load` endpoint is now self-healing: if `manager.is_ready` is False (post-evict state), it re-runs `manager.load_all()` before returning.
4. Modal's `/generate` adds a defensive inline reload — if a future stale container ever exists, the first request to it triggers `load_all()` instead of 500'ing.

### Turbo toggle no longer /unloads the remote

`_exit_turbo_mode` previously looped over both sidecars and called `/unload` on each. Now it only /unloads the local sidecar (the remote is scale-down-eligible via Modal's native mechanism).

### Verified end-to-end

5 concurrent `POST /v2/text-to-video {model: ltx-2-3-fast, resolution: 1920x1080, duration: 3s}` with pool target=3 + turbo on:
- Dispatch: all 5 entered `processing` within 1 s (2 local + 3 remote warm containers).
- All 5 completed in ~60 s wall clock (local done ~50 s, Modal ~60 s with warm containers).
- 0 failures.

### Files changed

| File | Change |
|------|--------|
| `server.py` | `_remote_worker_tasks`/`_remote_worker_target`; `_scale_remote_pool()`; pool GET/POST endpoints; `_enter_turbo_mode` / `_exit_turbo_mode` now use the pool scaler instead of the single-worker v1.5 path |
| `ltx_sidecar_client.py` | (from v1.5) `auth_token` + `label` kwargs; `ltx_remote_sidecar` module instance (unchanged in v1.6) |
| `config.py` | `LTX_REMOTE_SIDECAR_MAX_WORKERS` (default 4) |
| `dashboard.html` | "Remote Pool" button grid (0..MAX) + `pollPool()` / `updatePoolUI()` / `setRemoteWorkers()` JS, polled every 5 s |
| `CLAUDE.md` / `CHANGELOG.md` | docs + version bump |

Companion (not in this repo): `modal_app.py` gained `max_containers=4`, self-healing `/load`, worker-preserving `/unload`, defensive `/generate` reload.

---

## v1.5 — 2026-04-17

### Turbo-mode hardening: systemctl-stop replaces HTTP /unload for cuda:1 tenants

Root cause (observed in production today): `_enter_turbo_mode` called `await joyai.unload()` and `await ernie.unload()` via HTTP. Those requests can succeed on the wire while the sidecar's Python process keeps tensors resident. The subsequent `ltx_sidecar.load()` then tries to allocate ~46 GB of LTX transformer on a cuda:1 that still has 44 GB of JoyAI resident → CUDA OOM, turbo-enter fails mid-sequence, ACE already stopped, leaving the system in a broken state.

Fix (`server.py`):
- New `_systemctl_unit(unit, action)` helper — runs `systemctl --user <action> <unit>` in a thread, raises `RuntimeError` with stderr on non-zero exit. Replaces per-service `_ace_systemctl` (kept as back-compat alias).
- New `_stop_cuda1_tenants()` — stops `ace-step`, `joyai-sidecar`, `ernie-image-sidecar`, and any stale `ltx-sidecar` via systemctl. Best-effort; "already stopped" is not an error.
- New `_restore_cuda1_tenants()` — inverse: systemctl-start each configured tenant (`LOAD_*=1`). Called at turbo exit AND on turbo-entry abort rollback.
- New `_wait_cuda1_free(threshold_mib=2000, timeout_s=20.0)` — polls `nvidia-smi` until cuda:1 drops below the threshold. Returns False on timeout.
- New `_list_cuda1_processes()` — enumerates compute-app PIDs on the cuda:1 bus for diagnostics.
- `_enter_turbo_mode` rewritten: Flux unload → systemctl-stop all cuda:1 tenants → **wait for cuda:1 to drain (20 s deadline)** → abort with detailed error + tenant restore if not drained → systemctl-start ltx-sidecar → poll /health → /load → spawn workers. No more OOM on turbo entry.
- `_exit_turbo_mode` rewritten: HTTP /unload both sidecars (graceful) → systemctl-stop ltx-sidecar → `_restore_cuda1_tenants()`.

### LTX remote-sidecar pool (3-worker turbo)

Turbo mode previously topped out at 2 concurrent video workers (main cuda:0 in-process + local cuda:1 sidecar). v1.5 adds an OPTIONAL third worker that dispatches to a remote HTTP sidecar — e.g., Modal RTX Pro 6000 for overflow capacity.

- `config.py` — new env vars `LTX_REMOTE_SIDECAR_URL` + `LTX_REMOTE_SIDECAR_TOKEN`. When URL is empty (default), behavior is unchanged from v1.4. When set, turbo enter warms the remote via `/health` then spawns a third `worker_loop` dispatching via `_dispatch_job_turbo_remote`.
- `ltx_sidecar_client.py` — `LtxSidecarClient` gained `auth_token` + `label` kwargs. `_headers()` injects `Authorization: Bearer <token>` when configured. Module now exposes two instances: `ltx_sidecar` (local, label="local") + `ltx_remote_sidecar` (label="remote", or None if not configured).
- `server.py::_dispatch_job_turbo_remote` — routes to the remote client. **Unlike the local path, remote transport failures do NOT auto-exit turbo** — the remote is treated as optional extra capacity; jobs on that worker fail individually but main + local-sidecar workers keep serving.
- `_exit_turbo_mode` also /unloads the remote (saves Modal credit burn if the backing host is scale-to-pay like Modal).

Verified live: 3 concurrent fast t2v submissions → `processing: 3` in queue → 3 parallel workers → all completed in ~40 s wall-clock (vs ~90 s sequential).

### Companion: Modal RTX Pro 6000 LTX sidecar deployment

Scaffolded at `/mnt/nvme-1/servers/ltx-sidecar-modal/modal_app.py` (NOT in the taco-backend repo — ops tree). Includes:
- Custom image: debian_slim + torch cu130 + transformers 5.3.0 (pinned — Gemma3TextConfig attr mismatch in older transformers) + local `/mnt/nvme-1/repos/LTX-2` editable install (user has uncommitted `getattr(rope_local_base_freq)` fallback that upstream master lacks).
- `modal.Volume` at `/mnt/nvme-1/huggingface` pre-populated with 125 GB of LTX-2.3 checkpoints + Gemma 3 12B PT from HF.
- `@app.cls` + `@modal.enter` eager-loads the model per container boot. Cold start: ~60–80 s. Warm: instant.
- FastAPI app with Bearer-token middleware (secret `taco-sidecar-auth`).
- Public URL: `https://tacos8me--taco-ltx-sidecar-ltxsidecar-fastapi-app.modal.run`

Free Modal credit: $30/mo → ~10 hrs of RTX Pro 6000 ($3.03/hr). Scales to zero after 10 min idle (no burn when unused).

### Files changed

| File | Change |
|------|--------|
| `server.py` | `_enter_turbo_mode` + `_exit_turbo_mode` rewritten; new systemctl + cuda:1 drain helpers; `_dispatch_job_turbo_remote`; remote-worker spawn logic |
| `ltx_sidecar_client.py` | `auth_token` + `label` kwargs; module-level `ltx_remote_sidecar` instance |
| `config.py` | `LTX_REMOTE_SIDECAR_URL` + `LTX_REMOTE_SIDECAR_TOKEN` env vars |
| `CLAUDE.md` / `CHANGELOG.md` | version bump + docs |

---

## v1.4.1 — 2026-04-16

### Hot-fix

- **`import subprocess` missing from `server.py` top-level** — `_enter_turbo_mode` (line 1436) invokes `subprocess.run(["systemctl", "--user", "start", "ltx-sidecar"], ...)` inside a lambda, and `_warmup_page_cache` (line 529) invokes `subprocess.run` in an `asyncio.to_thread(...)` call. Both resolve `subprocess` via module scope, where it wasn't imported. Any `POST /v1/system/turbo {enable:true}` hit `NameError: name 'subprocess' is not defined` and left the system in a half-transitioned state (ACE stopped by `_ace_systemctl("stop")` before the lambda executed; LTX sidecar never started; no turbo dual-worker). Latent bug present since turbo mode landed in v1.2 — the page-cache warmup was also silently failing (`asyncio.create_task` swallowed the unhandled exception). The two functions that imported `subprocess` locally (`_ace_systemctl` at ~1078, `_query_gpu_info` at ~1324) worked fine, which masked the broader issue.

Symptom in production today: `POST /v1/system/turbo {"enable":true}` → 500 `turbo_toggle_failed`; queued gens stuck because main worker held `_inference_lock` while `_enter_turbo_mode` crashed mid-handshake.

Fix: one-line `import subprocess` at module top, covering all four call sites. No behavior change for previously-working paths.

**File**: `server.py`.

---

## v1.4 — 2026-04-16

Five semi-independent changesets landed together.

### Full-fidelity history capture (schema v2)

History DB now stores everything needed to re-run a generation exactly.

- **Schema v2 migration** — four new columns on `generations`: `params_json` (raw Pydantic request body with `storage://` URIs preserved), `gen_config_json` (LTX `_gen_config` snapshot at dispatch time, or `{turbo_steps, turbo_guidance}` for Flux-turbo), `seed` (resolved integer — auto-generated if client omits), `enhanced_prompt` (LTX prompt rewrite when `enhance_prompt=true`). Online `ALTER TABLE ADD COLUMN` gated on `PRAGMA user_version`; idempotent; old rows left intact with NULL new columns.
- **New endpoint** `GET /v2/history/{generation_id}` — returns the full record including parsed `params` + `gen_config`. Bearer auth, 404 for not-yours-or-not-found. `/v2/history` list endpoint shape unchanged (backward compat preserved).
- **Path → URI sanitizer** (`_sanitize_params_for_history`) rewrites `image_path` / `audio_path` / `video_path` / `source_audio_path` / `reference_audio_path` / `image_paths` list / `keyframes[].image_path` back to stable `storage://<uuid>` form before persistence.
- **Enhanced prompt plumbing** — `on_prompt_enhanced` callback threaded through `split_model_manager._encode_prompts` → 5 `_run_*` methods → 4 public async wrappers. Dispatcher captures the rewritten text onto `Job.enhanced_prompt`; worker_loop ships it to history.
- **Files**: `history_store.py`, `job_queue.py`, `server.py`, `split_model_manager.py`, `docs/API.md`, `CLAUDE.md`.

### Flux dashboard controls

Dashboard now has a collapsible "Flux" section exposing the turbo sub-parameters that previously required server restart to change.

- `_flux_config` dict (2 tunables: `turbo_steps` default 8, `turbo_guidance` default 2.5), persisted to `.flux_config.json`, survives restart.
- Endpoints: `GET /v1/system/flux-config`, `POST /v1/system/flux-config` (merge-update), `POST /v1/system/flux-config/reset`.
- `flux_manager._generate` / `_img2img` / `_edit` gained `turbo_steps` + `turbo_guidance` kwargs; dispatcher injects from `_flux_config`.
- `gen_config_snapshot` captures the turbo subset when `turbo=true` so history reflects what actually ran.
- **Files**: `server.py`, `flux_manager.py`, `dashboard.html`, `config.py`.

### PR 1 — perf quick-wins

Small, uncontroversial latency savings across every CFG-enabled video gen.

- **P1 negprompt cache** — `DEFAULT_NEGATIVE_PROMPT` encoded once per encoder lifecycle (nulled in `evict_all`); subsequent CFG-path gens skip the 0.4–0.8 s Gemma encode. Lives on encoder device (cuda:0); survives CPU↔GPU paging because the cached tensor is independent of the encoder's parameter tensors.
- **P4 redundant synchronize drop** — removed `torch.cuda.synchronize()` after `text_encoder.encode` in `_encode_prompts`. Same default stream already serializes; the subsequent `.to(target)` syncs implicitly.
- **P5 MP4 tmpfile on tmpfs** — `_video_to_bytes` now writes the PyAV intermediate to `/dev/shm` (verified tmpfs) via new `config.MP4_TMPDIR`. Saves 50–200 ms/job on ext-backed `/tmp` (confirmed by `stat -f /tmp` → ext2/ext3). Fallback to `/tmp` if shm missing.
- **P7 tqdm TTY auto-detect** — `tqdm(range(...), disable=None)` silences the per-step progress bar under systemd (no TTY). Cleans up journalctl, saves ~1 ms × steps.
- **O3 timed encoder** — wrapped `_encode_prompts` in `_timed("encode_prompts")`; acts as proof-of-P1 post-deploy (once root logger is configured to reach journal).
- **Files**: `split_model_manager.py`, `config.py`.

### PR 2 — ops resilience

Four independent defensive changes; O-A is live-verified, the other three activate only in failure modes.

- **O-A cancellation propagation** — `DELETE /v2/jobs/{id}` now actually stops the LTX denoiser. New `GenerationCancelledError` raised from `ProgressDenoiser.__call__` when `job.status == CANCELLED`; the sigma loop unwinds naturally. `worker_loop` distinguishes cancellation from failure (status → `cancelled`, not `failed`; no error recorded). Verified: GPU util 100 % → 0 % within 3 s of DELETE at 11 % progress. Previously the denoiser would have kept burning for ~25 more steps.
- **O-B LTX OOM recovery** — new `_oom_recovery(worker)` context manager + `@_with_oom_recovery` decorator applied to all five `_run_*` methods. On CUDA OOM: evict transformer + `cleanup_memory()`, then re-raise. Mirrors the `flux_manager.py` pattern; prevents the classic failure where a mid-VAE-decode OOM leaks ~22 GB into the allocator cache and OOMs every subsequent request.
- **O-C half-load recovery** — new `SplitModelManager.reset()` nulls workers + encoder_ledger + the neg-prompt cache, then per-GPU sync + `empty_cache()`. `_load_all_impl` wrapped so `_last_load_failed` is set on exception; `_ensure_ltx_resident` calls `reset()` before retry when the flag is up. Prevents blind `load_all()` retries against partially-populated GPU memory.
- **O-D sidecar crash → auto-exit turbo** — `_dispatch_job_turbo` catches `LtxSidecarError` with status 502/503/504 (transport failures) and schedules `_auto_exit_turbo_on_sidecar_failure` via `asyncio.create_task` so the next queued job doesn't fail the same way. Job-level errors (4xx) don't trigger.
- **Files**: `split_model_manager.py`, `server.py`, `job_queue.py`.

### PR 3 — cleanup audit (documentation + 2 safe drops)

Audited the 10 open-coded `gc.collect() + torch.cuda.synchronize() + torch.cuda.empty_cache()` triples in `split_model_manager.py`. Finding:

- **Do not dedup to `cleanup_memory()`**: the helper uses current-device sync, but our multi-GPU paths (`evict_transformer`, `evict_all`, `reset`) require explicit per-device sync (`torch.cuda.synchronize(self.device)` / `torch.cuda.synchronize(torch.device(device_name))`). Blind dedup would silently regress DUAL_GPU_LTX correctness.
- **Two sites are truly redundant**: back-to-back `gc.collect()` after `worker.evict_transformer()` in `_run_retake` (the encode-prep path and the pre-VAE-decode path). `evict_transformer` already calls gc+sync+empty internally; a second gc immediately after picks up nothing. Dropped both.
- **Eight sites are load-bearing**: added multi-line comments at `evict_transformer`, `ensure_transformer`, `_page_encoder_to_cpu`, `evict_all`, and the four identical inter-stage cleanup blocks in `_run_t2v` / `_run_t2v_hq` / `_run_i2v` / `_run_a2v`. Comments explain why the cleanup can't be safely dedup-ed (device-specific sync, multi-GPU loop, flush of just-cleared auxiliary model refs before VAE decode's ~15 GB peak). Future optimizers: don't strip these.

Measured savings: ~40 ms on retake only. The original "100–300 ms/job" estimate over-promised — most sites exist for memory-safety, not ceremony.

### Infra

- `.gitignore`: `.flux_config.json`, `*.bak-pr*`, `cufile.log`.
- Backup files `*.bak-pr1-*` / `*.bak-pr2-*` / `*.bak-pr3-*` on disk for rollback; ignored by git.

### Deferred (not shipped)

Tracked as task IDs in the working-set task list for future PRs:

- **#95** — Gemma `tokenizer.chat_template` missing: blocks `enhance_prompt=true` on all LTX modes. Discovered during v1.4 smoke. Enhance_prompt history column is wired and will populate correctly once the tokenizer is fixed.
- **#96** — Smoke test for remaining gen types (i2v/a2v/retake/i2i/image-edit/ernie) with uploaded source media. Text-to-image + text-to-video + music were smoke-tested; the source-media-requiring types need a manual test pass.
- **#100** PR 4 (feature trivials): G1 client seed on video, G7 retake boundary validation, G8 retake `enhance_prompt` field, G9 batch priority honoring, G10 stage-2 latent preservation on OOM.
- **#101** PR 5 (feature smalls): G2 custom `negative_prompt` on video, G3 HQ guider params aligned to upstream (`stg_scale=0.0, rescale_scale=0.45`), G5 per-request `gen_config` override.
- **#102** PR 6: P2 queue-aware encoder residency — skip `_page_encoder_to_cpu` when the next queued LTX job fits alongside the encoder (fast-mode batches save ~10 s / 5 jobs).
- **P8** (after #95 lands): cache `generate_enhanced_prompt` output keyed by `(prompt_hash, image_hash, seed)`; noodle-i Char loop repays the 1–3 s Gemma generation every iteration otherwise.
- **G4** (after #95 lands): route `enhance_i2v` for image-to-video mode; currently we always call `enhance_t2v` and ignore the reference image.
- **Structural**: pinned-host-memory base-weight cache for `ensure_transformer` — HQ jobs cycling `dev_lora_025 → dev_lora_050` mid-generation reload from disk every swap. Real refactor, own epic.

---

## v1.3 — 2026-04-13

- ERNIE-Image sidecar (8B DiT text-to-image on cuda:1 port 8094, Apache 2.0). Swaps with JoyAI on cuda:1; coexists with ACE.
- Dashboard advanced controls: 14 tunable LTX generation parameters (sampler, steps, scheduler shifts, CFG/STG/rescale/modality scales, stage-2 sigmas, eta). Persisted to `.gen_config.json` via `GET/POST /v1/system/config`.
- CFG++ sampler (euler_ancestral_cfg_pp ported from ComfyUI, adapted to LTX-2 CONST flow-matching) — now default.
- LTX-2.3 v1.1 distilled models.
- SingleGPUModelBuilder + CachingModelFactory replacing ModelLedger.
- BatchSplitAdapter on every transformer call.
- bf16 reduced-precision accumulation restored to PyTorch default (was previously False — caused character movement artifacts).

## v1.2

- Turbo mode: dual-GPU LTX via cuda:1 sidecar for 2 concurrent video workers.
- ACE music sidecar on cuda:1 (18 GB, xl-base + 4B LM via vLLM).
- JoyAI image-edit sidecar migrated from cuda:0 to cuda:1.
- Dashboard + `GET /v1/system/gpu` telemetry endpoint.
- Batch scheduler (1–50 items, priority stored).
- v2 job observability: phases, SSE stream, `/v2/jobs/{id}/preview`.

## v1.1

- Single-GPU swap mode (LTX ↔ Flux auto-swap on cuda:0).
- Keyframe symbolic indices (`"first" | "middle" | "last"` + negative ints).
- Flux LoRA folder-drop discovery (adapter mode, strength changes free).
- Fast-mode audio-to-video.
- History store (SQLite, WAL, thumbnails, 30-day retention).
- Approved images pipeline (noodle-i → noodle-v).

## v1.0

Initial release.
