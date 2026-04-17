# AGENTS.md — taco-backend

## Performance Optimization Backlog

Identified via full codebase audit (2026-03-14). Items grouped by tier.

### Tier 1 — Immediate, Low Risk

1. **`CUDA_MODULE_LOADING=LAZY` in run.sh** — 20-40s faster cold start
2. **CRF 18 + preset fast on encode_video** — patch media_io.py:205, current CRF 23/medium is slow and soft
3. ~~**Flux PNG → WEBP output**~~ — DONE (quality 95)
4. **FileResponse for v2 results** — server.py job result endpoint, saves ~300MB RAM per fetch
5. **Cache detect_params at module level** — split_model_manager.py, eliminates 4 safetensors header reads/request
6. **Atomic LoRA registry write** — lora_registry.py:60, write-to-temp-then-rename
7. **TOKENIZERS_PARALLELISM=false in run.sh** — prevents HF tokenizer deadlocks

### Tier 2 — Medium Effort, High Value

8. **Fix sigma schedule on pro/i2v-pro** — pass `latent=empty_latent` to LTX2Scheduler().execute(), currently uses hardcoded 4096 tokens (quality bug)
9. **FP8 cast quantization** — add QuantizationPolicy.fp8_cast() to ModelLedger, saves ~11GB VRAM per transformer
10. **Cache negative prompt embeddings** — cache ctx_n for DEFAULT_NEGATIVE_PROMPT, saves ~0.2s/request
11. **Remove redundant cleanup_memory() after evict** — replace 5 post-evict cleanup_memory() with gc.collect()
12. **AAC audio bitrate 128k → 192k** — media_io.py audio stream
13. ~~**TF32 + cudnn.allow_tf32 at startup**~~ — REVERSED: TF32 now DISABLED. Was degrading VAE decode quality on Blackwell (VAE uses force_upcast=True expecting real float32, TF32 reduced mantissa 23→10 bits)
14. **BytesIO for encode_video** — eliminate temp file disk I/O in _video_to_bytes (~100ms/gen)

### v1.1 Features (SHIPPED)

19. ~~**First/mid/last keyframes**~~ — DONE. Symbolic frame_index ("first", "middle", "last") + negative integers (-1 = last, -12 = landing room). Resolved server-side after num_frames computed.
20. ~~**Bounds checking**~~ — DONE. frame_index >= num_frames → 422.
24. ~~**Flux 2 LoRA (folder-drop)**~~ — DONE (2026-04-09). `flux_lora_registry.py` scans `flux_loras/` for `.safetensors`; optional sidecar `.json` for metadata; `GET /v1/flux-loras` + `POST /v1/flux-loras/rescan`; `lora: {id, strength}` field on Flux image requests; adapter mode with `(model, lora_path)` cache key. Works on both flux2-dev and flux2-klein.
25. ~~**Dithering artifacts on flux2-dev fixed**~~ — DONE (2026-04-06). Root cause: FP8 layerwise casting was quantizing `x_embedder` and `context_embedder` (input projections) into FP8, losing precision on the initial 128→6144 and 15360→6144 projections. Error propagated through all 56 transformer layers. Fix: `skip_modules_pattern=["x_embedder","context_embedder","proj_out"]` on both dev and klein `enable_layerwise_casting()` calls. Klein had already been set correctly; dev was not. Also stopped hard-fusing Turbo LoRA at load time (it's now a regular folder-drop LoRA with ID `flux2-turbo`) — the fused turbo weights + FP8 cast were shifting weights to non-standard FP8 grid points creating structured dithering patterns. See `docs/audit-2026-04-06-comfyui-comparison.md`.
26. ~~**Char mode vision ranking**~~ — DONE. `POST /v2/char/rank` routes to Gemma 4 31B (`gemma-4-31b-it` on llama-swap) via `chat_manager.generate_chat_completion(..., model=...)`. Returns structured JSON with face_match/eyes/proportions/overall_likeness + add/remove/modify edits. Client-side loop in noodle-i drives iterative refinement.
27. ~~**Full bf16 Flux 2 Dev (drop FP8 layerwise casting, adapter-mode LoRA)**~~ — DONE (2026-04-09, v1.1.1). Root cause: the v1.1 folder-drop LoRA feature reintroduced the exact `fuse_lora → enable_layerwise_casting(float8_e4m3fn)` sequence that was previously diagnosed as causing structured dithering with the hardcoded Turbo LoRA (entry 25, 2026-04-06). Diffusers PR #10685 documents the root cause: PEFT's input autocast hook forces compute back into FP8, defeating `compute_dtype=bfloat16`, and fused weights sit on non-standard FP8 grid points creating structured artifacts. Fix: (a) drop `enable_layerwise_casting` entirely — run full bf16 transformer (matches ComfyUI default per their issue #10087); (b) add `pipe.enable_model_cpu_offload(device="cuda:0")` on Dev branch (bf16 all-resident is 105.9 GB > 96 GB, so page TE↔GPU around prompt encoding); (c) switch LoRAs from fusion to **adapter mode** via `load_lora_weights(adapter_name="user_lora")` + runtime `set_adapters(["user_lora"], [strength])`. Cache key shrinks from `(model, (path, strength))` to `(model, path)`, so **strength-slider changes are now free** (solves the v1.1 reload-on-every-tick UX bug). Klein fits full bf16 resident without offload. Verified end-to-end: prototype script generates 4 back-to-back images at strengths 1.0/0.4/0.0/disable without a single pipeline reload, no OOM, no screendoor.
32. ~~**JoyAI-Image-Edit integration via sidecar (v1.1.8)**~~ — DONE (2026-04-11, commit `83fe611` for taco-backend, plus `/mnt/nvme-1/servers/joyai-sidecar/` created out-of-tree). `model="joyai-edit"` added to `/v1/image-edit` and `/v2/image-edit` endpoints. Sidecar runs `JoyImageEditPipeline` from the Moran232/diffusers fork + transformers 4.57.1 on 127.0.0.1:8092. Process isolation needed because the fork patches diffusers core registry files that can't be vendored (PR huggingface/diffusers#13444). Phase 0 VRAM gate passed: 65.5 GB peak reserved at 1024² / 30 steps, well under 80 GB threshold. Three-tenant swap protocol (LTX ↔ Flux ↔ JoyAI) added to `_inference_lock` via `_last_gpu_tenant` tracker and `_evict_other_tenants(new)` helper. All `_ensure_*_ready()` helpers now `async def`. Bug fix rider: `flux_manager._edit()` previously hardcoded Klein even when `model="flux2-dev"` was requested. LoRA not supported on joyai-edit (422). Single-image constraint enforced at handler level. Chat-template prompt wrapping done server-side. Full plan in `/home/ian/.claude/plans/melodic-sniffing-beacon.md`.

31. ~~**SSE job stream endpoint**~~ — DONE (2026-04-11, **v1.1.7**, commit `da82ecb`). `GET /v2/jobs/{id}/stream` was advertised as `stream_url` in the submission envelope but never actually implemented. Now: EventSource-compatible, emits JSON snapshots (same shape as the poll endpoint) on every `(status, progress, phase, error_code)` change, closes on terminal state, `: keepalive` every 15 s. Dual auth: bearer header OR `?token=` query param for browsers (via existing `POST /v1/sse-token`). Middleware bypass for `/v2/jobs/*/stream` plus `config.API_KEYS`-aware handler. Replaces the 240-GET/job polling loop with one long-lived connection. 3 new tests: unknown-job 404, missing-auth 401, completed-job final-event-then-close.

30. ~~**Post-denoise observability (phase field + async history save + WAL)**~~ — DONE (2026-04-11, **v1.1.6**, commit `eab8ae4`). Addresses the user-reported "stuck at 95%" perception bug plus the synchronous thumbnail stall that was blocking the queue worker. Four parts in one commit:

    **(a) Phase observability.** `_wrap_denoise` cap reduced from 0.99 → 0.90 (`split_model_manager.py::_wrap_denoise` line 492); the top 10% of progress is reserved for post-denoise phases. New `Job.phase` field (`job_queue.py`), surfaced in `/v2/jobs/{id}` status response. `split_model_manager._run_*` methods emit `on_progress(0.90, phase="decoding")` before VAE decode and `on_progress(0.95, phase="encoding")` before `_video_to_bytes`, all 5 methods (`_run_t2v`, `_run_t2v_hq`, `_run_i2v`, `_run_a2v`, `_run_retake`). `flux_manager._generate/_img2img/_edit` take a new `phase_sink` kwarg and emit `phase="encoding"` before `_to_webp`. Worker_loop flips `phase="saving"` at 0.99 before the upload-store write, then `None` on COMPLETED. `make_flux_callback` also capped at 0.90. `on_progress` in `server.py::_dispatch_job` short-circuits on `JobStatus.CANCELLED` so stale executor callbacks can't rewrite phase on cancelled jobs.

    **(b) Async history save.** `job_queue.py::worker_loop` wraps `history.save()` in an `asyncio.create_task(asyncio.to_thread(...))` fire-and-forget. Queue `task_done()` fires immediately; next job dequeues without waiting ~300 ms for PyAV first-frame decode + JPEG encode + SQLite commit. After save completes, the task backfills `job.preview_bytes` from the on-disk thumbnail so `/v2/jobs/{id}/preview` fast-path hits on the next poll. Dead `update_progress` and `make_progress_callback` helpers removed.

    **(c) Preview endpoint reuses saved thumbnail.** `server.py::v2_job_preview` (around line 1160) reordered to a 4-path flow: cached in-memory → on-disk `FileResponse` from `config.THUMBNAIL_DIR / f"thumb_{upload_id}"` (zero-copy sendfile) → lazy PyAV fallback (now offloaded via `asyncio.to_thread`) → 204. Eliminates the 100+ MB `result_path.read_bytes()` blocking read on the event loop for every first-time preview request.

    **(d) SQLite WAL mode.** `history_store.py::HistoryStore.__init__` runs `PRAGMA journal_mode=WAL` after the schema creation. Readers (`/v2/history`, `/v2/history/{id}/*`) no longer block behind the single writer (`history.save`). `.gitignore` updated to cover `history.db-wal` and `history.db-shm` sidecars (`history.db` → `history.db*`).

    **Timing logs** added at every phase boundary via a `_timed` context manager (split_model_manager.py + flux_manager.py). Grep `journalctl -u taco-backend | grep -E "vae_decode|encode_video|flux_webp_encode|history.save"` for real wall-clock per phase.

    **`docs/API.md`** updated in the same commit per the CLAUDE.md contract rule: `phase` field documented in the v2 status schema; v1.1.6 changelog entry.

    76 → 79 tests passing across both commits.

29. ~~**Video thumbnail extraction + preview 204**~~ — DONE (2026-04-09, **v1.1.5**, commit `b1f37e9`). `history_store._make_thumbnail` was silently failing on every video job with `PIL.UnidentifiedImageError: cannot identify image file` because MP4 bytes aren't a PIL-parseable image. Fix: detect MP4 via the ISO-BMFF `ftyp` box (`_is_mp4_bytes`), decode the first frame via PyAV (`_first_video_frame_as_pil`), then thumbnail-path unchanged. Also: `/v2/jobs/{id}/preview` now returns `204 No Content` (not 404) when no preview is available, and lazy-extracts the first frame on demand for completed video jobs that never ran through a live step-end callback. Caches on `job.preview_bytes` so subsequent polls are free.

28. ~~**Single-GPU swap mode + LTX evict leak fix**~~ — DONE (2026-04-11, **v1.1.4**, commits `d41b742` + `ab9fac8`). Two-part change that consolidates taco-backend onto `cuda:0` and frees `cuda:1` for external training.

    **(a) evict_all leak fix** (`split_model_manager.py::evict_all`, commit `d41b742`). Root cause: each `DenoiserWorker` holds TWO strong references to its source `ModelLedger` — direct via `worker._model_ledger` (set at split_model_manager.py:392) AND indirect via `worker.ledger._source_ledger` (from `CachingModelLedger(source_ledger=...)`). The ModelLedger's internal registry + weight builders kept GPU tensors pinned even after `worker.cache[key] = None` and `self._workers.clear()`. Result: ~22 GB of encoder-hub weights (Gemma text encoder + VAE + spatial upsampler) stayed resident on the LTX device after `/v1/ltx/unload`. Fix: explicitly null both reference paths in `evict_all()` before dropping the workers list, and null `_encoder_ledger._source_ledger` defensively. **Verified live: cuda:0 drops from 66.9 GB → 683 MiB after `/v1/ltx/unload`** (99% reclamation, vs 22+ GB residual before the fix).

    **(b) Single-GPU swap mode** (`server.py` + `config.py`, commit `ab9fac8`). `LTX_DEVICE` changed from `"cuda:1"` → `"cuda:0"`; `FLUX_DEVICE` stays `"cuda:0"`. Both managers target the same physical GPU. **`cuda:1` is now reserved exclusively for external training runs** — taco-backend never touches it. (Rationale: an ai-toolkit training run `train_flux2_klein_bong_a_v9_1.yaml` was active on cuda:1 during the refactor and made steps 506 → 1001 uninterrupted through the restart + test cycle.) LTX and Flux are mutually exclusive on cuda:0: ~79 GB LTX (60 GB transformer + 19 GB encoder hub + decoder activations) + ~81 GB Flux (60 GB transformer + 14 GB `enable_model_cpu_offload` forward-pass peak) = ~160 GB > 96 GB. They cannot coexist during forward pass. **Auto-swap helpers** added to server.py: `_ensure_ltx_resident()` (no-op if ready, else `manager.load_all()`) and `_ensure_flux_ready()` (no-op if LTX not ready, else `manager.evict_all()`). Both MUST be called while holding `_inference_lock`. Wired into `_dispatch_job()` (v2 async queue) and all 7 v1 sync handlers (text_to_video, image_to_video, audio_to_video, retake, text_to_image, image_to_image, image_edit), inside the `async with _inference_lock` block before `torch.cuda.set_device()`. The `if not manager.is_ready: return 500` early-return guards were removed from the 4 LTX sync handlers — auto-swap handles readiness lazily. New endpoints: `POST /v1/ltx/{unload,reload}` mirroring existing `/v1/flux/{unload,reload}` pair, both Bearer-auth. LTX is **not** auto-reloaded after a Flux request — stays evicted until the next video request. Long-stretch image-only or video-only workloads have zero swap overhead; mixed workloads pay per direction change: LTX→Flux ≈ +3 s eviction, Flux→LTX ≈ +7–30 s cold load depending on OS page cache. Verified: cuda:0 at rest 789 MiB idle, training PID 1956281 unaffected throughout, Flux image request 45.8 s end-to-end with auto-evict, 76 unit tests passing.

### v1.2 Features (SHIPPED)

33. ~~**v1.2 — ACE music gen + dual-GPU layout + batch scheduler + turbo mode**~~ — DONE (2026-04-11, commits `25722df`, `61dae88`, `4911bdb`).
    - **ACE music integration** (`25722df`): ACE Step xl-base+LM on cuda:1:8001, `ace_client.py` httpx proxy, `POST /v1/music` (sync) + `POST /v2/music` (async), `MusicGenerationRequest` with 30+ fields (task types: text2music/cover/repaint/extract/lego/complete), `LOAD_ACE` env var, `MAX_MUSIC_PENDING` queue cap.
    - **Dual-GPU layout** (`25722df`): cuda:0=LTX+Flux (2-tenant swap), cuda:1=ACE+JoyAI (coexisting, no swap). JoyAI migrated from cuda:0 to cuda:1. Dashboard at `/dashboard`, GPU telemetry at `/v1/system/gpu`. Scheduling hardening: pause/resume/manual-endpoint tracker fixes.
    - **Batch scheduler** (`61dae88`): `POST /v2/batch` + `GET /v2/batch/{id}` + `DELETE /v2/batch/{id}`. 1-50 items, swap-optimized sort (images before videos, Klein before Dev). `MAX_BATCH_QUEUE_DEPTH`, `MAX_BATCH_ITEMS` config.
    - **Turbo mode** (`61dae88` + fix `4911bdb`): `POST /v1/system/turbo` toggles dual-GPU LTX — 2 concurrent denoiser workers (one per GPU), 2 video jobs at a time. ACE+JoyAI+Flux evicted, return 503. `worker_loop` turbo_check callback, `batch_worker` asyncio.gather for 2-at-a-time processing. Dashboard turbo controls.
    - 111 tests passing.

### VAE Tiling Notes (from ComfyUI comparison)

ComfyUI uses spatial tiling (512/64px) + temporal_size=4096 (effectively no temporal tiling). Our approach:
- SHORT_VIDEO_THRESHOLD=257 — no tiling for ≤10s videos (best quality, single-pass)
- Temporal-only tiling (128/32) for longer videos
- Spatial tiling disabled (caused grid seams)
- If re-enabling spatial tiling: use 512/64 with cosine S-curve blending (already in ltx-core)

### v1.3 Features (SHIPPED)

34. ~~**Upstream LTX-2 migration**~~ — DONE (2026-04-13). ModelLedger → SingleGPUModelBuilder, CachingModelLedger → CachingModelFactory. New Denoiser classes: SimpleDenoiser, GuidedDenoiser, FactoryGuidedDenoiser. Updated sampler signatures.
35. ~~**ltx-core 1.1.1 + ltx-pipelines 1.1.1**~~ — DONE. Upstream sync with vocoder fp32 fix, cosine tiling, layer streaming, BatchSplitAdapter.
36. ~~**v1.1 distilled models**~~ — DONE. `ltx-2.3-22b-distilled-1.1.safetensors` + `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` + `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`.
37. ~~**CFG++ sampler**~~ — DONE. Ported from ComfyUI's `euler_ancestral_cfg_pp`. Default ON. Togglable via `GET/POST /v1/system/sampler` and dashboard UI. Uses alpha=(1-sigma) rescaling for better motion quality.
38. ~~**DUAL_GPU_LTX mode**~~ — DONE. `DUAL_GPU_LTX=1` env flag for 2 concurrent video workers via LTX sidecar on cuda:1:8093. Disables Flux, ACE, JoyAI at boot.
39. ~~**Batch result download**~~ — DONE. `GET /v2/batch/{id}/result/{index}` endpoint for individual batch item result files.
40. ~~**BatchSplitAdapter**~~ — DONE. All transformer calls wrapped in `BatchSplitAdapter(max_batch_size=1)` for correct multi-pass batching.
41. ~~**bf16 precision fix**~~ — DONE. Removed forced float32 accumulation (`bf16_reduced_precision_reduction`). Training/inference mismatch caused character movement artifacts across 56 layers × 20 steps.
42. ~~**A2V fixes**~~ — DONE. GuidedDenoiser (static) for stage 1, frozen audio noise_scale=0.0, null check, padding.
43. ~~**TilingConfig.default()**~~ — DONE. Upstream cosine tiling for VAE decode replaces custom tiling config.
44. ~~**Sidecar timeout**~~ — DONE. 300s → 600s for LTX sidecar and ACE sidecar generate calls.
45. ~~**torch.compile flag**~~ — DONE. `TORCH_COMPILE=1` env flag available but default OFF (no benefit on Blackwell with cuDNN FA4).
46. ~~**Dashboard advanced controls + config API**~~ — DONE. 14 tunable generation parameters exposed in `/dashboard` (sampler, step counts, scheduler shifts, CFG/STG/rescale/modality scales, stage 2 sigmas, eta controls). Preset dropdowns and reset button. `GET/POST /v1/system/config` + `POST /v1/system/config/reset` endpoints. Persisted to `.gen_config.json`, survives restarts.
47. ~~**ERNIE-Image sidecar**~~ — DONE (2026-04-13, v1.3). baidu/ERNIE-Image 8B DiT text-to-image (Apache 2.0) on cuda:1 at `127.0.0.1:8094`. `ernie_client.py` httpx proxy. `model="ernie-image"` on `/v1/text-to-image` and `/v2/text-to-image`. Swaps with JoyAI on cuda:1 (mutually exclusive, both coexist with ACE). ~39 GB on disk, ~33 GB VRAM (50 steps), ~18 GB turbo (8 steps). Tested: clean 1024x1024 in 11 s at 8 turbo steps. `LOAD_ERNIE=1` env var, `ERNIE_SIDECAR_URL` config, `ernie-image-sidecar.service` systemd unit.

### v1.4 Features (SHIPPED)

48. ~~**Full-fidelity history (schema v2)**~~ — DONE (2026-04-16, v1.4). Four new columns on `generations`: `params_json` (raw Pydantic request body, `storage://` URIs preserved via `_sanitize_params_for_history`), `gen_config_json` (LTX `_gen_config` snapshot at dispatch, or `{turbo_steps, turbo_guidance}` for Flux-turbo), `seed` (auto-generated if client omits), `enhanced_prompt` (`on_prompt_enhanced` callback threaded through `_encode_prompts` → all `_run_*` methods → dispatchers). Online `ALTER TABLE ADD COLUMN` gated on `PRAGMA user_version`. New `GET /v2/history/{id}` returns full parsed record.
49. ~~**Flux dashboard controls**~~ — DONE. `_flux_config` dict (`turbo_steps` default 8, `turbo_guidance` default 2.5) persisted to `.flux_config.json`. `GET/POST /v1/system/flux-config`, `POST /v1/system/flux-config/reset`. `flux_manager` `_generate`/`_img2img`/`_edit` accept `turbo_steps`+`turbo_guidance` kwargs.
50. ~~**PR 1 perf quick-wins**~~ — DONE. P1 default-negprompt embedding cache (encoder-lifecycle scoped, nulled in `evict_all`). P4 drop redundant `synchronize` after `text_encoder.encode`. P5 MP4 tmpfile to `/dev/shm` via `config.MP4_TMPDIR`. P7 `tqdm(disable=None)` TTY auto-detect. O3 `_timed("encode_prompts")` wrapper.
51. ~~**PR 2 ops resilience**~~ — DONE. O-A cancel propagation: `GenerationCancelledError` raised from `ProgressDenoiser.__call__` when `job.status == CANCELLED`; sigma loop unwinds naturally; worker_loop distinguishes cancel from fail. O-B LTX OOM recovery: `@_with_oom_recovery` on all five `_run_*` methods. O-C `SplitModelManager.reset()` for half-load recovery with `_last_load_failed` flag. O-D sidecar crash (502/503/504 on local) triggers `_auto_exit_turbo_on_sidecar_failure` via `asyncio.create_task`.
52. ~~**PR 3 cleanup audit**~~ — DONE. Dropped 2 truly-redundant `gc.collect()` in `_run_retake`. Left 8 load-bearing triples with inline comments explaining why dedup-to-`cleanup_memory()` would silently regress multi-GPU correctness (device-specific sync).

### v1.5 Features (SHIPPED)

53. ~~**Turbo-mode hardening: systemctl over HTTP /unload**~~ — DONE (2026-04-17, v1.5). `_enter_turbo_mode` no longer trusts HTTP `/unload` for cuda:1 tenants. New `_systemctl_unit(unit, action)` + `_stop_cuda1_tenants` + `_restore_cuda1_tenants` + `_wait_cuda1_free(threshold_mib=2000, timeout_s=20)` (nvidia-smi drain polling) + `_list_cuda1_processes` diagnostics. On drain timeout the entry aborts with tenant restore. Root cause: silent-success `/unload` while tensors stayed resident → CUDA OOM on subsequent ltx-sidecar load → ACE/JoyAI/ERNIE already stopped, system stuck half-transitioned.
54. ~~**LTX remote-sidecar (initial, 3-worker turbo)**~~ — DONE (2026-04-17, v1.5). `config.LTX_REMOTE_SIDECAR_URL` + `LTX_REMOTE_SIDECAR_TOKEN`. `LtxSidecarClient` gained `auth_token`+`label` kwargs. Module exposes `ltx_sidecar` (local, label="local") + `ltx_remote_sidecar` (remote, None if unconfigured). Turbo enter warms the remote and spawns a third `worker_loop` dispatching to `_dispatch_job_turbo_remote`. Transport failures on remote do NOT auto-exit turbo (remote is optional capacity).
55. ~~**Modal RTX Pro 6000 sidecar deploy**~~ — DONE (ops tree at `/mnt/nvme-1/servers/ltx-sidecar-modal/`). Custom image: torch cu130 + transformers 5.3.0 (pinned — Gemma3TextConfig attr mismatch in older) + editable `/mnt/nvme-1/repos/LTX-2`. `modal.Volume` at `/mnt/nvme-1/huggingface` pre-populated with 125 GB checkpoints. `@app.cls` + `@modal.enter` eager-loads per container boot. ~60–80 s cold, instant warm. Scales to zero after 10 min idle.

### v1.6 Features (SHIPPED)

56. ~~**Remote-sidecar pool with dashboard controls**~~ — DONE (2026-04-17, v1.6). Pool scales 0..N on demand. `config.LTX_REMOTE_SIDECAR_MAX_WORKERS` (default 4). `server.py` tracks `_remote_worker_tasks` + `_remote_worker_target` (persists across turbo toggles). `_scale_remote_pool()` reconciles live workers to target IF turbo is active (non-video jobs outside turbo would otherwise be stolen by video-only remote workers). New endpoints: `GET /v1/system/pool`, `POST /v1/system/pool/remote-workers`. Dashboard "Remote Pool" row, N+1 buttons 0..MAX, polled every 5 s. Total max concurrent video workers: **6** (2 local + 4 remote Modal).
57. ~~**Modal /unload no longer breaks the manager**~~ — DONE. Root cause: v1.5 pool scale-to-0 called `/unload` → `manager.evict_all()` cleared `self._workers`; `@modal.enter` only fires on container boot, so future warm-container `/generate` hit `No LTX workers available`. Fixes: (1) `_scale_remote_pool` scale-to-0 no longer calls `/unload` — Modal's 5-min `scaledown_window` handles it. (2) Modal `/unload` uses `worker.evict_transformer()` per worker (frees 46 GB transformer, keeps worker registry). (3) Modal `/load` self-heals if `not manager.is_ready`. (4) Modal `/generate` defensively reloads on stale container.

### v1.6.1 Features (SHIPPED)

58. ~~**Base64 media inlining for remote sidecars**~~ — DONE (2026-04-17, v1.6.1). Modal containers can't see taco-backend's `uploads/` filesystem → `FileNotFoundError` on every `a2v`/`i2v`/`retake`/keyframes when dispatched remote. Fix: `ltx_sidecar_client.LtxSidecarClient.generate()` gained `audio_b64`/`image_b64`/`video_b64` kwargs. `_dispatch_job_turbo_remote` in `server.py` reads each local media file via `Path(p).read_bytes()`, base64-encodes, and passes `*_b64` with the `*_path` field set to `None`. Keyframe images handled per-entry. Local sidecar path unchanged (direct filesystem access). Modal `/generate` materializes b64 → `tempfile.mkstemp` and cleans up in `finally`. Payload impact: 4/3 expansion; retake video up to ~135 MB. Fails fast with `ValueError("remote_dispatch: media file not found: ...")` on missing paths.

### v1.7.0 Features (SHIPPED)

59. ~~**IC-LoRA video outpaint**~~ — DONE (2026-04-17, v1.7.0). New endpoint `POST /v2/video-outpaint`. Backed by `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint` (Apache 2.0, registered as LoRA id `ic-lora-outpaint`, strategy `ic_lora_outpaint`). Server-side additions: `OutpaintPosition` Literal (9 values: center + 4 edges + 4 corners), `VideoOutpaintRequest` Pydantic model with `target_resolution`/`position`/`conditioning_strength`/`skip_stage_2`, `v2_video_outpaint` handler with default-LoRA substitution, `_dispatch_job` case for `JobType.VIDEO_OUTPAINT`. New `JobType.VIDEO_OUTPAINT` + `_MEDIA_TYPES` mapping. `_VIDEO_JOB_TYPES` includes outpaint so turbo workers handle it. `ltx_sidecar_client.py::generate()` gained `position`/`conditioning_strength`/`skip_stage_2` kwargs.
    - **Pipeline** (`split_model_manager._run_outpaint`, `@_with_oom_recovery`): 2-stage distilled, patterned on `_run_t2v` fast branch. Stage 1 at half target res with outpaint LoRA fused + `VideoConditionByReferenceLatent` appended to conditionings (wrapped in `ConditioningItemAttentionStrengthWrapper` when `conditioning_strength < 1.0`). Stage 2 upsamples 2x and refines at full target res. LoRA stays fused across both stages (accepted deviation from upstream `ICLoraPipeline` — reloading would cost ~30 s).
    - **Helpers** (module-level): `_read_lora_reference_downscale_factor` reads safetensors metadata (default 1); `_build_outpaint_reference_latent` scales source to fit, pads remainder with **-1 in normalized pixel space** (= RGB 0,0,0 after VAE decode = LoRA's training black sentinel). Temporal pad with black frames when source is short.
    - **Turbo + Modal parity**: outpaint works under turbo with local cuda:1 sidecar and via Modal pool. Modal container has LoRA pre-staged at `/mnt/nvme-1/huggingface/loras/ic-lora-outpaint.safetensors` (populated by `modal_app.py::download_weights`). `_dispatch_job_turbo_remote` rewrites local `lora_path` under `config.LORAS_DIR` to the Modal volume path. Custom (unknown) IC-LoRAs fall back to single-machine dispatch.
    - **Output**: silent MP4 (no audio). Source audio passthrough deferred to v1.7.x.
    - **Install**: `scripts/register_outpaint_lora.sh` is idempotent — `hf download` + symlink + registry.json insert.

### Tier 3 — Experimental / Higher Effort

15. **torch.compile on transformer** — available via `TORCH_COMPILE=1` but default OFF. No measurable benefit on Blackwell with cuDNN FA4. May help on Ampere/Hopper.
16. **FlashAttention3** — install flash_attn_interface for sm_100 Blackwell, auto-detected by existing code
17. **Streaming uploads** — request.stream() instead of request.body() for large files
18. **torch.compile cache** — TORCHINDUCTOR_CACHE_DIR + FX_GRAPH_CACHE for persistent compilation
21. **Retake distilled mode** — upstream supports `distilled=True` for fast retake
22. **ComfyUI last_frame_fix** — append/trim last latent frame to fix end-of-video causal conv artifacts
23. ~~**IC-LoRA pipeline**~~ — DONE for video outpaint (entry 59, v1.7.0). Other IC-LoRA uses (motion anchoring, stylized temporal coherence) remain experimental.

## Audit Reports

- [2026-04-06 ComfyUI Comparison](docs/audit-2026-04-06-comfyui-comparison.md) — 5-agent full codebase audit

### Key Files for Each Change

| Item | Primary File | Reference |
|------|-------------|-----------|
| 1, 7 | run.sh | — |
| 2, 12 | /mnt/nvme-1/repos/LTX-2/.../media_io.py:205 | encode_single_frame at :358 uses veryfast |
| 3 | flux_manager.py:116,156 | — |
| 4 | server.py v2_job_result | Use FastAPI FileResponse |
| 5 | split_model_manager.py:402,610,701,783 | Cache as module-level constant |
| 6 | lora_registry.py:60 | os.replace(tmp, target) |
| 8 | split_model_manager.py:403,611 | ti2vid_two_stages_hq.py:154 does it correctly |
| 9 | split_model_manager.py:199-204 | ltx_core/quantization/policy.py |
| 10 | split_model_manager.py | Cache in SplitModelManager.__init__ |
| 11 | split_model_manager.py:457,555,665,760,883 | evict_transformer already syncs+clears |
| 13 | server.py lifespan or config.py | — |
| 14 | split_model_manager.py:221-235 | Test av.open(BytesIO(), 'w', format='mp4') |
