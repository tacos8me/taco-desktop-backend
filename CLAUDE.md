# taco-backend

LTX-compatible inference server for noodle-i (image gen) + noodle-v (video gen).

**Version**: v1.4.1 (2026-04-16).

## Structure
- `server.py` — FastAPI app, all HTTP endpoints, job queue dispatch, history + approved-images APIs, batch scheduler, turbo mode, dashboard
- `split_model_manager.py` — Single-GPU LTX pipeline: SingleGPUModelBuilder + CachingModelFactory, CFG++ sampler (default), BatchSplitAdapter for multi-pass batching
- `flux_manager.py` — Flux 2 image generation: per-request LoRA adapter mode on cuda:0
- `ace_client.py` — ACE music generation sidecar client (httpx → ace-step on cuda:1:8001)
- `chat_manager.py` — Proxies /v1/chat/completions to llama-swap (supports per-request model override for vision ranking)
- `job_queue.py` — Async job queue: submit (202), poll, result, cancel; saves to history on completion
- `upload_store.py` — UUID file storage for uploads and job results
- `history_store.py` — SQLite-backed per-API-key generation history with thumbnails
- `lora_registry.py` — Flat-dir LoRA storage with registry.json index
- `flux_lora_registry.py` — Flux LoRA folder-drop discovery (filesystem-only, no registry.json)
- `ernie_client.py` — ERNIE-Image sidecar client (httpx → cuda:1:8094), swaps with JoyAI on cuda:1
- `ltx_sidecar_client.py` — LTX video sidecar client for DUAL_GPU_LTX mode (httpx → cuda:1:8093)
- `dashboard.html` — GPU management dashboard SPA (served at /dashboard)
- `config.py` — Paths, model mapping, device config, resolution tables, TF32 settings

## Key commands
- Run: `bash run.sh` (sets LD_LIBRARY_PATH, PYTORCH_CUDA_ALLOC_CONF, port 8090)
- Test: `uv run pytest tests/ -v`
- Health: `curl http://localhost:8090/health`

## GPU topology (v1.3 — dual-GPU layout)
- **cuda:0** → RTX PRO 6000 Blackwell 96GB — **LTX ↔ Flux** (2-tenant swap, mutually exclusive, auto-swapped on dispatch)
- **cuda:1** → RTX PRO 6000 Blackwell 96GB — **ACE xl-base+LM** (~18 GB) + **JoyAI** (~50 GB) OR **ERNIE-Image** (~33 GB), JoyAI and ERNIE swap (mutually exclusive), both coexist with ACE

Verified via `nvidia-smi -L`. No third GPU on this box — any earlier references to `cuda:2`/RTX 4000 are stale.

**DUAL_GPU_LTX mode** (v1.3): `DUAL_GPU_LTX=1` env flag dedicates both GPUs to LTX video generation. cuda:1 runs an LTX sidecar (`ltx_sidecar_client.py` → `127.0.0.1:8093`) for 2 concurrent video workers. Flux, ACE, and JoyAI are disabled. Unlike turbo mode (runtime toggle), DUAL_GPU_LTX is a boot-time flag requiring restart.

**Why mutually exclusive on cuda:0**: LTX active is ~79 GB (60 GB transformer + 19 GB encoder hub + decoder activations) and Flux active is ~81 GB (60 GB transformer + ~14 GB CPU-offload forward-pass peak via `enable_model_cpu_offload`). Combined ~160 GB > 96 GB physical. They cannot coexist on one GPU during forward pass, so the server evicts the other before running.

**Auto-swap (two tenants on cuda:0)**: `server.py` exposes `_ensure_ltx_resident()` and `_ensure_flux_ready()` helpers, called inside `_inference_lock` by `_dispatch_job()` (v2 async) and every v1 sync handler (text_to_video, image_to_video, audio_to_video, retake, text_to_image, image_to_image, image_edit) before `torch.cuda.set_device()`. LTX and Flux are mutually exclusive on cuda:0. LTX is **not** auto-reloaded after a Flux request — it stays evicted until the next video request. Long-stretch single-tenant workloads incur zero swap overhead; mixed workloads pay a per-direction-change cost (see swap section below).

### Turbo mode (v1.2)

Toggled via `POST /v1/system/turbo` (body: `{"enable": true/false}`). Temporarily claims cuda:1 for LTX, giving **2 concurrent LTX denoiser workers** (one per GPU) and processing **2 video jobs at a time**.

- **Entry**: evicts ACE, JoyAI, and Flux from both GPUs; loads dual-GPU LTX with encoder hub on cuda:0 and 2 denoiser workers; starts a second `worker_loop` pulling from `_job_queue`. Entry takes ~20 s.
- **Active**: Flux, ACE, JoyAI, and music endpoints all return `503 turbo_mode_active`. Only video generation works.
- **Exit**: cancels the second worker, evicts dual-GPU LTX, restores single-GPU LTX on cuda:0, restarts ACE+JoyAI on cuda:1. Exit takes ~15 s.
- **Batch integration**: `_batch_worker` uses `asyncio.gather` to run 2 items concurrently when turbo is active.

### ACE music sidecar (v1.2)

ACE Step (xl-base + LM) runs on cuda:1 at `127.0.0.1:8001` via a separate systemd service (`ace-step.service`). ~18 GB resident, coexists with JoyAI on cuda:1. `ace_client.py` proxies requests via httpx.

- Endpoints: `POST /v1/music` (sync), `POST /v2/music` (async job)
- Gated by `LOAD_ACE=1` env var. Returns `503` when disabled or during turbo mode.
- Music queue cap: `MAX_MUSIC_PENDING` (default 5), returns `429 music_queue_full` when exceeded.
- Phase for music jobs: `"generating"` (not `"denoising"`).

### JoyAI image-edit sidecar (v1.2, migrated from cuda:0)

Previously a third tenant on cuda:0 (v1.1.8). Now runs on cuda:1 at `127.0.0.1:8092`, coexisting with ACE. Mutually exclusive with ERNIE-Image (combined ~83 GB > 96 GB budget with ACE).

- Activation: `LOAD_JOYAI=1` env var in `.env`.
- Dispatch: `/v1/image-edit` and `/v2/image-edit` with `model="joyai-edit"` route to `joyai_client.edit()`.
- Service: `systemctl --user {start,stop,restart,status} joyai-sidecar`.
- Fallback: `503 joyai_disabled` / `503 sidecar_unreachable` — client should retry with `flux2-klein`.

### ERNIE-Image sidecar (v1.3)

baidu/ERNIE-Image (8B DiT text-to-image, Apache 2.0) runs on cuda:1 at `127.0.0.1:8094`. Swaps with JoyAI (mutually exclusive — combined ~83 GB exceeds 96 GB budget), both coexist with ACE (~18 GB).

- Activation: `LOAD_ERNIE=1` env var in `.env`.
- Dispatch: `/v1/text-to-image` and `/v2/text-to-image` with `model="ernie-image"` route to `ernie_client.generate()`.
- Service: `systemctl --user {start,stop,restart,status} ernie-image-sidecar`.
- VRAM: ~39 GB on disk, ~33 GB active (50 steps), ~18 GB turbo (8 steps).
- Resolutions: 1024x1024, 848x1264, 1264x848, etc.
- Latency: ~11 s at 8 turbo steps (1024x1024).
- Fallback: `503 ernie_disabled` / `503 sidecar_unreachable`.
- Env: `ERNIE_SIDECAR_URL` (default `http://127.0.0.1:8094`).

### Dashboard and GPU telemetry (v1.2, advanced controls v1.3)

- `GET /dashboard` — static HTML SPA for GPU management (served from `dashboard.html`)
- `GET /v1/system/gpu` — nvidia-smi telemetry (2 s cache): per-GPU memory/temp/utilization, turbo state, tenant info
- **Advanced controls** (v1.3): 14 tunable generation parameters exposed in the dashboard — sampler (Euler/CFG++), fast stage 1 steps (4-20), pro stage 1 steps (10-50), scheduler max_shift/base_shift, CFG/STG/rescale/modality scales, stage 2 steps + individual sigma sliders, eta controls, preset dropdowns, reset to default button. All persisted to `.gen_config.json` via `GET/POST /v1/system/config`.

## Flux pipeline details

### Model loading (flux_manager.py) — full bf16, no quantization
1. Build base pipeline in bf16:
   - Dev: `Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, local_files_only=True)` — weights land on CPU by default
   - Klein: `Flux2Transformer2DModel.from_single_file(...)` then `Flux2KleinKVPipeline.from_pretrained(transformer=..., torch_dtype=torch.bfloat16, local_files_only=True)`
2. **If a user LoRA is requested**: `pipe.load_lora_weights(path, adapter_name="user_lora")` — adapter mode, NO fusion. Strength is applied at inference time.
3. Device placement:
   - Dev: `pipe.enable_model_cpu_offload(device="cuda:0")` — text encoder (~45 GB), transformer (~60 GB), and VAE page CPU↔GPU on demand. Dev can NOT be fully resident in bf16 (105.9 GB > 96 GB).
   - Klein: `pipe.to("cuda:0")` — full bf16 resident (~32 GB total, fits comfortably).

**Why no FP8.** FP8 layerwise casting was removed in v1.1.1 after diagnosing screendoor / grid artifacts traced to the `fuse_lora → enable_layerwise_casting(float8_e4m3fn)` interaction. PEFT's input autocast hook forces compute back into FP8, silently defeating `compute_dtype=bfloat16`, and the shifted fused weights sit on non-standard FP8 grid points creating structured dithering (diffusers PR #10685, Flux issue #406). ComfyUI force-casts to bf16 on high-VRAM hardware for the same reason (ComfyUI issue #10087). On-disk upstream release is bf16 (`black-forest-labs/FLUX.2-dev`); we were casting bf16→fp8 at load time to save 30 GB on cuda:0, but that saving isn't worth the quality cost.

**Why `enable_model_cpu_offload` instead of `sequential_cpu_offload`.** Model-level paging moves whole components between CPU and GPU at pipeline call boundaries (text encoder for encoding, then transformer for denoising, then VAE for decoding). Sequential offload pages at the module level and is ~10x slower per step. Model-level offload adds roughly 2–5 s per request for PCIe transfer; sequential would add minutes.

**Interaction with single-GPU swap (v1.1.4).** Because Flux Dev uses `enable_model_cpu_offload`, its resident GPU footprint between requests is near-zero — the pipeline object exists in Python but weights live on pinned CPU memory. A LTX transformer can remain loaded alongside an idle Flux pipeline in Python, but **not** during a forward pass (see the mutual-exclusion math in GPU topology above). `_ensure_flux_ready()` evicts LTX only when Flux is about to run.

### Turbo mode
- `turbo: bool` field on `TextToImageRequest` / `ImageToImageRequest`
- When `turbo=true`: server overrides to 8 steps, guidance_scale 2.5, applies `FLUX_TURBO_SIGMAS` custom sigma schedule
- When `turbo=false`: uses client-provided steps/guidance and default scheduler sigmas
- Turbo and LoRA are **fully composable** — the user can attach the `flux2-turbo` folder-drop LoRA AND set `turbo: true` to get both the adapter weights and the 8-step sigma schedule. Turbo without the LoRA still works (sigma schedule only). LoRA without turbo also works (adapter with normal step count).

### Turbo sigma schedule (config.py)
```python
FLUX_TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]
```

### Klein model
- Loaded via `from_single_file()` from the HF cache (`Flux2KleinKVPipeline`)
- Full bf16 resident — no FP8, no offload
- Klein has its own 4-step distillation; the `flux2-turbo` LoRA is Dev-only (filtered client-side via `model_compat`)
- Klein ignores `guidance_scale` entirely (distilled, no CFG) — the server strips it

## Precision settings (config.py)
- `torch.backends.cuda.matmul.allow_tf32 = False` — full float32 precision for VAE decode
- `torch.backends.cudnn.allow_tf32 = False` — full float32 for VAE convolutions
- TF32 was previously enabled but degraded VAE output quality on Blackwell GPUs (VAE uses `force_upcast=True` expecting real float32)
- `bf16_reduced_precision_reduction`: left at PyTorch default (`True`). LTX-2 was trained with default bf16 accumulation — forcing float32 accumulation creates a training/inference mismatch that compounds across transformer layers × denoising steps. The old `False` setting was based on Flux VAE analysis but incorrectly applied globally.

## API contract
- **`docs/API.md` is the canonical, client-facing API spec.** Any commit that adds, removes, or changes an endpoint (URL, method, request/response shape, status codes, auth requirements) MUST update `docs/API.md` in the **same commit**. Do not split "code change" from "doc change" across commits — the doc is the contract the other-side developer reads, and drift breaks them silently.
- When touching `server.py` endpoints, re-read `docs/API.md` before writing code so you know what the external shape is supposed to be, then update both sides together.

## Conventions
- All generation runs under `@torch.inference_mode()`
- Flux output: WEBP quality 95
- LTX output: raw MP4 bytes with `Content-Type: video/mp4`
- LTX: evict transformer before VAE decode (reclaims ~22GB), don't reload after — next request handles its own state
- LTX swap: on dispatch, `_ensure_ltx_resident()` / `_ensure_flux_ready()` (server.py) auto-evicts the other tenant on cuda:0 before each request. Never assume either manager is loaded — call the helper inside `_inference_lock`.
- LTX LoRA: fusion is permanent (no unfuse), different strengths require full transformer reload. Cache key `(state_name, user_lora_tuple)`.
- Flux LoRA: adapter mode (NOT fused) — strength is applied at inference time via `pipe.set_adapters([...], [strength])`. Cache key `(model_name, lora_path)` — strength is NOT in the key, so strength changes are free. Only model or LoRA file changes trigger reload.
- Frame count must be 8k+1; resolution multiples of 64
- Port 8090, auth via `.api_keys` file (disabled when empty)
- CFG++ sampler is default for all LTX video generation (togglable via `/v1/system/config` or `/v1/system/sampler`)
- Generation config persisted to `.gen_config.json` — survives restarts, editable via dashboard or API
- VAE decode uses `TilingConfig.default()` (upstream cosine tiling from ltx-core)
- All transformer calls wrapped in `BatchSplitAdapter(max_batch_size=1)` for correct multi-pass batching
- torch.compile available via `TORCH_COMPILE=1` env flag but default OFF (no benefit on Blackwell with cuDNN FA4)

## Flux LoRA (v1.1 — folder-drop discovery, adapter mode)
- Storage: `/mnt/nvme-1/servers/taco-backend/flux_loras/` (filesystem is source of truth, no registry.json)
- ID = slugified filename stem (`MyStyle.safetensors` → `mystyle`)
- Optional sidecar `.json` next to `.safetensors` for name/description/trigger_word/model_compat
- Endpoints: `GET /v1/flux-loras` (list), `POST /v1/flux-loras/rescan` (re-scan folder)
- Request field: `lora: {id, strength}` on `TextToImageRequest` / `ImageToImageRequest` / `ImageEditRequest`
- Reuses the existing `LoRAInput` pydantic model (same `{id, strength}` shape as LTX)
- No upload endpoint by design — files managed via `cp`/`rm`

### LoRA load + strength (v1.1.1 — adapter mode, no fusion)
LoRAs are attached as **named adapters**, not fused. In `flux_manager.py::load()`:
1. Build base pipeline in bf16 (see "Model loading" above)
2. **If a user LoRA is requested**: `pipe.load_lora_weights(path, adapter_name="user_lora")` — attaches adapter layers to the bf16 transformer. NO fusion.
3. Dev: `pipe.enable_model_cpu_offload(device="cuda:0")`. Klein: `pipe.to("cuda:0")`.

At inference time, every generate method calls `_apply_lora_strength(lora_path, strength)` which runs either `pipe.set_adapters(["user_lora"], [strength])` (when a LoRA is active) or `pipe.disable_lora()` (when the request omits `lora`). This is a free O(ms) operation — no weight copy, no kernel recompile.

### FluxManager cache key
- `(self._current_model, self._current_lora_path)` — strength is NOT in the key
- `ensure_model(model_name, user_lora_path)` cache-hits iff both match
- Model change (Dev ↔ Klein) or LoRA file change → full pipeline unload + reload (~30–60 s for Dev including offload hook setup)
- **Strength change → no reload**. Only a runtime `set_adapters` call, ~0 ms
- LoRA file removed (request with no `lora` field) → no reload, just `disable_lora()` call
- Why PR #10685 doesn't apply to us: that bug is specifically about the PEFT input-autocast hook firing when the transformer is FP8-cast. Since we no longer call `enable_layerwise_casting`, the hook has nothing to fight against.

## GPU swap mode (v1.2 — 2-tenant swap on cuda:0)

LTX and Flux target `cuda:0`. `config.py` sets `LTX_DEVICE = FLUX_DEVICE = "cuda:0"`. cuda:1 runs ACE + JoyAI concurrently (no swap needed).

**Auto-swap helpers** (`server.py`):
- `_ensure_ltx_resident()` — no-op if `ltx_manager.is_ready`, else calls `ltx_manager.load_all()` (cold load is 7–30 s depending on OS page cache)
- `_ensure_flux_ready()` — no-op if `not ltx_manager.is_ready`, else calls `ltx_manager.evict_all()` (~3 s)
- Both **must** be called while holding `_inference_lock`. Wired into `_dispatch_job()` (v2 async queue) and all v1 sync handlers. The old `if not manager.is_ready: return 500` early-return guards have been removed from LTX sync handlers — auto-swap handles readiness lazily.

**evict_all leak fix** (`split_model_manager.py::evict_all`): prior to v1.1.4, `DenoiserWorker` held strong refs to source model builders that kept ~22 GB of encoder hub pinned after eviction. Fix: explicitly null reference paths before dropping workers. Verified: cuda:0 drops from 66.9 GB → **683 MiB** after unload. (v1.3 refactored from `ModelLedger` → `SingleGPUModelBuilder` / `CachingModelFactory` but the eviction pattern is the same.)

**Swap + system endpoints** (Bearer auth required):
- `POST /v1/ltx/unload`, `POST /v1/ltx/reload`
- `POST /v1/flux/unload`, `POST /v1/flux/reload`
- `POST /v1/system/pause`, `POST /v1/system/resume` (acquire `_inference_lock`)
- `POST /v1/system/turbo` — toggle turbo mode (dual-GPU LTX, see GPU topology section)
- `GET /v1/system/sampler`, `POST /v1/system/sampler` — get/toggle CFG++ vs Euler sampler (v1.3, alias for gen_config subset)
- `GET /v1/system/config`, `POST /v1/system/config` — get/update all generation parameters (v1.3)
- `POST /v1/system/config/reset` — reset generation config to defaults (v1.3)
- `GET /v1/system/gpu` — nvidia-smi telemetry
- `GET /dashboard` — GPU management dashboard with advanced generation controls

**Latency**:
- Within-type (video→video, image→image with same LoRA): unchanged, fast
- LTX→Flux (image request after video): +3 s eviction + normal Flux forward pass
- Flux→LTX (video request after image): +7–30 s cold LTX load + normal video generation
- Turbo mode entry: ~20 s (evict ACE+JoyAI+Flux, load dual-GPU LTX)
- Turbo mode exit: ~15 s (evict dual-GPU LTX, restore single-GPU, restart ACE+JoyAI)
- LoRA strength changes: still free runtime op, unchanged by the swap refactor

## Batch scheduler (v1.2, updated v1.3)

- `POST /v2/batch` — submit a batch of generation jobs (1-50 items per batch)
- `GET /v2/batch/{batch_id}` — poll batch status + partial results
- `GET /v2/batch/{batch_id}/result/{index}` — download result file for a completed batch item (v1.3)
- `DELETE /v2/batch/{batch_id}` — cancel remaining items
- Items are sorted images-first (Klein before Dev) to minimize GPU swaps
- In turbo mode, `batch_worker` uses `asyncio.gather` to process 2 items concurrently
- Max batch queue depth: `MAX_BATCH_QUEUE_DEPTH` (default 5)
- Max items per batch: `MAX_BATCH_ITEMS` (default 50)
- Supported item types: `text-to-image`, `image-to-image`, `image-edit`, `text-to-video`, `image-to-video`

## Keyframe symbolic indices (v1.1)
- `KeyframeInput.frame_index` accepts `int | "first" | "middle" | "last"`
- Negative integers supported: -1 = last frame, -12 = 12 frames before end
- Symbolic values resolved in `_resolve_keyframes(body, num_frames)` after num_frames computed
- "first"=0, "middle"=num_frames//2, "last"=num_frames-1
- Duplicate detection on resolved integer values (e.g., "first" and 0 conflict → 422)
- Bounds check: frame_index >= num_frames → 422
- Recommended strengths: first=1.0, middle=0.5, last=1.0

## Char mode — character consistency ranking
- `POST /v2/char/rank` — takes a `rank_image_uri` + `generated_image_uri` + `prompt`, sends both images to Gemma 4 31B (unsloth/gemma-4-31B-it-GGUF on llama-swap) as a multimodal OpenAI-style message
- System prompt is `CHAR_RANKING_PROMPT` in server.py — asks for strict JSON with face_match/eyes/proportions/overall_likeness (1-10) and structured `edits: {add, remove, modify}`
- `chat_manager.generate_chat_completion(..., model=config.CHAR_VISION_MODEL)` routes the request to the specific llama-swap model; default `CHAT_MODEL` is `gemma-3-12b-nvfp4` (used for other chat endpoints)
- The noodle-i Char tab runs a client-side loop: generate (klein multi-ref edit) → rank → apply edits → regenerate until score ≥ 9 or user hits Stop
- Score ≥ 9 cutoff and structured-edit format are both prompt-engineered, not hardcoded in the server

## Generation history (history_store.py)
- SQLite DB at `/mnt/nvme-1/servers/taco-backend/history.db` — **WAL mode** (v1.1.6), readers never block behind the single writer
- Saves every completed v2 job with prompt, model, dimensions, result_uri, thumbnail
- API key hashed with SHA-256 (raw keys never stored)
- Thumbnails: 256px-wide JPEG at `/mnt/nvme-1/servers/taco-backend/thumbnails/`. Video thumbnails extract the first frame via PyAV (v1.1.5).
- Endpoints: `GET /v2/history`, `GET /v2/history/{id}`, `GET /v2/history/{id}/image`, `GET /v2/history/{id}/thumbnail`
- Cleanup: job_queue keeps completed result files (history manages lifecycle), 30-day retention
- Both noodle-i and noodle-v consume the same history API
- **`history.save()` runs in an `asyncio.to_thread` task** (v1.1.6) fire-and-forgotten from `worker_loop`, so the queue worker dequeues the next job immediately instead of stalling ~300 ms per job on PyAV + SQLite
- **Schema v2 (2026-04-16)**: four new columns — `params_json`, `gen_config_json`, `seed`, `enhanced_prompt` — for full reproducibility. Online migration via `PRAGMA user_version`, idempotent, no backfill.
- **`params_json`**: raw request body (Pydantic `body.model_dump(mode="json")`) — preserves `storage://` URIs, resolution enum, LoRA `id+strength`, keyframes symbolic indices. Music sanitizes paths back to URIs via `_sanitize_params_for_history`.
- **`gen_config_json`**: LTX `_gen_config` snapshot at dispatch time OR `{turbo_steps, turbo_guidance}` for Flux-turbo requests. NULL for non-turbo Flux, ERNIE, JoyAI.
- **`enhanced_prompt`**: LTX-rewritten prompt text when `enhance_prompt=true` (captured via `on_prompt_enhanced` callback). Always NULL for Flux/ERNIE/JoyAI/retake.

## v2 job observability (v1.1.6 / v1.1.7)

- **`Job.phase` field** — coarse post-denoise phase: `"denoising" | "decoding" | "encoding" | "saving" | None`. Denoising callbacks cap at **0.90** (was 0.99 in v1.1.4); the top 10% of progress is reserved for post-denoise phases emitted explicitly by `split_model_manager._run_*` and `flux_manager._generate/_img2img/_edit`.
- **`/v2/jobs/{id}` status response** exposes `phase` when processing.
- **`/v2/jobs/{id}/stream` SSE endpoint** (v1.1.7) — EventSource-compatible live status stream. Emits on `(status, progress, phase, error_code)` change, closes on terminal state, keepalive comment every 15 s. Accepts bearer header OR `?token=` query param (browsers). Replaces the 240-GET polling loop per video job with one long-lived connection.
- **`/v2/jobs/{id}/preview`** (v1.1.6) serves the on-disk thumbnail written by `history.save()` via zero-copy `FileResponse`. Fallback lazy extraction still exists for jobs without api_key but is offloaded via `asyncio.to_thread`.
- **`GET /v2/history/{id}`** — full record with parsed params + gen_config (v1.3).
- **Timing logs** at every post-denoise phase boundary in `split_model_manager` (`vae_decode`, `video_decode+encode`) and `flux_manager` (`flux_webp_encode`), plus `history.save` in `job_queue`. Grep production logs for real wall-clock per phase: `journalctl -u taco-backend | grep -E "vae_decode|encode|history.save"`.

## Approved images (noodle-i → noodle-v pipeline)
- Manifest: `/mnt/nvme-1/servers/taco-backend/approved-images/manifest.json`
- Images stored in shared uploads dir (referenced by `storage://{uuid}` URIs)
- Endpoints: `POST /v1/approved-images`, `GET /v1/approved-images`, `GET /v1/approved-images/{id}/file`
- Per-API-key scoped (key hash in manifest entries)
- noodle-i "To Video" button uploads image then POSTs metadata
- noodle-v polls the GET endpoint to display approved feed

## Generation config (v1.3)

All LTX generation parameters are stored in `.gen_config.json` (project root) and survive restarts. Managed via `GET/POST /v1/system/config` and the dashboard advanced controls.

### Parameters (14 tunable via dashboard)
- `sampler`: `"cfg_pp"` (default) or `"euler"` — CFG++ uses alpha=(1-sigma) rescaling for improved motion quality
- `fast_stage1_steps`: 8 (default), range 4-20
- `pro_stage1_steps`: 30 (default), range 10-50
- `scheduler_max_shift`: 2.05, `scheduler_base_shift`: 0.95
- `cfg_scale`: 3.0, `stg_scale`: 1.0, `rescale_scale`: 0.7, `modality_scale`: 3.0
- `stg_blocks`: [28]
- `stage2_sigmas`: [0.85, 0.725, 0.4219, 0.0] — individual sigma sliders in dashboard
- `eta_stage1`: 1.0 (ancestral noise for distilled stage 1), `eta_default`: 0.0 (deterministic for guided/stage 2)

### Endpoints
- `GET /v1/system/config` — returns full config dict
- `POST /v1/system/config` — merge-update (partial body OK, unknown keys ignored)
- `POST /v1/system/config/reset` — restore all defaults
- `GET/POST /v1/system/sampler` — alias for the sampler/eta/stage2_sigmas subset

No server restart required — changes take effect on the next generation request. Dashboard preset dropdowns and reset button use these endpoints.

## Critical patterns
- `cleanup_memory()` calls gc.collect + empty_cache + synchronize — avoid redundant calls after evict_transformer (which already syncs+clears)
- `detect_params(checkpoint)` opens safetensors metadata — cache the result, don't call per-request
- `encode_prompts()` with CachingModelFactory keeps text encoder loaded — the internal `del` only drops local ref
- Retake uses `MultiModalGuider(...)` directly, NOT `create_multimodal_guider_factory().build()` — factory has no `.build()` method
- Audio latent must be trimmed/padded to `AudioLatentShape.from_video_pixel_shape(output_shape).frames`
- A2V uses `GuidedDenoiser` (static) for stage 1, frozen audio with `noise_scale=0.0`

## Text encoder variants
- `GEMMA_VARIANT=default` — Google Gemma 3 12B PT (standard, BF16)
- `GEMMA_VARIANT=sikaworld` — Sikaworld abliterated FP4 (uncensored, NVFP4 quantized)
- Set via `.env` or environment variable, requires server restart
- Sikaworld path: `/mnt/nvme-1/huggingface/gemma-3-12b-sikaworld/`

## Dependencies
- **PyTorch 2.11.0+cu130** — FlexAttention/FA4 on Blackwell sm_120, SDPA auto-selects cuDNN FlashAttention
- **diffusers 0.38.0.dev0** (git main) — required for Flux2KleinKVPipeline (not in any stable release)
- **ltx-core 1.1.1** (editable install from `/mnt/nvme-1/repos/LTX-2/packages/ltx-core`) — upstream sync with vocoder fp32 fix, cosine tiling, layer streaming, BatchSplitAdapter
- **ltx-pipelines 1.1.1** (editable install from `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines`) — new Denoiser classes (SimpleDenoiser, GuidedDenoiser, FactoryGuidedDenoiser), updated sampler signatures
- LTX-2 repo: `/mnt/nvme-1/repos/LTX-2` (editable install, `torch~=2.7` pin is PEP 440 compatible with 2.11)
- Checkpoints: `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/` — v1.1 distilled models (`ltx-2.3-22b-distilled-1.1.safetensors`, `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`, `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`)
- cuDNN >=9.20 (fixes conv3d memory bug) — currently 9.20.0.48
- cuBLAS >=13.2 (BF16/FP8 Blackwell speedup) — currently 13.3.0.5
- nvidia packages revert on `uv sync` — use `--no-sync` for runtime, manual pip for upgrades
- peft (required for LoRA loading via diffusers)
- comfy-kitchen (required for NVFP4 dequantization of Sikaworld text encoder)
