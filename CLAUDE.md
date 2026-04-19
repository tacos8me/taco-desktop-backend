# taco-backend

LTX-compatible inference server for noodle-i (image gen) + noodle-v (video gen).

**Version**: v1.9.2 (2026-04-19).

## Quick lookup

| Topic | Go to |
|---|---|
| Adding a new endpoint | [Structure](#structure) + [API contract](#api-contract) + [Conventions](#conventions) |
| Changing GPU behavior (swap, turbo, pool) | [GPU topology](#gpu-topology) + [Turbo mode](#turbo-mode-v12-hardened-v15) + [Remote-sidecar pool](#remote-sidecar-pool-v16) |
| Touching LTX generation (denoising, stages, sampler) | [Conventions](#conventions) + [Critical patterns](#critical-patterns) + `split_model_manager.py` |
| Touching Flux generation | [Flux pipeline details](#flux-pipeline-details) + `flux_manager.py` |
| LoRA plumbing | [LTX LoRA](#conventions) / [Flux LoRA](#flux-lora-v11--folder-drop-discovery-adapter-mode) / [IC-LoRA outpaint](#ic-lora-video-outpaint-v170) |
| Async job queue, phases, SSE | [v2 job observability](#v2-job-observability-v116--v117) + `job_queue.py` |
| History + reproducibility | [Generation history](#generation-history-history_storepy) |
| Sidecars (ACE / JoyAI / ERNIE / LTX-remote) | [ACE](#ace-music-sidecar-v12) / [JoyAI](#joyai-image-edit-sidecar-v12-migrated-from-cuda0) / [ERNIE](#ernie-image-sidecar-v13) / [Remote pool](#remote-sidecar-pool-v16) |
| Video outpaint | [IC-LoRA video outpaint](#ic-lora-video-outpaint-v170) |
| Dashboard + live tuning | [Dashboard](#dashboard-and-gpu-telemetry-v12-advanced-controls-v13) + [Generation config](#generation-config-v13) |
| Client-facing API shape | `docs/API.md` (canonical) |
| Shipped-feature archaeology | `CHANGELOG.md` + `AGENTS.md` (per-version deltas) |

## Structure
- `server.py` — FastAPI app, all HTTP endpoints, job queue dispatch, history + approved-images APIs, batch scheduler, turbo mode + remote pool, dashboard. ~3.9 k lines; key anchors: `_enter_turbo_mode` (~:1712), `_exit_turbo_mode` (~:1795), `_scale_remote_pool` (~:1604), `_dispatch_job` (~:320), `_dispatch_job_turbo` / `_dispatch_job_turbo_remote` (~:1520/1666), `v2_video_outpaint` (~:2685)
- `split_model_manager.py` — Single-GPU LTX pipeline: SingleGPUModelBuilder + CachingModelFactory, CFG++ sampler (default), BatchSplitAdapter for multi-pass batching. Houses all `_run_*` methods (t2v / i2v / a2v / retake / outpaint / HQ). ~2.2 k lines.
- `flux_manager.py` — Flux 2 image generation: per-request LoRA adapter mode on cuda:0, bf16, `enable_model_cpu_offload` on Dev
- `ace_client.py` — ACE music generation sidecar client (httpx → ace-step on cuda:1:8001)
- `chat_manager.py` — Proxies /v1/chat/completions to llama-swap (supports per-request model override for vision ranking)
- `ernie_client.py` — ERNIE-Image sidecar client (httpx → cuda:1:8094), swaps with JoyAI on cuda:1
- `joyai_client.py` — JoyAI-Image-Edit sidecar client (httpx → cuda:1:8092)
- `ltx_sidecar_client.py` — LTX video sidecar client. One primary `ltx_sidecar` (local cuda:1:8093) + optional `ltx_remote_sidecar` (Modal, configured via `LTX_REMOTE_SIDECAR_URL`). `generate()` supports base64 media inlining (`audio_b64` / `image_b64` / `video_b64`) for remote sidecars that can't see the local `uploads/` filesystem, plus outpaint extras (`position`, `conditioning_strength`, `skip_stage_2`).
- `job_queue.py` — Async job queue: submit (202), poll, result, cancel; saves to history on completion. `JobType` enum includes `VIDEO_OUTPAINT` (v1.7.0).
- `upload_store.py` — UUID file storage for uploads and job results
- `history_store.py` — SQLite-backed per-API-key generation history with thumbnails; schema v2 (params_json, gen_config_json, seed, enhanced_prompt)
- `lora_registry.py` — Flat-dir LTX LoRA storage with registry.json index. IC-LoRA outpaint LoRA lives here with `strategy="ic_lora_outpaint"`.
- `flux_lora_registry.py` — Flux LoRA folder-drop discovery (filesystem-only, no registry.json)
- `composition_store.py` / `export_handler.py` — Composition export (video concat / transcode)
- `nvfp4_loader.py` — NVFP4→BF16 dequantizer for Sikaworld Gemma variant
- `dashboard.html` — GPU management dashboard SPA (served at /dashboard). Advanced LTX controls, Flux config, **Remote Pool** button grid (0..MAX), turbo toggle, GPU telemetry.
- `config.py` — Paths, model mapping, device config, resolution tables, TF32 settings, env-var sidecar toggles (`LOAD_FLUX`, `LOAD_ACE`, `LOAD_JOYAI`, `LOAD_ERNIE`, `LTX_REMOTE_SIDECAR_URL`, `LTX_REMOTE_SIDECAR_MAX_WORKERS`, etc.)
- `scripts/register_outpaint_lora.sh` — idempotent cold-start: `hf download` + symlink + registry.json insert for the IC-LoRA outpaint LoRA (id `ic-lora-outpaint`).
- `run.sh` — entrypoint: sets `LD_LIBRARY_PATH`, `PYTORCH_CUDA_ALLOC_CONF`, env flags, then `uv run python server.py` (port 8090)

## Key commands
- Run: `bash run.sh`
- Test: `uv run pytest tests/ -v`
- Health: `curl http://localhost:8090/health`
- Register outpaint LoRA: `bash scripts/register_outpaint_lora.sh`

## GPU topology
- **cuda:0** → RTX PRO 6000 Blackwell 96GB — **LTX ↔ Flux** (2-tenant swap, mutually exclusive, auto-swapped on dispatch)
- **cuda:1** → RTX PRO 6000 Blackwell 96GB — **ACE xl-base+LM** (~18 GB) + **JoyAI** (~50 GB) OR **ERNIE-Image** (~33 GB), JoyAI and ERNIE swap (mutually exclusive), both coexist with ACE

Verified via `nvidia-smi -L`. No third GPU on this box — any earlier references to `cuda:2`/RTX 4000 are stale.

**DUAL_GPU_LTX mode**: `DUAL_GPU_LTX=1` env flag dedicates both GPUs to LTX video generation. cuda:1 runs an LTX sidecar (`ltx_sidecar_client.py` → `127.0.0.1:8093`) for 2 concurrent video workers. Flux, ACE, and JoyAI are disabled. Unlike turbo mode (runtime toggle), DUAL_GPU_LTX is a boot-time flag requiring restart.

**Why mutually exclusive on cuda:0**: LTX active is ~79 GB (60 GB transformer + 19 GB encoder hub + decoder activations) and Flux active is ~81 GB (60 GB transformer + ~14 GB CPU-offload forward-pass peak via `enable_model_cpu_offload`). Combined ~160 GB > 96 GB physical. They cannot coexist on one GPU during forward pass, so the server evicts the other before running.

**Auto-swap (two tenants on cuda:0)**: `server.py` exposes `_ensure_ltx_resident()` and `_ensure_flux_ready()` helpers, called inside `_inference_lock` by `_dispatch_job()` (v2 async) and every v1 sync handler (text_to_video, image_to_video, audio_to_video, retake, video_outpaint, text_to_image, image_to_image, image_edit) before `torch.cuda.set_device()`. LTX and Flux are mutually exclusive on cuda:0. LTX is **not** auto-reloaded after a Flux request — it stays evicted until the next video request. Long-stretch single-tenant workloads incur zero swap overhead; mixed workloads pay a per-direction-change cost (see swap section below).

### Turbo mode (v1.2, hardened v1.5)

Toggled via `POST /v1/system/turbo` (body: `{"enable": true/false}`). Temporarily claims cuda:1 for LTX, giving **2 concurrent LTX denoiser workers** (one per GPU). With the remote-sidecar pool added (v1.6) turbo supports **up to 6 concurrent video workers total** (2 local + up to 4 Modal).

- **Entry** (`_enter_turbo_mode`, server.py:~1712): Flux unloaded → `_stop_cuda1_tenants()` runs `systemctl --user stop` on `ace-step`, `joyai-sidecar`, `ernie-image-sidecar`, `ltx-sidecar` → `_wait_cuda1_free(threshold_mib=2000, timeout_s=20)` drains cuda:1 via nvidia-smi polling → on timeout, aborts with `_restore_cuda1_tenants()` rollback → `systemctl start ltx-sidecar` → poll /health → /load → spawn second `worker_loop` → `_scale_remote_pool()` up to `_remote_worker_target`. Entry takes ~20 s.
- **Active**: Flux, ACE, JoyAI, ERNIE, music endpoints all return `503 turbo_mode_active`. Only video generation works.
- **Exit** (`_exit_turbo_mode`, server.py:~1795): scales remote pool to 0 → HTTP /unload local sidecar → `systemctl stop ltx-sidecar` → `_restore_cuda1_tenants()` re-starts configured `LOAD_*=1` services. Exit takes ~15 s.
- **Batch integration**: `_batch_worker` uses `asyncio.gather` to run items concurrently (2 under turbo).
- **Why systemctl-stop, not HTTP /unload**: v1.4 trusted HTTP `/unload` to free cuda:1. Silent unloads that succeeded on the wire while tensors stayed resident caused `CUDA OOM` on the subsequent ltx-sidecar `load`. `systemctl stop` + nvidia-smi drain verification is the hammer.

### Remote-sidecar pool (v1.6 → v1.9.0 multi-provider)

Optional remote workers augment turbo's local 2. v1.9.0: **multi-provider** — Modal and RunPod run side-by-side, each with independent target/active/max counts. Dict-keyed structure in `ltx_sidecar_client.py`: `ltx_remote_sidecars: dict[str, LtxSidecarClient]`.

- **Env (v1.9.0)**: `LTX_MODAL_SIDECAR_URL/TOKEN/MAX_WORKERS` and `LTX_RUNPOD_SIDECAR_URL/TOKEN/MAX_WORKERS`. Legacy `LTX_REMOTE_SIDECAR_*` (singular) still honored and aliased to the modal provider for backwards compat.
- **Client module** (`ltx_sidecar_client.py`): `ltx_remote_sidecars` dict keyed by provider name. Module-level `ltx_remote_sidecar` (singular) is preserved as an alias pointing at the modal entry so v1.6-v1.8 code keeps working.
- **Server state** (`server.py`): `_remote_worker_targets: dict[str, int]` (per-provider targets, persist across turbo toggles) + `_remote_worker_tasks: dict[str, list[asyncio.Task]]`. `_PROVIDERS = ("modal", "runpod")`.
- **Endpoints**:
  - `GET /v1/system/pool` → `{turbo_active, providers: {modal: {...}, runpod: {...}}, remote_*: <legacy aliases>}`.
  - `POST /v1/system/pool/remote-workers` accepts either `{"count": N}` (legacy — scales modal only) or `{"modal": N, "runpod": M}` (per-provider).
  - `POST /v1/system/pool/remote-workers/{provider}` with `{"count": N}` — RESTful per-provider.
- **Dispatch**: `_dispatch_job_turbo_remote(job, *, provider: str)` takes the provider kwarg. `_scale_remote_pool()` uses `functools.partial` to bind each worker task to its provider at spawn time. The LoRA-path rewrite for outpaint uses `config.LTX_PROVIDER_LORAS_MOUNT[provider]` → Modal `/mnt/nvme-1/huggingface/loras/`, RunPod `/runpod-volume/loras/`.
- **Media base64**: same for both providers — `_read_b64()` reads local `audio_path`/`image_path`/`video_path`/keyframe images and ships as `*_b64` kwargs. Neither Modal nor RunPod can see the host's `uploads/`.
- **Transport failure**: neither provider's transport errors auto-exit turbo — remotes are optional extra capacity. Local sidecar failure still triggers `_auto_exit_turbo_on_sidecar_failure`.
- **Dashboard**: two rows — `poolBtnGrid` (Modal) + `poolBtnGridRunpod` (RunPod). Each row renders N+1 buttons 0..MAX, polled every 5 s via `GET /v1/system/pool`. JS walks `data.providers.{modal,runpod}` with fallback to legacy flat fields for safety.
- **RunPod sidecar**: new repo at `/mnt/nvme-1/servers/ltx-sidecar-runpod/`. Load-Balancing Serverless on RTX PRO 6000 Blackwell, FastAPI inside each worker, `/ping` health probe. Dockerfile mirrors Modal's image recipe.

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
- Latency: ~11 s at 8 turbo steps (1024x1024).
- Fallback: `503 ernie_disabled` / `503 sidecar_unreachable`.
- Env: `ERNIE_SIDECAR_URL` (default `http://127.0.0.1:8094`).

### Dashboard and GPU telemetry (v1.2, advanced controls v1.3)

- `GET /dashboard` — static HTML SPA for GPU management (served from `dashboard.html`)
- `GET /v1/system/gpu` — nvidia-smi telemetry (2 s cache): per-GPU memory/temp/utilization, turbo state, tenant info
- **Advanced controls**: 14 tunable LTX generation parameters (sampler, fast/pro stage 1 steps, scheduler max/base shift, CFG/STG/rescale/modality scales, stage 2 steps + individual sigma sliders, eta controls, preset dropdowns, reset). All persisted to `.gen_config.json` via `GET/POST /v1/system/config`.
- **Flux config** (v1.4): 2 Flux-turbo tunables (`turbo_steps`, `turbo_guidance`) persisted to `.flux_config.json` via `GET/POST /v1/system/flux-config`.
- **Remote Pool** (v1.6): N+1 button grid 0..`LTX_REMOTE_SIDECAR_MAX_WORKERS`, active highlighted, status line reflects configured / target / active / turbo-pending.

## IC-LoRA video outpaint (v1.7.0)

New async endpoint `POST /v2/video-outpaint`. Expands a source video's canvas to a larger target resolution by letterboxing with pure-black padding, then uses `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint` (registered as LoRA id `ic-lora-outpaint`, strategy `ic_lora_outpaint`) to fill the black regions with temporally coherent content.

- **Request** (`VideoOutpaintRequest`, server.py:~706): `video_uri`, `prompt`, `target_resolution`, `position` (`OutpaintPosition` 9-value Literal: `center` + 4 edges + 4 corners), `duration`, `fps`, `seed`, `enhance_prompt`, `lora` (optional; defaults to `id="ic-lora-outpaint"` if omitted), `conditioning_strength` ∈ [0, 1], `skip_stage_2` escape hatch.
- **Handler** (server.py:~2685): if `lora=None`, substitutes `LoRAInput(id="ic-lora-outpaint", strength=1.0)`. Submits `JobType.VIDEO_OUTPAINT` through the standard `_submit_job` path.
- **Pipeline** (`_run_outpaint` in split_model_manager.py:~1825, `@_with_oom_recovery`): 2-stage distilled, patterned on `_run_t2v` fast branch:
  1. Stage 1 at half target res with the outpaint LoRA fused into the distilled transformer. Letterboxed source is VAE-encoded via `_build_outpaint_reference_latent` and appended to `stage_1_cond` as `VideoConditionByReferenceLatent(latent=ref_latent, downscale_factor=ref_scale, strength=user_lora[1])`. If `conditioning_strength < 1.0`, wrapped in `ConditioningItemAttentionStrengthWrapper(..., attention_mask=conditioning_strength)`.
  2. Stage 2 (if not skipped) upsamples 2x and refines at full target res. **LoRA stays fused across both stages** (accepted deviation from upstream `ltx_pipelines.ic_lora.ICLoraPipeline`, which drops LoRA for stage 2; reloading mid-request would cost ~30 s of fusion work).
- **Helpers** (split_model_manager.py module-level, ~:180): `_read_lora_reference_downscale_factor(lora_path)` reads `reference_downscale_factor` from safetensors metadata (default 1); `_build_outpaint_reference_latent(...)` scales source proportionally to fit, pads remainder with **-1 in normalized pixel space** (= RGB 0,0,0 after VAE decode = the LoRA's training black sentinel). Temporal dim padded with black frames if source is shorter than `num_frames`.
- **Output**: silent MP4 (no audio). Source audio passthrough deferred to v1.7.x.
- **LoRA cache key**: same as every other LTX flow — `(state_name="distilled", user_lora_tuple)`. Fusion is permanent; strength changes require full transformer reload.
- **Turbo parity**: `JobType.VIDEO_OUTPAINT` is in `_VIDEO_JOB_TYPES` (server.py:111), so turbo workers handle it. Local cuda:1 sidecar and Modal both work. `ltx_sidecar_client.py::generate()` carries `position` / `conditioning_strength` / `skip_stage_2` kwargs.
- **Modal staging**: `scripts/register_outpaint_lora.sh` installs the LoRA locally; `modal_app.py::download_weights` stages the same LoRA into the Modal HF volume at `/mnt/nvme-1/huggingface/loras/`.

## Flux pipeline details

### Model loading (flux_manager.py) — full bf16, no quantization
1. Build base pipeline in bf16:
   - Dev: `Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, local_files_only=True)` — weights land on CPU by default
   - Klein: `Flux2Transformer2DModel.from_single_file(...)` then `Flux2KleinKVPipeline.from_pretrained(transformer=..., torch_dtype=torch.bfloat16, local_files_only=True)`
2. **If a user LoRA is requested**: `pipe.load_lora_weights(path, adapter_name="user_lora")` — adapter mode, NO fusion. Strength is applied at inference time.
3. Device placement:
   - Dev: `pipe.enable_model_cpu_offload(device="cuda:0")` — text encoder (~45 GB), transformer (~60 GB), and VAE page CPU↔GPU on demand. Dev can NOT be fully resident in bf16 (105.9 GB > 96 GB).
   - Klein: `pipe.to("cuda:0")` — full bf16 resident (~32 GB total, fits comfortably).

**Why no FP8.** FP8 layerwise casting was removed in v1.1.1 after diagnosing screendoor / grid artifacts traced to the `fuse_lora → enable_layerwise_casting(float8_e4m3fn)` interaction. PEFT's input autocast hook forces compute back into FP8, silently defeating `compute_dtype=bfloat16`, and the shifted fused weights sit on non-standard FP8 grid points creating structured dithering (diffusers PR #10685, Flux issue #406). ComfyUI force-casts to bf16 on high-VRAM hardware for the same reason (ComfyUI issue #10087).

**Why `enable_model_cpu_offload` instead of `sequential_cpu_offload`.** Model-level paging moves whole components between CPU and GPU at pipeline call boundaries. Sequential offload pages at the module level and is ~10x slower per step.

**Interaction with single-GPU swap.** Because Flux Dev uses `enable_model_cpu_offload`, its resident GPU footprint between requests is near-zero — the pipeline object exists in Python but weights live on pinned CPU memory. A LTX transformer can remain loaded alongside an idle Flux pipeline in Python, but **not** during a forward pass. `_ensure_flux_ready()` evicts LTX only when Flux is about to run.

### Turbo (Flux image)
- `turbo: bool` field on `TextToImageRequest` / `ImageToImageRequest`
- When `turbo=true`: server overrides to `_flux_config["turbo_steps"]` (default 8), `turbo_guidance` (default 2.5), applies `FLUX_TURBO_SIGMAS` custom sigma schedule
- When `turbo=false`: uses client-provided steps/guidance and default scheduler sigmas
- Turbo and LoRA are **fully composable** — attach the `flux2-turbo` folder-drop LoRA AND set `turbo: true` for both adapter weights and the 8-step sigma schedule.
- `_flux_config` (v1.4) persisted to `.flux_config.json`; tunable via dashboard or `POST /v1/system/flux-config`.

### Klein model
- Loaded via `from_single_file()` (`Flux2KleinKVPipeline`), full bf16 resident
- Klein has its own 4-step distillation; `flux2-turbo` LoRA is Dev-only (client-side `model_compat` filter)
- Klein ignores `guidance_scale` entirely (distilled, no CFG) — the server strips it

## Precision settings (config.py)
- `torch.backends.cuda.matmul.allow_tf32 = False` — full float32 precision for VAE decode
- `torch.backends.cudnn.allow_tf32 = False` — full float32 for VAE convolutions
- TF32 was previously enabled but degraded VAE output quality on Blackwell GPUs (VAE uses `force_upcast=True` expecting real float32)
- `bf16_reduced_precision_reduction`: left at PyTorch default (`True`). LTX-2 was trained with default bf16 accumulation — forcing float32 accumulation creates a training/inference mismatch that compounds across transformer layers × denoising steps.

## API contract
- **`docs/API.md` is the canonical, client-facing, LLM-optimized API spec.** Any commit that adds, removes, or changes an endpoint (URL, method, request/response shape, status codes, auth requirements) MUST update `docs/API.md` in the **same commit**. Do not split "code change" from "doc change" across commits — the doc is the contract the other-side developer reads, and drift breaks them silently.
- When touching `server.py` endpoints, re-read `docs/API.md` before writing code so you know what the external shape is supposed to be, then update both sides together.
- CLAUDE.md (this file) and AGENTS.md describe **how to work in the code**. API.md describes **how clients talk to the service**. Don't duplicate substantive detail — link.

## Conventions
- All generation runs under `@torch.inference_mode()`
- All LTX `_run_*` methods are decorated with `@_with_oom_recovery` — on CUDA OOM the wrapper evicts the transformer + `cleanup_memory()` then re-raises (mirrors Flux pattern)
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
- Cancellation: `DELETE /v2/jobs/{id}` raises `GenerationCancelledError` from `ProgressDenoiser.__call__` when `job.status == CANCELLED`, unwinding the sigma loop naturally (v1.4)
- MP4 encode tmpfile lives on `/dev/shm` via `config.MP4_TMPDIR` (tmpfs; fallback `/tmp`)
- Default negative prompt embeddings are cached per encoder lifecycle (`DEFAULT_NEGATIVE_PROMPT`, nulled in `evict_all`)
- torch.compile available via `TORCH_COMPILE=1` env flag but default OFF (no benefit on Blackwell with cuDNN FA4)

## Flux LoRA (v1.1 — folder-drop discovery, adapter mode)
- Storage: `/mnt/nvme-1/servers/taco-backend/flux_loras/` (filesystem is source of truth, no registry.json)
- ID = slugified filename stem (`MyStyle.safetensors` → `mystyle`)
- Optional sidecar `.json` next to `.safetensors` for name/description/trigger_word/model_compat
- Endpoints: `GET /v1/flux-loras` (list), `POST /v1/flux-loras/rescan` (re-scan folder)
- Request field: `lora: {id, strength}` on `TextToImageRequest` / `ImageToImageRequest` / `ImageEditRequest`
- Reuses the existing `LoRAInput` pydantic model (same `{id, strength}` shape as LTX)
- No upload endpoint by design — files managed via `cp`/`rm`

LoRAs attach as **named adapters**, not fused. At inference time every generate method calls `_apply_lora_strength(lora_path, strength)` → either `pipe.set_adapters(["user_lora"], [strength])` or `pipe.disable_lora()`. Free O(ms) op.

**FluxManager cache key** = `(model_name, lora_path)`. Model change (Dev ↔ Klein) or LoRA file change → full reload (~30–60 s for Dev). Strength change → zero reload.

## GPU swap mode (2-tenant on cuda:0)

LTX and Flux target `cuda:0`. `config.py` sets `LTX_DEVICE = FLUX_DEVICE = "cuda:0"`. cuda:1 runs ACE + JoyAI (or ERNIE) concurrently (no swap needed).

**Auto-swap helpers** (`server.py`):
- `_ensure_ltx_resident()` — no-op if `ltx_manager.is_ready`, else calls `ltx_manager.load_all()` (cold load is 7–30 s depending on OS page cache)
- `_ensure_flux_ready()` — no-op if `not ltx_manager.is_ready`, else calls `ltx_manager.evict_all()` (~3 s)
- Both **must** be called while holding `_inference_lock`. Wired into `_dispatch_job()` (v2 async) and all v1 sync handlers.

**evict_all leak fix** (`split_model_manager.py::evict_all`, v1.1.4): prior to the fix, `DenoiserWorker` held strong refs to source model builders that kept ~22 GB of encoder hub pinned after eviction. Fix: explicitly null reference paths before dropping workers. Verified: cuda:0 drops from 66.9 GB → **683 MiB** after unload. (v1.3 refactored from `ModelLedger` → `SingleGPUModelBuilder` / `CachingModelFactory` but the eviction pattern is the same.)

**Half-load recovery** (v1.4): `SplitModelManager.reset()` nulls workers + encoder_ledger + neg-prompt cache, per-GPU sync + `empty_cache()`. `_load_all_impl` sets `_last_load_failed` on exception; `_ensure_ltx_resident` calls `reset()` before retry when the flag is up.

**Swap + system endpoints** (Bearer auth required):
- `POST /v1/ltx/unload`, `POST /v1/ltx/reload`
- `POST /v1/flux/unload`, `POST /v1/flux/reload`
- `POST /v1/system/pause`, `POST /v1/system/resume` (acquire `_inference_lock`)
- `POST /v1/system/turbo` — toggle turbo mode (see above)
- `GET /v1/system/pool`, `POST /v1/system/pool/remote-workers` — Modal pool control (v1.6)
- `GET /v1/system/sampler`, `POST /v1/system/sampler` — alias for sampler/eta/stage2_sigmas subset
- `GET /v1/system/config`, `POST /v1/system/config`, `POST /v1/system/config/reset` — LTX generation parameters
- `GET /v1/system/flux-config`, `POST /v1/system/flux-config`, `POST /v1/system/flux-config/reset` — Flux-turbo tunables (v1.4)
- `GET /v1/system/gpu` — nvidia-smi telemetry
- `GET /dashboard` — dashboard SPA

**Latency**:
- Within-type (video→video, image→image same LoRA): unchanged, fast
- LTX→Flux: +3 s eviction + Flux forward pass
- Flux→LTX: +7–30 s cold LTX load + video generation
- Turbo entry: ~20 s (evict ACE+JoyAI+Flux, drain cuda:1, load LTX sidecar)
- Turbo exit: ~15 s (scale remote pool to 0, unload sidecar, restart cuda:1 tenants)
- LoRA strength changes (Flux): free runtime op

## Batch scheduler

- `POST /v2/batch` — submit 1-50 items per batch
- `GET /v2/batch/{batch_id}` — poll status + partial results
- `GET /v2/batch/{batch_id}/result/{index}` — download individual item result file
- `DELETE /v2/batch/{batch_id}` — cancel remaining items
- Items are sorted images-first (Klein before Dev) to minimize GPU swaps
- Under turbo, `_batch_worker` uses `asyncio.gather` to process in parallel (2 local + up to MAX remote)
- `MAX_BATCH_QUEUE_DEPTH` (default 5), `MAX_BATCH_ITEMS` (default 50)
- Supported item types: `text-to-image`, `image-to-image`, `image-edit`, `text-to-video`, `image-to-video`

## Keyframe symbolic indices
- `KeyframeInput.frame_index` accepts `int | "first" | "middle" | "last"`
- Negative integers supported: -1 = last frame, -12 = 12 frames before end
- Symbolic values resolved in `_resolve_keyframes(body, num_frames)` after num_frames computed
- "first"=0, "middle"=num_frames//2, "last"=num_frames-1
- Duplicate detection on resolved integer values; bounds check `frame_index >= num_frames → 422`
- Recommended strengths: first=1.0, middle=0.5, last=1.0

## Char mode — character consistency ranking
- `POST /v2/char/rank` — `rank_image_uri` + `generated_image_uri` + `prompt` → Gemma 4 31B on llama-swap as multimodal chat completion
- System prompt is `CHAR_RANKING_PROMPT` in server.py — strict JSON with face_match/eyes/proportions/overall_likeness (1-10) + structured `edits: {add, remove, modify}`
- Routing: `chat_manager.generate_chat_completion(..., model=config.CHAR_VISION_MODEL)` (default `gemma-4-31b-it`). Other chat endpoints default to `CHAT_MODEL` (`gemma-3-12b-nvfp4`).
- noodle-i Char tab runs client-side loop: generate → rank → apply edits → regenerate until score ≥ 9 or user hits Stop

## Generation history (history_store.py)
- SQLite DB at `history.db` — **WAL mode** (readers never block behind the single writer)
- Saves every completed v2 job with prompt, model, dimensions, result_uri, thumbnail
- API key hashed with SHA-256 (raw keys never stored)
- Thumbnails: 256px-wide JPEG at `thumbnails/`. Video thumbnails extract the first frame via PyAV.
- Endpoints: `GET /v2/history`, `GET /v2/history/{id}`, `GET /v2/history/{id}/image`, `GET /v2/history/{id}/thumbnail`
- 30-day retention; history manages result-file lifecycle
- `history.save()` runs in `asyncio.to_thread` task fire-and-forgotten from `worker_loop` — queue dequeues next job immediately without stalling on PyAV + SQLite (~300 ms)
- **Schema v2** (v1.4): four columns — `params_json`, `gen_config_json`, `seed`, `enhanced_prompt` — for full reproducibility. Online migration via `PRAGMA user_version`, idempotent, no backfill.
  - `params_json`: raw request body (Pydantic `body.model_dump(mode="json")`) — preserves `storage://` URIs, resolution enum, LoRA `id+strength`, keyframes symbolic indices. Music sanitizes paths back to URIs via `_sanitize_params_for_history`.
  - `gen_config_json`: LTX `_gen_config` snapshot at dispatch time OR `{turbo_steps, turbo_guidance}` for Flux-turbo. NULL for non-turbo Flux, ERNIE, JoyAI.
  - `enhanced_prompt`: LTX-rewritten prompt text when `enhance_prompt=true` (captured via `on_prompt_enhanced` callback). Always NULL for Flux/ERNIE/JoyAI/retake.

## v2 job observability (v1.1.6 / v1.1.7)

- **`Job.phase` field** — coarse post-denoise phase: `"denoising" | "decoding" | "encoding" | "saving" | None`. Denoising callbacks cap at **0.90**; the top 10% of progress is reserved for post-denoise phases emitted by `split_model_manager._run_*` and `flux_manager._generate/_img2img/_edit`.
- **`/v2/jobs/{id}`** status response exposes `phase` when processing.
- **`/v2/jobs/{id}/stream`** SSE endpoint — EventSource-compatible live status stream. Emits on `(status, progress, phase, error_code)` change, closes on terminal state, keepalive comment every 15 s. Accepts bearer header OR `?token=` query param (browsers). Replaces the 240-GET polling loop per video job with one long-lived connection.
- **`/v2/jobs/{id}/preview`** serves the on-disk thumbnail via zero-copy `FileResponse`. Fallback lazy extraction for jobs without api_key is offloaded via `asyncio.to_thread`.
- **`GET /v2/history/{id}`** — full record with parsed params + gen_config.
- **Timing logs** at every post-denoise phase boundary in `split_model_manager` (`vae_decode`, `video_decode+encode`) and `flux_manager` (`flux_webp_encode`), plus `history.save` and `encode_prompts` via `_timed`. Grep production logs: `journalctl -u taco-backend | grep -E "vae_decode|encode|history.save"`.

## Approved images (noodle-i → noodle-v pipeline)
- Manifest: `approved-images/manifest.json`
- Images stored in shared uploads dir (referenced by `storage://{uuid}` URIs)
- Endpoints: `POST /v1/approved-images`, `GET /v1/approved-images`, `GET /v1/approved-images/{id}/file`
- Per-API-key scoped (key hash in manifest entries)
- noodle-i "To Video" button uploads image then POSTs metadata
- noodle-v polls the GET endpoint to display approved feed

## Generation config (v1.3)

All LTX generation parameters stored in `.gen_config.json` (project root). Changes take effect on the next generation request — no restart.

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
- `GET/POST /v1/system/sampler` — alias for sampler/eta/stage2_sigmas subset

## Critical patterns

- `cleanup_memory()` calls gc.collect + empty_cache + synchronize on **current device only** — don't blindly dedup the 10 open-coded `gc.collect()+synchronize(device)+empty_cache()` triples in `split_model_manager.py`; several need explicit per-device sync for multi-GPU paths (`evict_transformer`, `evict_all`, `reset`, DUAL_GPU_LTX inter-stage). Load-bearing comments mark the 8 that must stay.
- `detect_params(checkpoint)` opens safetensors metadata — cache the result, don't call per-request
- `encode_prompts()` with CachingModelFactory keeps text encoder loaded — the internal `del` only drops local ref
- Retake uses `MultiModalGuider(...)` directly, NOT `create_multimodal_guider_factory().build()` — factory has no `.build()` method
- Audio latent must be trimmed/padded to `AudioLatentShape.from_video_pixel_shape(output_shape).frames`
- A2V uses `GuidedDenoiser` (static) for stage 1, frozen audio with `noise_scale=0.0`
- **IC-LoRA outpaint**: uses `VideoConditionByReferenceLatent` + `ConditioningItemAttentionStrengthWrapper` from `ltx_core.conditioning`. Letterbox uses **-1 fill value** because the VAE expects `[-1, 1]` normalized space — `-1` decodes to RGB 0,0,0, matching the LoRA's training black sentinel. `reference_downscale_factor` comes from the LoRA safetensors metadata via `_read_lora_reference_downscale_factor` (default 1).
- **Outpaint LoRA stays fused through stage 2** — accepted deviation from upstream `ICLoraPipeline`. Reloading would cost ~30 s.
- **Remote dispatch media**: `_dispatch_job_turbo_remote` must base64-encode local files (`audio_path` → `audio_b64`, etc.); Modal has no view of the local `uploads/` dir. Base64 expands 4/3; payloads up to ~135 MB for retake video.
- **Turbo entry aborts**: if `_wait_cuda1_free` times out, `_restore_cuda1_tenants()` is called before raising so we don't leave cuda:1 empty of services.

## Text encoder variants
- `GEMMA_VARIANT=default` — Google Gemma 3 12B PT (standard, BF16)
- `GEMMA_VARIANT=sikaworld` — Sikaworld abliterated FP4 (uncensored, NVFP4 quantized)
- Set via `.env` or environment variable, requires server restart
- Sikaworld path: `/mnt/nvme-1/huggingface/gemma-3-12b-sikaworld/`

## Dependencies
- **PyTorch 2.11.0+cu130** — FlexAttention/FA4 on Blackwell sm_120, SDPA auto-selects cuDNN FlashAttention
- **diffusers 0.38.0.dev0** (git main) — required for Flux2KleinKVPipeline
- **ltx-core 1.1.1** (editable install from `/mnt/nvme-1/repos/LTX-2/packages/ltx-core`) — vocoder fp32 fix, cosine tiling, layer streaming, BatchSplitAdapter, IC-LoRA conditioning primitives
- **ltx-pipelines 1.1.1** (editable install from `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines`) — SimpleDenoiser, GuidedDenoiser, FactoryGuidedDenoiser, sampler signatures
- LTX-2 repo: `/mnt/nvme-1/repos/LTX-2` (editable install, `torch~=2.7` pin is PEP 440 compatible with 2.11)
- Checkpoints: `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/` — v1.1 distilled models + spatial upscaler
- cuDNN >=9.20 (fixes conv3d memory bug) — currently 9.20.0.48
- cuBLAS >=13.2 (BF16/FP8 Blackwell speedup) — currently 13.3.0.5
- nvidia packages revert on `uv sync` — use `--no-sync` for runtime, manual pip for upgrades
- peft (LoRA loading via diffusers)
- comfy-kitchen (NVFP4 dequantization of Sikaworld text encoder)
- IC-LoRA outpaint weights: `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint` (Apache 2.0, ~1.3 GB). Install via `scripts/register_outpaint_lora.sh`.
