# taco-backend

LTX-compatible inference server for noodle-i (image gen) + noodle-v (video gen).

**Version**: v1.1.8 (2026-04-11).

## Structure
- `server.py` — FastAPI app, all HTTP endpoints, job queue dispatch, history + approved-images APIs
- `split_model_manager.py` — Single-GPU LTX pipeline: shared encoder hub + swappable transformer
- `flux_manager.py` — Flux 2 image generation: per-request LoRA fusion + FP8 layerwise casting on cuda:0
- `chat_manager.py` — Proxies /v1/chat/completions to llama-swap (supports per-request model override for vision ranking)
- `job_queue.py` — Async job queue: submit (202), poll, result, cancel; saves to history on completion
- `upload_store.py` — UUID file storage for uploads and job results
- `history_store.py` — SQLite-backed per-API-key generation history with thumbnails
- `lora_registry.py` — Flat-dir LoRA storage with registry.json index
- `config.py` — Paths, model mapping, device config, resolution tables, TF32 settings

## Key commands
- Run: `bash run.sh` (sets LD_LIBRARY_PATH, PYTORCH_CUDA_ALLOC_CONF, port 8090)
- Test: `uv run pytest tests/ -v`
- Health: `curl http://localhost:8090/health`

## GPU topology (v1.1.4 — single-GPU swap mode, extended to three tenants in v1.1.8)
- **cuda:0** → RTX PRO 6000 Blackwell 96GB — **Flux 2 + LTX + JoyAI-edit sidecar**, all mutually exclusive, auto-swapped on dispatch (v1.1.8 added JoyAI as a third tenant via an out-of-process sidecar — see the "JoyAI image-edit sidecar" subsection below)
- **cuda:1** → RTX PRO 6000 Blackwell 96GB — **reserved exclusively for external training runs** (e.g. ai-toolkit). taco-backend never touches it.

Verified via `nvidia-smi -L`. No third GPU on this box — any earlier references to `cuda:2`/RTX 4000 are stale.

**Why mutually exclusive**: LTX active is ~79 GB (60 GB transformer + 19 GB encoder hub + decoder activations) and Flux active is ~81 GB (60 GB transformer + ~14 GB CPU-offload forward-pass peak via `enable_model_cpu_offload`). Combined ~160 GB > 96 GB physical. They cannot coexist on one GPU during forward pass, so the server evicts the other before running.

**Auto-swap (three tenants, v1.1.8)**: `server.py` exposes `_ensure_ltx_resident()`, `_ensure_flux_ready()`, and `_ensure_joyai_ready()` helpers, called inside `_inference_lock` by `_dispatch_job()` (v2 async) and every v1 sync handler (text_to_video, image_to_video, audio_to_video, retake, text_to_image, image_to_image, image_edit) before `torch.cuda.set_device()`. All three tenants (LTX, Flux, JoyAI) are mutually exclusive on cuda:0. LTX is **not** auto-reloaded after a Flux/JoyAI request — it stays evicted until the next video request. Long-stretch single-tenant workloads incur zero swap overhead; mixed workloads pay a per-direction-change cost (see swap section below).

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

## API contract
- **`docs/API.md` is the canonical, client-facing API spec.** Any commit that adds, removes, or changes an endpoint (URL, method, request/response shape, status codes, auth requirements) MUST update `docs/API.md` in the **same commit**. Do not split "code change" from "doc change" across commits — the doc is the contract the other-side developer reads, and drift breaks them silently.
- When touching `server.py` endpoints, re-read `docs/API.md` before writing code so you know what the external shape is supposed to be, then update both sides together.

## Conventions
- All generation runs under `@torch.inference_mode()`
- Flux output: WEBP quality 95
- LTX output: raw MP4 bytes with `Content-Type: video/mp4`
- LTX: evict transformer before VAE decode (reclaims ~22GB), don't reload after — next request handles its own state
- LTX swap: on dispatch, `_ensure_ltx_resident()` / `_ensure_flux_ready()` (server.py) auto-evicts the other manager before each request. Never assume either manager is loaded — call the helper inside `_inference_lock`.
- LTX LoRA: fusion is permanent (no unfuse), different strengths require full transformer reload. Cache key `(state_name, user_lora_tuple)`.
- Flux LoRA: adapter mode (NOT fused) — strength is applied at inference time via `pipe.set_adapters([...], [strength])`. Cache key `(model_name, lora_path)` — strength is NOT in the key, so strength changes are free. Only model or LoRA file changes trigger reload.
- Frame count must be 8k+1; resolution multiples of 64
- Port 8090, auth via `.api_keys` file (disabled when empty)

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

## Single-GPU swap mode (v1.1.4, three tenants v1.1.8)

Both Flux and LTX target `cuda:0`. `config.py` sets `LTX_DEVICE = FLUX_DEVICE = "cuda:0"`. `cuda:1` is reserved for external training runs and the backend never reads/writes it. **v1.1.8 added a third tenant**: the JoyAI image-edit sidecar, which also lives on `cuda:0` and is mutually exclusive with both LTX and Flux. See the "JoyAI image-edit sidecar" subsection below for details.

**Auto-swap helpers** (`server.py`):
- `_ensure_ltx_resident()` — no-op if `ltx_manager.is_ready`, else calls `ltx_manager.load_all()` (cold load is 7–30 s depending on OS page cache)
- `_ensure_flux_ready()` — no-op if `not ltx_manager.is_ready`, else calls `ltx_manager.evict_all()` (~3 s)
- Both **must** be called while holding `_inference_lock`. Wired into `_dispatch_job()` (v2 async queue) and all 7 v1 sync handlers. The old `if not manager.is_ready: return 500` early-return guards have been removed from LTX sync handlers — auto-swap handles readiness lazily.

**evict_all leak fix** (`split_model_manager.py::evict_all`): each `DenoiserWorker` holds two strong refs to its source `ModelLedger` — a direct `worker._model_ledger` (set at split_model_manager.py:392) and an indirect `worker.ledger._source_ledger` via `CachingModelLedger(source_ledger=...)`. Prior to v1.1.4 neither was cleared, so the encoder hub (Gemma text encoder + VAE + spatial upsampler, ~22 GB) stayed pinned on the LTX device after `/v1/ltx/unload`. Fix: explicitly null both paths plus `_encoder_ledger._source_ledger` defensively before dropping the workers list. Verified: cuda:0 drops from 66.9 GB → **683 MiB** after unload.

**Swap endpoints** (Bearer auth required):
- `POST /v1/ltx/unload`, `POST /v1/ltx/reload`
- `POST /v1/flux/unload`, `POST /v1/flux/reload`
- `POST /v1/system/pause`, `POST /v1/system/resume` (acquire `_inference_lock`)

**Latency**:
- Within-type (video→video, image→image with same LoRA): unchanged, fast
- LTX→Flux (image request after video): +3 s eviction + normal Flux forward pass
- Flux→LTX (video request after image): +7–30 s cold LTX load + normal video generation
- LoRA strength changes: still free runtime op, unchanged by the swap refactor

### JoyAI image-edit sidecar (v1.1.8)

Third tenant of the cuda:0 mutual-exclusion slot, running **out-of-process** at
`/mnt/nvme-1/servers/joyai-sidecar/` on `127.0.0.1:8092` (8091 is taken by
`noodle-t-backend`). Separate venv because JoyAI needs `transformers==4.57.1`
+ `diffusers` fork (Moran232/diffusers@joyimage_edit), incompatible with
taco-backend's `transformers==5.3` + `diffusers==0.38.0.dev0`.

- Activation: `LOAD_JOYAI=1` env var in `.env` (off by default).
- Dispatch: `/v1/image-edit` and `/v2/image-edit` with `model="joyai-edit"` route to `joyai_client.edit()` instead of `flux.generate_image_edit()`.
- GPU coordination: taco-backend holds `_inference_lock` for the full HTTP call. Sidecar's internal asyncio.Lock is defense-in-depth.
- Three-tenant swap: `_evict_other_tenants(new)` in server.py flips any tenant→`new`. `_ensure_joyai_ready()` mirrors `_ensure_ltx_resident()` / `_ensure_flux_ready()`. All three helpers are now `async def` (was sync in v1.1.7).
- Phase 0 measurement: 50.3 GB resident, 65.5 GB peak reserved, 78 s for 30 steps at 1024² bf16. Within the 96 GB budget with headroom.
- Memory pressure: JoyAI + LTX (129 GB) and JoyAI + Flux peak (125 GB) both exceed 96 GB — always evict before swapping.
- Service: `systemctl --user {start,stop,restart,status} joyai-sidecar`. Unit ordered `Before=taco-backend.service`.
- Fallback on sidecar failure: `/v2/image-edit` with `model="joyai-edit"` returns `503 {"error": "sidecar_unreachable"}` — client should retry with `flux2-klein`.
- **Pre-existing bug fixed in same commit**: `flux_manager._edit()` hardcoded `ensure_model("flux2-klein", ...)`, silently ignoring `model="flux2-dev"`. Now respects the passed model.

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
- Endpoints: `GET /v2/history`, `GET /v2/history/{id}/image`, `GET /v2/history/{id}/thumbnail`
- Cleanup: job_queue keeps completed result files (history manages lifecycle), 30-day retention
- Both noodle-i and noodle-v consume the same history API
- **`history.save()` runs in an `asyncio.to_thread` task** (v1.1.6) fire-and-forgotten from `worker_loop`, so the queue worker dequeues the next job immediately instead of stalling ~300 ms per job on PyAV + SQLite

## v2 job observability (v1.1.6 / v1.1.7)

- **`Job.phase` field** — coarse post-denoise phase: `"denoising" | "decoding" | "encoding" | "saving" | None`. Denoising callbacks cap at **0.90** (was 0.99 in v1.1.4); the top 10% of progress is reserved for post-denoise phases emitted explicitly by `split_model_manager._run_*` and `flux_manager._generate/_img2img/_edit`.
- **`/v2/jobs/{id}` status response** exposes `phase` when processing.
- **`/v2/jobs/{id}/stream` SSE endpoint** (v1.1.7) — EventSource-compatible live status stream. Emits on `(status, progress, phase, error_code)` change, closes on terminal state, keepalive comment every 15 s. Accepts bearer header OR `?token=` query param (browsers). Replaces the 240-GET polling loop per video job with one long-lived connection.
- **`/v2/jobs/{id}/preview`** (v1.1.6) serves the on-disk thumbnail written by `history.save()` via zero-copy `FileResponse`. Fallback lazy extraction still exists for jobs without api_key but is offloaded via `asyncio.to_thread`.
- **Timing logs** at every post-denoise phase boundary in `split_model_manager` (`vae_decode`, `video_decode+encode`) and `flux_manager` (`flux_webp_encode`), plus `history.save` in `job_queue`. Grep production logs for real wall-clock per phase: `journalctl -u taco-backend | grep -E "vae_decode|encode|history.save"`.

## Approved images (noodle-i → noodle-v pipeline)
- Manifest: `/mnt/nvme-1/servers/taco-backend/approved-images/manifest.json`
- Images stored in shared uploads dir (referenced by `storage://{uuid}` URIs)
- Endpoints: `POST /v1/approved-images`, `GET /v1/approved-images`, `GET /v1/approved-images/{id}/file`
- Per-API-key scoped (key hash in manifest entries)
- noodle-i "To Video" button uploads image then POSTs metadata
- noodle-v polls the GET endpoint to display approved feed

## Critical patterns
- `cleanup_memory()` calls gc.collect + empty_cache + synchronize — avoid redundant calls after evict_transformer (which already syncs+clears)
- `detect_params(checkpoint)` opens safetensors metadata — cache the result, don't call per-request
- `encode_prompts()` with CachingModelLedger keeps text encoder loaded — the internal `del` only drops local ref
- Retake uses `MultiModalGuider(...)` directly, NOT `create_multimodal_guider_factory().build()` — factory has no `.build()` method
- Audio latent must be trimmed/padded to `AudioLatentShape.from_video_pixel_shape(output_shape).frames`

## Text encoder variants
- `GEMMA_VARIANT=default` — Google Gemma 3 12B PT (standard, BF16)
- `GEMMA_VARIANT=sikaworld` — Sikaworld abliterated FP4 (uncensored, NVFP4 quantized)
- Set via `.env` or environment variable, requires server restart
- Sikaworld path: `/mnt/nvme-1/huggingface/gemma-3-12b-sikaworld/`

## Dependencies
- **PyTorch 2.11.0+cu130** — FlexAttention/FA4 on Blackwell sm_120, SDPA auto-selects cuDNN FlashAttention
- **diffusers 0.38.0.dev0** (git main) — required for Flux2KleinKVPipeline (not in any stable release)
- LTX-2 repo: `/mnt/nvme-1/repos/LTX-2` (editable install, `torch~=2.7` pin is PEP 440 compatible with 2.11)
- Checkpoints: `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/`
- cuDNN >=9.20 (fixes conv3d memory bug) — currently 9.20.0.48
- cuBLAS >=13.2 (BF16/FP8 Blackwell speedup) — currently 13.3.0.5
- nvidia packages revert on `uv sync` — use `--no-sync` for runtime, manual pip for upgrades
- peft (required for LoRA loading via diffusers)
- comfy-kitchen (required for NVFP4 dequantization of Sikaworld text encoder)
