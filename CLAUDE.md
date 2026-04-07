# taco-backend

LTX-compatible inference server for noodle-i (image gen) + noodle-v (video gen).

## Structure
- `server.py` — FastAPI app, all HTTP endpoints, job queue dispatch, history + approved-images APIs
- `split_model_manager.py` — Single-GPU LTX pipeline: shared encoder hub + swappable transformer
- `flux_manager.py` — Flux 2 image generation: fused Turbo LoRA + FP8 layerwise casting on cuda:0
- `chat_manager.py` — Proxies /v1/chat/completions to llama-swap
- `job_queue.py` — Async job queue: submit (202), poll, result, cancel; saves to history on completion
- `upload_store.py` — UUID file storage for uploads and job results
- `history_store.py` — SQLite-backed per-API-key generation history with thumbnails
- `lora_registry.py` — Flat-dir LoRA storage with registry.json index
- `config.py` — Paths, model mapping, device config, resolution tables, TF32 settings

## Key commands
- Run: `bash run.sh` (sets LD_LIBRARY_PATH, PYTORCH_CUDA_ALLOC_CONF, port 8090)
- Test: `uv run pytest tests/ -v`
- Health: `curl http://localhost:8090/health`

## GPU topology
- cuda:0 → RTX PRO 6000 96GB — Flux 2 Dev/Klein (FP8 transformer ~32GB + text encoder + VAE ~77GB total)
- cuda:1 → RTX PRO 6000 96GB — LTX (encoder hub + transformer ~69GB)
- cuda:2 → RTX PRO 4000 24GB — unused

## Flux pipeline details

### Model loading (flux_manager.py)
1. Load base Flux2Pipeline from `black-forest-labs/FLUX.2-dev` in bf16
2. Load fal/FLUX.2-dev-Turbo LoRA weights (2.76GB)
3. Fuse LoRA into base weights (`pipe.fuse_lora()`) then unload LoRA adapter
4. Apply FP8 layerwise casting (`storage_dtype=float8_e4m3fn, compute_dtype=bfloat16`) on the fused transformer
5. Move pipeline to GPU

Full bf16 does NOT fit — transformer (64GB) + Mistral-3 text encoder (~24GB) + VAE exceeds 95GB VRAM. FP8 layerwise casting halves transformer to ~32GB making it fit.

### Turbo mode
- `turbo: bool` field on `TextToImageRequest` and `ImageToImageRequest`
- When `turbo=true`: server overrides to 8 steps, guidance_scale 2.5, custom sigma schedule
- When `turbo=false`: uses client-provided steps/guidance, default scheduler sigmas
- The Turbo LoRA is always fused — turbo vs standard is controlled by the sigma schedule and step count, not by LoRA weight toggling

### Turbo sigma schedule (config.py)
```python
FLUX_TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]
```

### Klein model
- Loaded via `from_single_file()` from the HF cache
- Also uses FP8 layerwise casting
- No Turbo LoRA (Klein has its own step distillation at 4 steps)

## Precision settings (config.py)
- `torch.backends.cuda.matmul.allow_tf32 = False` — full float32 precision for VAE decode
- `torch.backends.cudnn.allow_tf32 = False` — full float32 for VAE convolutions
- TF32 was previously enabled but degraded VAE output quality on Blackwell GPUs (VAE uses `force_upcast=True` expecting real float32)

## Conventions
- All generation runs under `@torch.inference_mode()`
- Flux output: WEBP quality 95
- LTX output: raw MP4 bytes with `Content-Type: video/mp4`
- Evict transformer before VAE decode (reclaims ~22GB), don't reload after — next request handles its own state
- LoRA fusion is permanent (no unfuse) — different strengths require full transformer reload
- Cache key for transformer: `(state_name, user_lora_tuple)` — prevents unnecessary reloads
- Frame count must be 8k+1; resolution multiples of 64
- Port 8090, auth via `.api_keys` file (disabled when empty)

## Keyframe symbolic indices (v1.1)
- `KeyframeInput.frame_index` accepts `int | "first" | "middle" | "last"`
- Negative integers supported: -1 = last frame, -12 = 12 frames before end
- Symbolic values resolved in `_resolve_keyframes(body, num_frames)` after num_frames computed
- "first"=0, "middle"=num_frames//2, "last"=num_frames-1
- Duplicate detection on resolved integer values (e.g., "first" and 0 conflict → 422)
- Bounds check: frame_index >= num_frames → 422
- Recommended strengths: first=1.0, middle=0.5, last=1.0

## Generation history (history_store.py)
- SQLite DB at `/mnt/nvme-1/servers/taco-backend/history.db`
- Saves every completed v2 job with prompt, model, dimensions, result_uri, thumbnail
- API key hashed with SHA-256 (raw keys never stored)
- Thumbnails: 256px-wide JPEG at `/mnt/nvme-1/servers/taco-backend/thumbnails/`
- Endpoints: `GET /v2/history`, `GET /v2/history/{id}/image`, `GET /v2/history/{id}/thumbnail`
- Cleanup: job_queue keeps completed result files (history manages lifecycle), 30-day retention
- Both noodle-i and noodle-v consume the same history API

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
