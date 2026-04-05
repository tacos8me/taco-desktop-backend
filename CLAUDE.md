# taco-backend

LTX-compatible inference server for noodle-i (image gen) + noodle-v (video gen).

## Structure
- `server.py` — FastAPI app, all HTTP endpoints, job queue dispatch, history + approved-images APIs
- `split_model_manager.py` — Single-GPU LTX pipeline: shared encoder hub + swappable transformer
- `flux_manager.py` — Flux 2 image generation: model swapping (Dev/Klein KV), FP8 layerwise casting
- `chat_manager.py` — Proxies /v1/chat/completions to llama-swap
- `job_queue.py` — Dual async job queues: flux_queue + ltx_queue with per-device workers
- `upload_store.py` — UUID file storage for uploads and job results
- `history_store.py` — SQLite-backed per-API-key generation history with thumbnails
- `lora_registry.py` — Flat-dir LoRA storage with registry.json index
- `nvfp4_loader.py` — NVFP4→BF16 dequantizer for ComfyUI-format safetensors (via comfy-kitchen)
- `config.py` — Paths, model mapping, device config, resolution tables

## Key commands
- Run: `bash run.sh` (sets LD_LIBRARY_PATH, PYTORCH_CUDA_ALLOC_CONF, port 8090)
- Service: `systemctl --user start/stop/restart taco-backend`
- Logs: `journalctl --user -u taco-backend -f`
- Test: `uv run --no-sync pytest tests/ -q -p no:cacheprovider`
- Health: `curl http://localhost:8090/health`

## GPU topology
- cuda:0 → RTX PRO 6000 96GB — Flux 2 Dev/Klein KV (FP8, ~18-77GB depending on model)
- cuda:1 → RTX PRO 6000 96GB — LTX (encoder hub + transformer ~69GB)
- cuda:2 → RTX PRO 4000 24GB — unused

## Concurrency
- Flux (cuda:0) and LTX (cuda:1) run **concurrently** via per-device locks
- `_flux_lock` serializes all Flux inference on cuda:0
- `_ltx_lock` serializes all LTX inference on cuda:1
- Two job queue workers (flux-worker + ltx-worker) process jobs in parallel
- Jobs routed by type: IMAGE_* → flux_queue, VIDEO_* → ltx_queue
- Pause/resume acquires both locks to drain both pipelines

## Flux pipeline details

### Model swapping (flux_manager.py)
- `flux2-dev`: Flux2Pipeline + Turbo LoRA fused + FP8 layerwise casting (~77GB)
- `flux2-klein`: Flux2KleinKVPipeline from single-file checkpoint + FP8 casting (~18GB)
- Models swap on demand — request specifies model, FluxManager evicts current and loads new
- Klein boundary layers (x_embedder, context_embedder, proj_out) kept in BF16 to reduce FP8 artifacts
- Klein is distilled (4 steps default), guidance_scale is ignored (is_distilled=True disables CFG)

### Turbo mode
- `turbo: bool` field on TextToImageRequest and ImageToImageRequest
- When `turbo=true`: server overrides to 8 steps, guidance_scale 2.5, custom sigma schedule
- Only works with flux2-dev, not flux2-klein

### Image editing
- `POST /v1/image-edit` + `POST /v2/image-edit` — multi-reference editing via Klein KV
- Accepts 1-10 images via `image_uris` list
- Always uses Klein (model defaults to flux2-klein)

## LTX pipeline details

### Models
- `ltx-2-3-fast` → distilled transformer, 8 steps, no CFG
- `ltx-2-3-pro` → dev transformer (30 euler steps + CFG + STG), then dev_lora (5 stage 2 steps)
- `ltx-2-3-hq` → dev+distilled_lora@0.25 (15 res2s steps), then dev_lora@0.50 (stage 2)

### Prompt enhancement
- `enhance_prompt: bool = false` on t2v, i2v, a2v endpoints
- Uses Gemma 3's model.generate() to cinematically rewrite terse prompts (~2-5s overhead)
- For i2v: first keyframe image used as visual context for vision-aware rewriting

### VAE decode
- Temporal-only tiling: tile_size=128 frames, overlap=32 frames, no spatial tiling
- Videos ≤257 frames (~10s at 24fps) use single-pass decode (no tiling at all)
- Stage 2 uses 5 distilled steps (upstream default is 3) for better motion resolution
- Decoder runs in bfloat16 with cuDNN 9.20 (fixes conv3d memory bug)
- Ghosting during fast motion is partially architectural (1:192 compression ratio, causal conv3d)

## Precision settings (config.py)
- `torch.backends.cuda.matmul.allow_tf32 = False` — full float32 precision for matmuls
- `torch.backends.cudnn.allow_tf32 = False` — full float32 for cuDNN convolutions
- `torch.backends.cudnn.deterministic = True` — stable algorithm selection
- Note: these only affect float32 ops. VAE runs in bfloat16 (cuDNN uses float32 accumulators internally)

## System endpoints
- `POST /v1/system/pause` — evicts all models from both GPUs, cancels queued jobs, returns 503 while paused
- `POST /v1/system/resume` — reloads all models, re-enables inference
- `GET /health` — returns `{"status": "paused"}` when paused, `{"status": "ok"}` when ready

## Text encoder variants
- `GEMMA_VARIANT=default` — Google Gemma 3 12B PT (standard, BF16)
- `GEMMA_VARIANT=sikaworld` — Sikaworld abliterated FP4 (uncensored, NVFP4 quantized, dequantized to BF16 via comfy-kitchen)
- Set via `.env` or environment variable, requires server restart

## Critical patterns
- `cleanup_memory()` calls gc.collect + empty_cache + synchronize — avoid redundant calls after evict_transformer (which already syncs+clears)
- `detect_params(checkpoint)` opens safetensors metadata — cached as module-level `_DEV_PARAMS` constant
- `encode_prompts()` with CachingModelLedger keeps text encoder loaded — the internal `del` only drops local ref
- Retake uses `MultiModalGuider(...)` directly, NOT `create_multimodal_guider_factory().build()` — factory has no `.build()` method
- Audio latent must be trimmed/padded to `AudioLatentShape.from_video_pixel_shape(output_shape).frames`
- Free transformer before upsample_video — upsample doesn't need it, saves ~22GB during stage transition
- Free video_encoder + spatial_upsampler from cache before VAE decode (all pipelines)

## Dependencies
- LTX-2 repo: `/mnt/nvme-1/repos/LTX-2` (editable install)
- Checkpoints: `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/`
- cuDNN >=9.15 required (fixes conv3d memory bug in VAE decode) — currently 9.20
- nvidia-cublas 13.2+ (20% BF16/FP8 speedup on Blackwell) — currently 13.3
- comfy-kitchen (required for NVFP4 dequantization of Sikaworld text encoder)
- diffusers from git main (required for Flux2KleinKVPipeline)
- peft (required for LoRA loading via diffusers)
- LD_LIBRARY_PATH must include cudnn/lib (run.sh handles this)
