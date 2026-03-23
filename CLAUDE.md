# taco-backend

LTX-compatible inference server for taco-desktop.

## Structure
- `server.py` — FastAPI app, all HTTP endpoints, job queue dispatch
- `split_model_manager.py` — Single-GPU LTX pipeline: shared encoder hub + swappable transformer
- `flux_manager.py` — Flux 2 Dev FP8 image generation on cuda:1
- `chat_manager.py` — Proxies /v1/chat/completions to llama-swap
- `job_queue.py` — Async job queue: submit (202), poll, result, cancel
- `upload_store.py` — UUID file storage for uploads and job results
- `lora_registry.py` — Flat-dir LoRA storage with registry.json index
- `config.py` — Paths, model mapping, device config, resolution tables

## Key commands
- Run: `bash run.sh` (sets LD_LIBRARY_PATH, PYTORCH_CUDA_ALLOC_CONF, port 8090)
- Test: `uv run pytest tests/ -v`
- Health: `curl http://localhost:8090/health`

## GPU topology
- cuda:0 → RTX PRO 6000 96GB — LTX (encoder hub + transformer ~69GB)
- cuda:1 → RTX PRO 6000 96GB — Flux 2 Dev FP8 (~77GB)
- cuda:2 → RTX PRO 4000 24GB — unused

## Conventions
- All generation runs under `@torch.inference_mode()`
- Return raw MP4 bytes with `Content-Type: video/mp4`
- Evict transformer before VAE decode (reclaims ~22GB), don't reload after — next request handles its own state
- LoRA fusion is permanent (no unfuse) — different strengths require full transformer reload
- Cache key for transformer: `(state_name, user_lora_tuple)` — prevents unnecessary reloads
- Frame count must be 8k+1; resolution multiples of 64
- Port 8090, auth via `.api_keys` file (disabled when empty)

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
- LTX-2 repo: `/mnt/nvme-1/repos/LTX-2` (editable install)
- Checkpoints: `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/`
- cuDNN >=9.15 required (fixes conv3d memory bug in VAE decode)
- nvidia-cublas 13.2+ (20% BF16/FP8 speedup on Blackwell)
