# Session Resume — 2026-03-12

## Project Dir
`/mnt/nvme-1/servers/taco-backend`

## Server Status
Running on port 8090. Start: `bash run.sh > /tmp/taco-backend.log 2>&1 &`

## What Was Done This Session

### LTX 2.3 Codebase Audit (5-agent review vs reference repo)
Found and fixed the following issues in `split_model_manager.py`:

1. **Retake crash (CRITICAL)** — `.build()` doesn't exist on `MultiModalGuiderFactory`. Replaced with direct `MultiModalGuider(params=..., negative_context=...)` matching reference `retake.py`.

2. **Retake audio latent shape mismatch** — Raw encoded audio latent wasn't trimmed/padded to match `AudioLatentShape.from_video_pixel_shape(output_shape).frames`. Added trim/pad logic from reference.

3. **Retake OOM on VAE decode** — Transformer (~22GB) stayed loaded during decode. Added `evict_transformer()` + `cleanup_memory()` before decode.

4. **All pipelines: unnecessary `ensure_transformer("dev")` before decode** — Replaced with `evict_transformer()` in t2v, t2v_hq, i2v, a2v. Frees ~22GB without loading a replacement, saving ~15s per request AND reducing OOM risk during decode.

### Previous Session (carried forward, uncommitted)
- LoRA system (lora_registry.py, server.py endpoints, split_model_manager.py integration)
- `python-multipart` dependency + `max_part_size` override for LoRA uploads
- Retake OOM fix (evict transformer before VAE encode)

## Verified Correct (no changes needed)
- Generation params (steps, CFG, STG, sigmas, rescale) match reference
- LoRA strengths (HQ 0.25/0.50, pro 1.0) correct
- Sampler selection (Euler for pro/fast, Res2s for HQ) correct
- VAE decode tiling, seed/RNG, negative prompts all correct

## Not Fixed (product decision)
- **Prompt enhancement** — Reference supports `enhance_prompt=True` (Gemma 3 as LLM expands short prompts). Not exposed in our API. Could improve quality for short/vague prompts.

## Open Issue
- **Client 413 on LoRA upload**: POST /v1/loras never reaches server from client. Works fine via curl. Likely tunnel/proxy issue. Workaround: drop file in `loras/` dir and register via localhost API.
- **Registered LoRA**: "Farm MK1 S2000" (193MB) — ID `6c69727201bb48dbb8e9ec95ae73720e`

## Uncommitted Changes
- `split_model_manager.py`: retake guider fix, audio latent trim/pad, evict_transformer in all decode paths
- `server.py`: `request.form(max_part_size=config.MAX_LORA_SIZE_BYTES)`
- `pyproject.toml` / `uv.lock`: added `python-multipart` dependency
