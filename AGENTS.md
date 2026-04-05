# AGENTS.md — taco-backend

## Performance Optimization Backlog

Identified via full codebase audit (2026-03-14), updated 2026-04-05. Items grouped by tier.

### Tier 1 — Completed

1. ~~**CUDA_MODULE_LOADING=LAZY**~~ — DONE in run.sh
2. ~~**CRF 18 + preset fast**~~ — DONE in media_io.py
3. ~~**Flux PNG → WEBP output**~~ — DONE (quality 95)
4. ~~**FileResponse for v2 results**~~ — DONE
5. ~~**Cache detect_params at module level**~~ — DONE (_DEV_PARAMS constant)
6. ~~**Atomic LoRA registry write**~~ — DONE (write-to-temp-then-rename)
7. ~~**TOKENIZERS_PARALLELISM=false**~~ — DONE in run.sh
8. ~~**Fix sigma schedule**~~ — DONE (pass latent= to LTX2Scheduler)
9. ~~**AAC audio bitrate 128k → 192k**~~ — DONE
10. ~~**Remove redundant cleanup_memory()**~~ — DONE (gc.collect after evict)
11. ~~**torch.tensor on device**~~ — DONE (sigma tensors created directly on GPU)
12. ~~**Concurrent Flux + LTX**~~ — DONE (per-device locks, dual job queues)
13. ~~**Prompt enhancement**~~ — DONE (enhance_prompt: bool on t2v/i2v/a2v)
14. ~~**Free transformer before upsample**~~ — DONE (all 4 pipelines)
15. ~~**Consistent cache cleanup**~~ — DONE (i2v/a2v now match t2v/hq)
16. ~~**Stage 2: 5 steps**~~ — DONE (was 3, better motion resolution)

### Tier 2 — Remaining

17. **TF32 precision** — REVERSED: TF32 now DISABLED. Was degrading VAE quality on Blackwell. The TF32 flags only affect float32 ops; VAE runs in bfloat16 so they're vestigial.
18. **FP8 cast quantization** — add QuantizationPolicy.fp8_cast() to ModelLedger, saves ~11GB VRAM per transformer
19. **Cache negative prompt embeddings** — cache ctx_n for DEFAULT_NEGATIVE_PROMPT, saves ~0.2s/request
20. **BytesIO for encode_video** — can't do: av.open needs format= for file-like objects
21. **Gradient estimating euler** — imported behind USE_GE_EULER flag, untested. Lightricks exports but uses in zero pipelines.

### Tier 3 — Experimental / Higher Effort

22. **torch.compile on transformer** — 20-40% denoising speedup, needs careful testing with weight swaps
23. **FlashAttention3** — install flash_attn_interface for sm_100 Blackwell, auto-detected by existing code
24. **Streaming uploads** — request.stream() instead of request.body() for large files
25. **Float32 VAE decode** — load decoder in float32 permanently at startup (+777MB, zero per-request overhead). Tested: per-call .to(float32) was too slow (minutes overhead). Permanent upcast viable but untested for quality improvement.
26. **STG skip_step=1** — tested, reverted. Saved 33% NFE but created temporal oscillation in guidance signal contributing to ghost trails.

## VAE Decode Quality Notes

### Known architectural limitations (can't fix without retraining)
- **1:192 compression ratio** (32x32x8 pixels per latent token) — most aggressive in current video diffusion
- **Causal conv3d symmetric padding** (decoder causal=False) — ±8 frame temporal bleed per conv layer
- **decode_noise_scale=0.025** is dead code for current checkpoint (timestep_conditioning=False)
- Fast motion ghosting is inherent to the architecture — objects moving >32px between frames exceed single latent cell's temporal span

### Tiling configuration rationale
- Temporal-only tiling (no spatial) to avoid visible grid seams
- Videos ≤257 frames (~10s): single-pass decode, zero tiling artifacts
- Longer videos: 128-frame tiles, 32-frame overlap
- Previous 56-frame overlap was counterproductive (70% of frames in blend zone → more smearing)

## Klein KV Integration Notes

- Uses Flux2KleinKVPipeline (not Flux2KleinPipeline) — requires diffusers from git main
- KV variant caches reference image K/V projections for 2.5x multi-ref speedup
- Loaded via from_single_file (repo shards are gated, single checkpoint is available)
- Boundary layers (x_embedder, context_embedder, proj_out) excluded from FP8 to reduce 4-step artifacts
- guidance_scale is stripped from kwargs — Klein is distilled, CFG disabled entirely
- gc.collect + empty_cache after each Klein gen (VRAM leak mitigation, diffusers #13079)

## Sikaworld Text Encoder Notes

- NVFP4 format requires comfy-kitchen's native CUDA dequantization kernel
- Manual FP4 unpacking was wrong (cosine sim -0.001 vs ground truth) — NVFP4 block scales use swizzled layout
- Dequantized to BF16 + merged with vision_tower/multi_modal_projector from standard Gemma
- Key prefix: Sikaworld uses `model.*`, standard uses `language_model.model.*` — dequantizer adds prefix
