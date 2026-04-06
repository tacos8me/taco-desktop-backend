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

### VAE Tiling Notes (from ComfyUI comparison)

ComfyUI uses spatial tiling (512/64px) + temporal_size=4096 (effectively no temporal tiling). Our approach:
- SHORT_VIDEO_THRESHOLD=257 — no tiling for ≤10s videos (best quality, single-pass)
- Temporal-only tiling (128/32) for longer videos
- Spatial tiling disabled (caused grid seams)
- If re-enabling spatial tiling: use 512/64 with cosine S-curve blending (already in ltx-core)

### Tier 3 — Experimental / Higher Effort

15. **torch.compile on transformer** — 20-40% denoising speedup, needs careful testing with weight swaps
16. **FlashAttention3** — install flash_attn_interface for sm_100 Blackwell, auto-detected by existing code
17. **Streaming uploads** — request.stream() instead of request.body() for large files
18. **torch.compile cache** — TORCHINDUCTOR_CACHE_DIR + FX_GRAPH_CACHE for persistent compilation
21. **Retake distilled mode** — upstream supports `distilled=True` for fast retake
22. **ComfyUI last_frame_fix** — append/trim last latent frame to fix end-of-video causal conv artifacts
23. **IC-LoRA pipeline** — motion anchoring for temporal coherence (upstream `ICLoraPipeline`)

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
