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
24. ~~**Flux 2 LoRA (folder-drop)**~~ — DONE (2026-04-09). `flux_lora_registry.py` scans `flux_loras/` for `.safetensors`; optional sidecar `.json` for metadata; `GET /v1/flux-loras` + `POST /v1/flux-loras/rescan`; `lora: {id, strength}` field on Flux image requests; fuse-before-FP8-cast with `(model, user_lora)` cache key. Works on both flux2-dev and flux2-klein.
25. ~~**Dithering artifacts on flux2-dev fixed**~~ — DONE (2026-04-06). Root cause: FP8 layerwise casting was quantizing `x_embedder` and `context_embedder` (input projections) into FP8, losing precision on the initial 128→6144 and 15360→6144 projections. Error propagated through all 56 transformer layers. Fix: `skip_modules_pattern=["x_embedder","context_embedder","proj_out"]` on both dev and klein `enable_layerwise_casting()` calls. Klein had already been set correctly; dev was not. Also stopped hard-fusing Turbo LoRA at load time (it's now a regular folder-drop LoRA with ID `flux2-turbo`) — the fused turbo weights + FP8 cast were shifting weights to non-standard FP8 grid points creating structured dithering patterns. See `docs/audit-2026-04-06-comfyui-comparison.md`.
26. ~~**Char mode vision ranking**~~ — DONE. `POST /v2/char/rank` routes to Gemma 4 31B (`gemma-4-31b-it` on llama-swap) via `chat_manager.generate_chat_completion(..., model=...)`. Returns structured JSON with face_match/eyes/proportions/overall_likeness + add/remove/modify edits. Client-side loop in noodle-i drives iterative refinement.
27. ~~**Full bf16 Flux 2 Dev (drop FP8 layerwise casting, adapter-mode LoRA)**~~ — DONE (2026-04-09, v1.1.1). Root cause: the v1.1 folder-drop LoRA feature reintroduced the exact `fuse_lora → enable_layerwise_casting(float8_e4m3fn)` sequence that was previously diagnosed as causing structured dithering with the hardcoded Turbo LoRA (entry 25, 2026-04-06). Diffusers PR #10685 documents the root cause: PEFT's input autocast hook forces compute back into FP8, defeating `compute_dtype=bfloat16`, and fused weights sit on non-standard FP8 grid points creating structured artifacts. Fix: (a) drop `enable_layerwise_casting` entirely — run full bf16 transformer (matches ComfyUI default per their issue #10087); (b) add `pipe.enable_model_cpu_offload(device="cuda:0")` on Dev branch (bf16 all-resident is 105.9 GB > 96 GB, so page TE↔GPU around prompt encoding); (c) switch LoRAs from fusion to **adapter mode** via `load_lora_weights(adapter_name="user_lora")` + runtime `set_adapters(["user_lora"], [strength])`. Cache key shrinks from `(model, (path, strength))` to `(model, path)`, so **strength-slider changes are now free** (solves the v1.1 reload-on-every-tick UX bug). Klein fits full bf16 resident without offload. Verified end-to-end: prototype script generates 4 back-to-back images at strengths 1.0/0.4/0.0/disable without a single pipeline reload, no OOM, no screendoor.
28. ~~**Single-GPU swap mode + LTX evict leak fix**~~ — DONE (2026-04-11, **v1.1.4**, commits `d41b742` + `ab9fac8`). Two-part change that consolidates taco-backend onto `cuda:0` and frees `cuda:1` for external training.

    **(a) evict_all leak fix** (`split_model_manager.py::evict_all`, commit `d41b742`). Root cause: each `DenoiserWorker` holds TWO strong references to its source `ModelLedger` — direct via `worker._model_ledger` (set at split_model_manager.py:392) AND indirect via `worker.ledger._source_ledger` (from `CachingModelLedger(source_ledger=...)`). The ModelLedger's internal registry + weight builders kept GPU tensors pinned even after `worker.cache[key] = None` and `self._workers.clear()`. Result: ~22 GB of encoder-hub weights (Gemma text encoder + VAE + spatial upsampler) stayed resident on the LTX device after `/v1/ltx/unload`. Fix: explicitly null both reference paths in `evict_all()` before dropping the workers list, and null `_encoder_ledger._source_ledger` defensively. **Verified live: cuda:0 drops from 66.9 GB → 683 MiB after `/v1/ltx/unload`** (99% reclamation, vs 22+ GB residual before the fix).

    **(b) Single-GPU swap mode** (`server.py` + `config.py`, commit `ab9fac8`). `LTX_DEVICE` changed from `"cuda:1"` → `"cuda:0"`; `FLUX_DEVICE` stays `"cuda:0"`. Both managers target the same physical GPU. **`cuda:1` is now reserved exclusively for external training runs** — taco-backend never touches it. (Rationale: an ai-toolkit training run `train_flux2_klein_bong_a_v9_1.yaml` was active on cuda:1 during the refactor and made steps 506 → 1001 uninterrupted through the restart + test cycle.) LTX and Flux are mutually exclusive on cuda:0: ~79 GB LTX (60 GB transformer + 19 GB encoder hub + decoder activations) + ~81 GB Flux (60 GB transformer + 14 GB `enable_model_cpu_offload` forward-pass peak) = ~160 GB > 96 GB. They cannot coexist during forward pass. **Auto-swap helpers** added to server.py: `_ensure_ltx_resident()` (no-op if ready, else `manager.load_all()`) and `_ensure_flux_ready()` (no-op if LTX not ready, else `manager.evict_all()`). Both MUST be called while holding `_inference_lock`. Wired into `_dispatch_job()` (v2 async queue) and all 7 v1 sync handlers (text_to_video, image_to_video, audio_to_video, retake, text_to_image, image_to_image, image_edit), inside the `async with _inference_lock` block before `torch.cuda.set_device()`. The `if not manager.is_ready: return 500` early-return guards were removed from the 4 LTX sync handlers — auto-swap handles readiness lazily. New endpoints: `POST /v1/ltx/{unload,reload}` mirroring existing `/v1/flux/{unload,reload}` pair, both Bearer-auth. LTX is **not** auto-reloaded after a Flux request — stays evicted until the next video request. Long-stretch image-only or video-only workloads have zero swap overhead; mixed workloads pay per direction change: LTX→Flux ≈ +3 s eviction, Flux→LTX ≈ +7–30 s cold load depending on OS page cache. Verified: cuda:0 at rest 789 MiB idle, training PID 1956281 unaffected throughout, Flux image request 45.8 s end-to-end with auto-evict, 76 unit tests passing.

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
