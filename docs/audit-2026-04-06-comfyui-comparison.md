# Codebase Audit: taco-backend vs ComfyUI/Reference (2026-04-06)

5-agent swarm audit comparing our LTX pipeline, Flux pipeline, VAE decode, Python dependencies, and ComfyUI integration against upstream reference code and community best practices.

## Executive Summary

**Our implementation is well-aligned with the LTX-2.3 reference.** The main divergences are intentional quality improvements (5-step stage 2, token-adapted sigma schedule, cosine S-curve blending). Dependencies need attention (torch version mismatch, uncommitted LTX-2 changes). No security issues found.

---

## 1. LTX Pipeline Comparison

### What matches reference

- Guider params: cfg=3.0, stg=1.0, rescale=0.7, stg_blocks=[28] — exact match with `LTX_2_3_PARAMS`
- Stage 1 step counts: 30 (pro), 15 (hq), 8 (fast) — match
- Stage 2 denoising: `simple_denoising_func` — match
- LoRA states: dev_lora@1.0 (pro stage 2), dev_lora_025/dev_lora_050 (hq) — match
- Retake guider: `MultiModalGuider(...)` directly — match (not factory)

### Intentional divergences

| Our code | Reference | Reason |
|----------|-----------|--------|
| Stage 2: 5 steps (`_STAGE_2_SIGMAS`) | 3 steps (`STAGE_2_DISTILLED_SIGMA_VALUES`) | Better motion resolution from 91% noise |
| `LTX2Scheduler(latent=empty_latent)` | No `latent=` arg | Token-adapted sigma shift — we're more correct |
| Fast a2v with distilled transformer | No reference a2v fast mode | Custom, verified architecture supports it |
| Fast t2v split-schedule (4+4 model swap) | No reference | Custom audio quality fix, unvalidated |

### Missing upstream features

- **Retake distilled mode**: Upstream `retake.py` supports `distilled=True` for fast retake. We always use dev+CFG.
- **IC-LoRA pipeline**: Upstream has `ICLoraPipeline` with attention masking. Not implemented.
- **Keyframe interpolation pipeline**: Upstream has `KeyframeInterpolationPipeline`. Not implemented.
- **Retake negative seed handling**: Upstream treats `seed < 0` as random. We always pass positive seeds.
- **Retake enhance_prompt**: Not passed through (supported upstream).

### Local patches

- **Cosine S-curve tiling** (`8c3ced4`): Patched into local LTX-2 repo at `ltx_core/model/video_vae/tiling.py`. NOT upstream. Will be lost on `git pull`.

---

## 2. VAE Decode Comparison

| Setting | taco-backend | ComfyUI | LTX Reference |
|---------|-------------|---------|---------------|
| Spatial tiling | Disabled | 4 tiles/axis, linear blend | 512px tiles, 64px overlap, cosine blend |
| Temporal tile size | 128 frames | 128 frames (16 latents) | 64 frames (8 latents) |
| Temporal overlap | 32 frames | 8 frames (clamped to 16) | 24 frames |
| Short video bypass | ≤257 frames (no tiling) | Never | Never |
| Blend function | Cosine S-curve (ltx-core) | Linear ramp (custom) | Cosine S-curve (ltx-core) |
| Decoder dtype | bfloat16 | auto (inherits) | bfloat16 |
| `last_frame_fix` | Not implemented | Optional (default False) | Not implemented |

### Key findings

- **Our temporal overlap (32 frames) is the best of all three** — more overlap = smoother blending
- **ComfyUI uses LINEAR blending** while we use cosine S-curve — ours is theoretically better
- **ComfyUI's temporal_overlap=1 latent is below ltx-core minimum** (16 frames) — silently clamped
- **257-frame bypass is well-justified** for 96GB GPUs after transformer eviction (~90GB free)
- **No community-improved VAE weights exist** — stock LTX-2.3 checkpoint is the best available
- **`_decode_video_fp32` is a misleading function name** — does no fp32 work, pure passthrough

### ComfyUI `last_frame_fix`

Appends a copy of the last latent frame before decoding, then trims output. Addresses causal conv end-of-sequence artifacts where final frames have less context. Worth examining if end-of-video quality is a concern.

---

## 3. Flux Pipeline Status

### Current state

- **Dev**: Flux2Pipeline + fused Turbo LoRA + FP8 layerwise casting (~77GB)
- **Klein KV**: Flux2KleinKVPipeline from single-file checkpoint + FP8 with boundary layer exclusion (~18GB)
- **Model swapping**: on-demand evict/reload between Dev and Klein
- **Klein boundary layers**: x_embedder, context_embedder, proj_out kept in BF16

### Dependencies

- Requires `diffusers` from git main (Flux2KleinKVPipeline not in stable 0.37.1)
- Klein loaded via `from_single_file` with local config path (repo shards are gated)

---

## 4. Python Dependencies

### Critical issues

| Issue | Severity | Detail |
|-------|----------|--------|
| **Torch version mismatch** | CRITICAL | pyproject.toml pins `>=2.9,<2.10`, installed is `2.10.0+cu128` |
| **CUDA version mismatch** | HIGH | pyproject uses `pytorch-cu130` index, installed is cu128 |
| **LTX-Core declares torch~=2.7** | INFO | Editable install bypasses check, but formally incompatible |
| **LTX-2 uncommitted local changes** | HIGH | `media_io.py` (CRF 18 patch), `encoder_configurator.py` |

### Package versions

| Package | Installed | Latest | Status |
|---------|-----------|--------|--------|
| torch | 2.10.0+cu128 | 2.10.0+cu130 | Mismatched CUDA |
| diffusers | 0.38.0.dev0 | 0.37.1 (stable) | On dev branch |
| transformers | 5.3.0 | 5.3.0 | Current |
| fastapi | 0.128.0 | 0.135.2 | Slightly behind |
| uvicorn | 0.40.0 | 0.41.0+ | Slightly behind |
| nvidia-cudnn-cu13 | 9.20.0.48 | 9.20.0 | Current |
| nvidia-cublas | 13.3.0.5 | 13.3.0.5 | Current |
| safetensors | 0.7.0 | 0.7.0 | Current |
| peft | 0.18.1 | 0.18.1 | Current |

### Security

- No CVEs affecting our stack
- Safetensors default format prevents pickle RCE (CVE-2025-32434)
- FastAPI/uvicorn versions are patched against known vulnerabilities

---

## 5. Recommended Actions

### Priority 1 — Housekeeping (do now)

1. **Rename `_decode_video_fp32` → `_decode_video`** — misleading name, does no fp32 work
2. **Commit LTX-2 local changes** — CRF 18 patch in media_io.py and encoder_configurator.py changes
3. **Document cosine S-curve patch** as tracked local modification (not upstream)
4. **Fix `test_job_queue.py` docstring** — says "both queues" but there's only one

### Priority 2 — Dependency resolution (soon)

5. **Resolve torch 2.9 vs 2.10** — decide version, update pyproject.toml to match installed
6. **Resolve cu128 vs cu130** — document CUDA version choice
7. **Consider pinning diffusers to 0.37.1** stable if Klein KV pipeline is available there, otherwise document why git main is required
8. **Add explicit `peft` dependency** to pyproject.toml if actively used

### Priority 3 — Features (future, defer)

9. **Retake distilled mode** — fast retake using distilled transformer
10. **ComfyUI `last_frame_fix`** — end-of-video artifact mitigation
11. **IC-LoRA pipeline** — motion anchoring for temporal coherence
12. **Spatial tiling** — re-evaluate if we ever support >4K resolution

### No changes needed

- VAE tiling config is optimal for our hardware (best overlap, best blending)
- Pipeline params match reference (intentional divergences documented)
- nvidia packages are current
- No security patches needed

---

## Appendix: File Locations

| What | Path |
|------|------|
| Our LTX pipeline | `split_model_manager.py` |
| Our Flux pipeline | `flux_manager.py` |
| LTX reference t2v | `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py` |
| LTX reference HQ | `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py` |
| LTX reference a2v | `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py` |
| LTX reference retake | `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/retake.py` |
| LTX constants | `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py` |
| VAE tiling code | `/mnt/nvme-1/repos/LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/tiling.py` |
| Local cosine patch | `/mnt/nvme-1/repos/LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/tiling.py` (commit 8c3ced4) |
