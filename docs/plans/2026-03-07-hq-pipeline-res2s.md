# HQ Pipeline / res2s Sampler Integration Plan

## Overview

LTX 2.3 provides `TI2VidTwoStagesHQPipeline` alongside the standard `TI2VidTwoStagesPipeline`. The HQ pipeline replaces the first-order Euler sampler with a second-order **res2s** (Runge-Kutta 2nd order with SDE noise) sampler, achieving comparable or better quality with fewer denoising steps. This plan covers integrating the res2s sampler into taco-backend as a new model option.

## What res2s Does Differently

### Euler (current, first-order)
- Single model evaluation per step
- `x_next = x + velocity * dt` (simple forward Euler)
- Standard pipeline uses **30 steps** (stage 1) via `LTX2Scheduler`
- Total neural network evaluations: **30** (stage 1) + **3** (stage 2) = **33**

### res2s (second-order, Runge-Kutta with SDE)
- **Two model evaluations per step**: once at current sigma, once at a midpoint (`c2 = 0.5`, so `sub_sigma = sqrt(sigma * sigma_next)`)
- Combines both evaluations using RK coefficients (`a21`, `b1`, `b2`) derived from phi functions
- Includes **SDE noise injection** at both substep and step level for variance-preserving transitions
- Has **bong iteration** (iterative anchor refinement) when step size is small (`h < 0.5`) and sigma > 0.03
- Uses `Res2sDiffusionStep` instead of `EulerDiffusionStep` as the stepper
- HQ pipeline defaults to **15 steps** (stage 1)
- Total neural network evaluations: **2 * 15 + 1** = **31** (stage 1; +1 for final denoise when sigma[-1]==0) + **2 * 3 + 1** = **7** (stage 2) = **38**

### Key Algorithmic Difference
The res2s loop (`res2s_audio_video_denoising_loop`) at each step:
1. Evaluates denoiser at current point -> `denoised_1`
2. Computes RK coefficients from phi functions (cached)
3. Computes midpoint `x_mid` using coefficient `a21` and `eps_1`
4. Injects SDE noise at midpoint via `Res2sDiffusionStep.step()`
5. Optionally runs bong iteration to refine anchor point
6. Evaluates denoiser at midpoint -> `denoised_2`
7. Combines both estimates: `x_next = x_anchor + h * (b1 * eps_1 + b2 * eps_2)`
8. Injects SDE noise at step level
9. Final step: if `sigmas[-1] == 0`, one extra denoise call to fully remove noise

The higher-order method captures curvature in the denoising trajectory, producing smoother results with fewer steps. The SDE noise prevents mode collapse.

## Step Count Comparison

| Pipeline | Stage 1 Steps | Stage 1 NFE | Stage 2 Steps | Stage 2 NFE | Total NFE |
|----------|--------------|-------------|---------------|-------------|-----------|
| Standard (Euler) | 30 | 30 | 3 | 3 | 33 |
| HQ (res2s) | 15 | 31 | 3 | 7 | 38 |

NFE = Neural Function Evaluations (transformer forward passes).

The HQ pipeline does ~15% more total transformer forward passes despite using half the step count. The benefit is better quality from the second-order integration, not fewer evaluations. However, in practice the res2s approach tends to produce better temporal coherence and reduced artifacts at fewer nominal steps.

## HQ-Specific Parameter Differences

From `LTX_2_3_HQ_PARAMS` vs `LTX_2_3_PARAMS`:

| Parameter | Standard | HQ |
|-----------|----------|-----|
| `num_inference_steps` | 30 | 15 |
| `video.stg_scale` | 1.0 | 0.0 |
| `video.stg_blocks` | [28] | [] |
| `video.rescale_scale` | 0.7 | 0.45 |
| `audio.stg_scale` | 1.0 | 0.0 |
| `audio.stg_blocks` | [28] | [] |
| `audio.rescale_scale` | 0.7 | 1.0 |

Key: HQ disables STG (Spatio-Temporal Guidance) entirely, which means **one fewer transformer forward pass per guided step** compared to standard. This partially offsets the 2x evaluations from res2s.

### Distilled LoRA in HQ
The HQ pipeline applies the **distilled LoRA in both stages** with different strengths:
- Stage 1: distilled LoRA at strength **0.25** (fused into dev checkpoint)
- Stage 2: distilled LoRA at strength **0.5** (fused into dev checkpoint)

This is different from the standard pipeline which:
- Stage 1: dev checkpoint with NO distilled LoRA
- Stage 2: dev checkpoint WITH distilled LoRA at strength 1.0

### HQ guider API difference
The HQ pipeline uses `MultiModalGuider` directly (non-factory), while the standard pipeline uses `MultiModalGuiderFactory`. The factory creates new guider instances per step based on sigma, while the HQ version uses a fixed guider. This is because HQ disables STG (stg_scale=0), making per-sigma guider construction unnecessary.

## Proposed Model Name

Expose as `"ltx-2-3-hq"` in the API. Users choose between:
- `ltx-2-3-fast` -- distilled, 8 steps, no CFG, fastest
- `ltx-2-3-pro` -- dev, 30 euler steps + CFG, current "pro" quality
- `ltx-2-3-hq` -- dev + distilled LoRA (0.25), 15 res2s steps + CFG, highest quality

## Checkpoints

Uses the **same checkpoints** as pro:
- `ltx-2.3-22b-dev.safetensors` (base transformer)
- `ltx-2.3-22b-distilled-lora-384.safetensors` (distilled LoRA, but at 0.25/0.5 strength instead of 1.0)
- `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` (upsampler)

No new checkpoint downloads needed.

## Config Changes

```python
# config.py -- no new paths needed, just document that HQ uses same checkpoints
# The HQ params are imported from ltx_pipelines constants
```

No changes to `config.py`. The model name `"ltx-2-3-hq"` will be recognized in `split_model_manager.py` and `server.py`.

## SplitModelManager Changes

### New transformer state: `"dev_lora_025"` and `"dev_lora_050"`

The HQ pipeline needs the distilled LoRA fused at non-default strengths. Since LoRA fusion is permanent (no unfuse), we need dedicated transformer states:

- `"dev_lora_025"` -- dev checkpoint + distilled LoRA at strength 0.25 (HQ stage 1)
- `"dev_lora_050"` -- dev checkpoint + distilled LoRA at strength 0.5 (HQ stage 2)

These get added to `DenoiserWorker.ensure_transformer()`.

### New method: `_run_t2v_hq` (or flag on `_run_t2v`)

**Recommended approach: new private method `_run_t2v_hq`** rather than adding flags to `_run_t2v`.

Rationale: The HQ path differs in several structural ways:
1. Different sampler loop function (`res2s_audio_video_denoising_loop` vs `euler_denoising_loop`)
2. Different stepper class (`Res2sDiffusionStep` vs `EulerDiffusionStep`)
3. Different LoRA strengths per stage (0.25 then 0.5 vs none then 1.0)
4. Different guider params (no STG)
5. HQ uses `multi_modal_guider_denoising_func` directly (non-factory)
6. Scheduler needs `latent` parameter for token-count-dependent shifting

Adding these as conditionals inside `_run_t2v` would make the already-long method harder to follow. A separate method shares the same overall structure but with HQ-specific wiring.

### Implementation outline for `_run_t2v_hq`:

```python
@torch.inference_mode()
def _run_t2v_hq(
    self, worker: DenoiserWorker, prompt: str, width: int, height: int,
    num_frames: int, fps: float, seed: int, generate_audio: bool,
    on_progress=None,
) -> bytes:
    device = worker.device
    dtype = torch.bfloat16

    # Swap to dev+distilled_lora@0.25 for stage 1
    worker.ensure_transformer("dev_lora_025")

    # Text encoding (shared encoder, same as pro)
    ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
    ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
    v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
    v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = Res2sDiffusionStep()  # <-- KEY CHANGE

    # Stage 1: half-res, 15 res2s steps
    hq_params = LTX_2_3_HQ_PARAMS
    stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
    video_encoder = worker.ledger.video_encoder()
    stage_1_cond = combined_image_conditionings(...)

    transformer = worker.ledger.transformer()

    # Scheduler with latent-aware shifting
    empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
    sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=hq_params.num_inference_steps).to(...)

    def denoising_loop(sigmas, video_state, audio_state, stepper):
        dfn = multi_modal_guider_denoising_func(
            video_guider=MultiModalGuider(params=hq_params.video_guider_params, negative_context=v_context_n),
            audio_guider=MultiModalGuider(params=hq_params.audio_guider_params, negative_context=a_context_n),
            v_context=v_context_p, a_context=a_context_p, transformer=transformer,
        )
        if on_progress:
            dfn = self._wrap_denoise(dfn, on_progress, ...)
        return res2s_audio_video_denoising_loop(  # <-- KEY CHANGE
            sigmas=sigmas, video_state=video_state, audio_state=audio_state,
            stepper=stepper, denoise_fn=dfn,
        )

    video_state, audio_state = denoise_audio_video(...)

    # Stage 2: swap to dev_lora@0.5, use res2s for stage 2 as well
    del transformer, ...
    cleanup_memory()
    worker.ensure_transformer("dev_lora_050")
    transformer = worker.ledger.transformer()
    distilled_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=device)

    def stage2_loop(sigmas, video_state, audio_state, stepper):
        dfn = simple_denoising_func(...)
        return res2s_audio_video_denoising_loop(...)

    # ... decode, return bytes
```

### New imports needed in split_model_manager.py:

```python
from ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ltx_core.components.guiders import MultiModalGuider
from ltx_core.tools import VideoLatentShape
from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS
from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop
```

### Async wrapper

Add `generate_text_to_video` dispatch for `model == "ltx-2-3-hq"` to call `_run_t2v_hq`. Simplest approach: check model in the existing `generate_text_to_video` method:

```python
async def generate_text_to_video(self, ...):
    worker = await self._acquire_worker()
    try:
        loop = asyncio.get_running_loop()
        if model == "ltx-2-3-hq":
            return await loop.run_in_executor(None, self._run_t2v_hq, worker, ...)
        else:
            return await loop.run_in_executor(None, self._run_t2v, worker, ...)
    finally:
        worker.lock.release()
```

### Transformer swap sequence for HQ

1. Start: `dev_lora_025` (stage 1)
2. After stage 1: swap to `dev_lora_050` (stage 2)
3. After stage 2: swap back to `dev` (restore default state for next request)

This is 2 swaps, same as pro. Each swap takes ~5-8s (load from disk).

### Optimization: combine lora states

Since `dev_lora_025` and `dev_lora_050` both use the same dev checkpoint + same LoRA file at different strengths, we could consider caching the base dev transformer and applying LoRA on-the-fly. However, LoRA fusion is permanent in the current ModelLedger implementation (no unfuse). So we must do full reloads.

## Progress Reporting

The res2s loop calls `denoise_fn` **twice per step** (once at current sigma, once at midpoint). Our `_wrap_denoise` progress wrapper counts denoise_fn calls, not nominal steps.

For HQ stage 1 with 15 steps:
- `denoise_fn` called 2 * 15 + 1 = 31 times (including final denoise)
- Progress should track these 31 calls, not 15 "steps"

The `_wrap_denoise` approach still works -- it counts function invocations. Just set `total_steps` to `2 * hq_params.num_inference_steps + 1` for accurate progress bars.

For stage 2 with 3 distilled sigmas (2 steps + final):
- `denoise_fn` called 2 * 2 + 1 = 5 times

## Risk Assessment

### Stability: MEDIUM RISK
- The res2s sampler is more complex (RK coefficients, SDE noise, bong iteration) than Euler
- It uses `torch.float64` internally for precision in coefficient computation, then casts back to bf16
- The bong iteration loop runs up to 100 iterations (fixed-point convergence), but these are cheap (no neural network calls)
- SDE noise injection uses separate generators for step vs substep noise

### Compatibility with progress wrapping: LOW RISK
- `_wrap_denoise` wraps `denoise_fn` which is called independently of the loop structure
- The res2s loop calls `denoise_fn` the same way as Euler (same signature)
- Just need correct `total_steps` count (2N+1 instead of N)

### Memory: LOW RISK
- Same transformer size as pro (dev checkpoint)
- res2s uses some extra tensors (anchor points, midpoints in float64) but these are small compared to transformer weights
- Two `torch.Generator` instances instead of one (step + substep noise)
- Bong iteration doesn't allocate new tensors

### VRAM peak: MEDIUM CONCERN
- Stage 1 uses `dev_lora_025` which is a full dev checkpoint + LoRA fusion -- same size as `dev` or `dev_lora`
- The `VideoLatentShape.from_pixel_shape()` call for scheduler needs the shape tensor on device
- No additional models loaded compared to pro

### Transformer swap latency: same as pro
- 2 swaps (dev_lora_025 -> dev_lora_050 -> dev), same pattern as pro (dev -> dev_lora -> dev)

## Testing Strategy

### 1. Smoke test: basic generation
```bash
# Test t2v-hq endpoint
curl -X POST http://localhost:8090/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking on a beach", "model": "ltx-2-3-hq", "width": 768, "height": 512, "num_frames": 41, "seed": 42}' \
  --output test_hq.mp4
```

### 2. Quality comparison
Generate the same prompt/seed with `ltx-2-3-pro` and `ltx-2-3-hq`, compare:
- Visual quality (temporal coherence, detail)
- Audio quality
- Generation time

### 3. Progress reporting
Verify SSE progress events increment smoothly from 0 to ~0.99 during HQ generation.

### 4. Transformer swap verification
After HQ generation completes, verify worker state is `"dev"` (restored for next request). Then run a `ltx-2-3-fast` or `ltx-2-3-pro` request to confirm no state corruption.

### 5. Memory leak check
Run 3 consecutive HQ generations, monitor GPU memory via `nvidia-smi`. Ensure no monotonic increase.

### 6. Edge cases
- HQ with image conditioning (`_run_i2v_hq` -- future work, not in initial scope)
- HQ with audio conditioning -- not supported initially, same as standard HQ pipeline
- Very long videos (121 frames) -- check for OOM

## Implementation Order

1. Add `"dev_lora_025"` and `"dev_lora_050"` states to `DenoiserWorker.ensure_transformer()`
2. Add new imports to `split_model_manager.py`
3. Implement `_run_t2v_hq` method
4. Update `generate_text_to_video` dispatch
5. Update `server.py` to accept `"ltx-2-3-hq"` model name
6. Smoke test
7. Optionally implement `_run_i2v_hq` for image-to-video with res2s

## Future Considerations

- The `gradient_estimating_euler_denoising_loop` in samplers.py is another alternative (GE-Euler, first-order with velocity tracking). Could be a middle ground between Euler and res2s.
- If LoRA unfusion becomes available in ltx-core, we could avoid separate transformer states for different LoRA strengths.
- The HQ pipeline's `MultiModalGuider` (non-factory) vs standard's `MultiModalGuiderFactory` -- the factory is more flexible for per-sigma guidance schedules but HQ doesn't need it since STG is disabled.
