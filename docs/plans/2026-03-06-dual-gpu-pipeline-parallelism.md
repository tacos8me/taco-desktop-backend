# Dual-GPU Pipeline Parallelism Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split LTX pipeline execution across two GPUs — text encoding on GPU:0, denoising on GPU:1 — to eliminate model loading churn and keep both GPUs utilized.

**Architecture:** GPU:0 holds Gemma 12B text encoder + VAE encoders permanently. GPU:1 holds the 22B transformer (swappable between dev/distilled checkpoints) + spatial upsampler + decoders permanently. Text embeddings (~8MB) transfer once per generation via `.to()`.

**Tech Stack:** Python 3.12, PyTorch 2.10+cu130, ltx-pipelines, ltx-core, FastAPI

---

## Research Summary

### Current Architecture (problems)
- Each GPU loads ALL 4 pipeline types independently with duplicate models
- ModelLedger uses DummyRegistry — no weight caching, models rebuilt from disk every call
- Video encoder loaded 3× per generation, transformer loaded 2× (different LoRAs)
- Pro model: ~270s denoising + ~10-15s model loading overhead per generation
- When one GPU is generating, the other sits idle

### Proposed Architecture
```
GPU:0 (Encoder Hub, ~28GB of 96GB):
  - Gemma 3 12B text encoder (permanent, 24GB)
  - Embeddings processor (permanent, ~0.1GB)
  - Video VAE encoder (permanent, ~0.16GB)
  - Audio VAE encoder (permanent, ~0.05GB)

GPU:1 (Denoiser Hub, ~49GB of 96GB):
  - LTX-2.3 22B transformer (swappable: dev/distilled/dev+LoRA, 44GB)
  - Video VAE encoder (duplicate for per_channel_statistics + stage 2 conditioning, ~0.16GB)
  - Spatial upsampler (permanent, 0.24GB)
  - Video VAE decoder (permanent, ~0.16GB)
  - Audio VAE decoder (permanent, ~0.05GB)
  - Vocoder (permanent, ~0.04GB)
```

### Transformer Swap Strategy

Three transformer variants are needed:
1. **Dev** — pro stage 1, retake, a2v stage 1 (`ltx-2.3-22b-dev.safetensors`)
2. **Distilled** — fast model stages 1+2 (`ltx-2.3-22b-distilled.safetensors`)
3. **Dev + distilled LoRA** — pro stage 2, a2v stage 2 (`dev + ltx-2.3-22b-distilled-lora-384.safetensors`)

Cannot fit all simultaneously (3 × 44GB = 132GB > 96GB). Strategy:
- Track current transformer state: `"dev"`, `"distilled"`, or `"dev_lora"`
- Swap only when needed — consecutive same-model requests avoid swap
- Swap cost: ~3-5s from NVMe (vs ~270s denoising for pro, ~10s for fast)
- Use load-then-swap pattern for error safety

### Validated Properties
- All 4 pipelines (t2v, i2v, a2v, retake) have clean split points
- Only ~8MB of embeddings cross the GPU boundary, once per generation
- No bidirectional communication during denoising
- Retake is single-stage, fully sequential, compatible
- LoRA fusion is permanent (no unfuse) — different LoRA combos need full reload
- Local code matches upstream Lightricks/LTX-2 main (as of 2026-03-05)
- Safetensors loading is selective per component (not full 43GB checkpoint)

### VRAM Budget
| GPU | Weights | Activations | Total | Headroom |
|-----|---------|-------------|-------|----------|
| GPU:0 | 24.3 GB | 3-4 GB | ~28 GB | 68 GB |
| GPU:1 | 44.7 GB | 3-5 GB | ~50 GB | 46 GB |

### Key API Corrections (from code review)
- ModelLedger method: `gemma_embeddings_processor()` (NOT `embeddings_processor()`)
- Embeddings processor method: `process_hidden_states(hidden_states, attention_mask)`
- `encode_prompts()` in helpers.py loads text_encoder, encodes, deletes, then loads embeddings_processor, processes, deletes
- `upsample_video()` requires `video_encoder.per_channel_statistics` on same device as latent
- Stage 2 of pro/a2v re-encodes image conditionings at full resolution (needs video_encoder)
- A2V uses `denoise_video_only()` (not `denoise_audio_video()`) and returns original audio
- Retake has TemporalRegionMask conditioning, encodes input video+audio, single-stage
- All denoising requires `PipelineComponents` object (holds patchifiers, device, dtype)
- `torch.Generator` must be created on denoiser device
- `TilingConfig` required for VAE decode

---

## Task 1: Create SplitModelManager skeleton with dual-GPU hubs

**Files:**
- Create: `split_model_manager.py`
- Test: `tests/test_split_model_manager.py`

**Context:** Replace the current `PipelineManager` which loads full pipeline objects per GPU.

**Key design decisions:**
- `EncoderHub` on GPU:0: text_encoder, embeddings_processor, video_encoder, audio_encoder
- `DenoiserHub` on GPU:1: transformer (swappable), video_encoder (duplicate), spatial_upsampler, video_decoder, audio_decoder, vocoder
- Video encoder on BOTH GPUs — GPU:0 for encoding inputs, GPU:1 for per_channel_statistics + stage 2 conditioning
- Single `asyncio.Lock` on denoiser hub (text encoding happens inside the locked section since generation is sequential anyway)
- Track `_transformer_state: str` to know which checkpoint is loaded (`"dev"`, `"distilled"`, `"dev_lora"`)
- `is_ready` property for server readiness checks (replaces `manager.workers`)

**Step 1: Write the SplitModelManager skeleton**

```python
"""Split-GPU model manager for LTX-2 video generation.

Keeps text encoder + VAE encoders on GPU:0 (encoder_device),
transformer + decoders on GPU:1 (denoiser_device). Models stay
resident — no per-request loading/unloading except transformer
checkpoint swaps between dev/distilled variants.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import torch
from ltx_pipelines.utils.model_ledger import ModelLedger

import config

logger = logging.getLogger(__name__)


@dataclass
class EncoderHub:
    device: torch.device
    text_encoder: object
    embeddings_processor: object
    video_encoder: object
    audio_encoder: object


@dataclass
class DenoiserHub:
    device: torch.device
    transformer: object
    video_encoder: object  # duplicate for per_channel_statistics + stage 2 conditioning
    spatial_upsampler: object
    video_decoder: object
    audio_decoder: object
    vocoder: object
    lock: asyncio.Lock


class SplitModelManager:
    def __init__(self) -> None:
        self.encoder_hub: EncoderHub | None = None
        self.denoiser_hub: DenoiserHub | None = None
        self._transformer_state: str = ""  # "dev", "distilled", or "dev_lora"

    @property
    def is_ready(self) -> bool:
        return self.encoder_hub is not None and self.denoiser_hub is not None

    def load_all(self) -> None:
        encoder_device = torch.device(config.GPU_DEVICES[0])
        denoiser_device = torch.device(config.GPU_DEVICES[1])

        # Encoder hub — load from dev checkpoint (text encoder + VAE encoders are the same across checkpoints)
        enc_ledger = ModelLedger(
            dtype=torch.bfloat16, device=encoder_device,
            checkpoint_path=config.DEV_CHECKPOINT,
            gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
            loras=(),
        )
        logger.info("Loading encoder hub on %s ...", encoder_device)
        self.encoder_hub = EncoderHub(
            device=encoder_device,
            text_encoder=enc_ledger.text_encoder(),
            embeddings_processor=enc_ledger.gemma_embeddings_processor(),
            video_encoder=enc_ledger.video_encoder(),
            audio_encoder=enc_ledger.audio_encoder(),
        )

        # Denoiser hub — load dev transformer by default
        den_ledger = ModelLedger(
            dtype=torch.bfloat16, device=denoiser_device,
            checkpoint_path=config.DEV_CHECKPOINT,
            gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
            loras=(),
        )
        logger.info("Loading denoiser hub on %s ...", denoiser_device)
        self.denoiser_hub = DenoiserHub(
            device=denoiser_device,
            transformer=den_ledger.transformer(),
            video_encoder=den_ledger.video_encoder(),
            spatial_upsampler=den_ledger.spatial_upsampler(),
            video_decoder=den_ledger.video_decoder(),
            audio_decoder=den_ledger.audio_decoder(),
            vocoder=den_ledger.vocoder(),
            lock=asyncio.Lock(),
        )
        self._transformer_state = "dev"
        logger.info("All models loaded.")
```

**Step 2: Add transformer swap methods with load-then-swap safety**

```python
def _ensure_transformer(self, state: str) -> None:
    """Swap transformer checkpoint if needed. Must hold denoiser lock."""
    if self._transformer_state == state:
        return

    logger.info("Swapping transformer: %s -> %s", self._transformer_state, state)
    hub = self.denoiser_hub

    # Determine checkpoint and loras
    if state == "dev":
        checkpoint, loras = config.DEV_CHECKPOINT, ()
    elif state == "distilled":
        checkpoint, loras = config.DISTILLED_CHECKPOINT, ()
    elif state == "dev_lora":
        distilled_lora = LoraPathStrengthAndSDOps(
            path=config.DISTILLED_LORA, strength=1.0,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
        checkpoint, loras = config.DEV_CHECKPOINT, (distilled_lora,)
    else:
        raise ValueError(f"Unknown transformer state: {state}")

    # Load new transformer BEFORE deleting old one (error safety)
    ledger = ModelLedger(
        dtype=torch.bfloat16, device=hub.device,
        checkpoint_path=checkpoint, gemma_root_path=config.GEMMA_ROOT,
        spatial_upsampler_path=config.SPATIAL_UPSAMPLER, loras=loras,
    )
    new_transformer = ledger.transformer()

    # Swap
    old = hub.transformer
    hub.transformer = new_transformer
    self._transformer_state = state
    del old
    torch.cuda.synchronize(hub.device)
    torch.cuda.empty_cache()
    logger.info("Transformer swapped to %s", state)
```

Note: load-then-swap means ~88GB peak during swap (old 44GB + new 44GB). This fits in 96GB with ~8GB margin. If too tight, fall back to delete-first pattern with try/except recovery.

**Step 3: Write test**

```python
def test_split_model_manager_imports():
    from split_model_manager import SplitModelManager
    mgr = SplitModelManager()
    assert not mgr.is_ready
    assert mgr._transformer_state == ""
```

**Step 4: Run and commit**

```bash
uv run --no-sync pytest tests/test_split_model_manager.py -v
git add split_model_manager.py tests/test_split_model_manager.py
git commit -m "feat: add SplitModelManager skeleton with dual-GPU hubs"
```

---

## Task 2: Implement text encoding and cross-GPU transfer

**Files:**
- Modify: `split_model_manager.py`

**Context:** Extract text encoding logic from helpers.py `encode_prompts()` (lines 48-86). Use pre-loaded text_encoder and embeddings_processor instead of loading/deleting each time.

**Key reference:** Study `encode_prompts()` in `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`. It calls:
1. `model_ledger.text_encoder()` → `text_encoder.encode(prompt)` → returns `(hidden_states: tuple[Tensor, ...], attention_mask: Tensor)` → `del text_encoder`
2. `model_ledger.gemma_embeddings_processor()` → `embeddings_processor.process_hidden_states(hidden_states, attention_mask)` → returns `EmbeddingsProcessorOutput(video_encoding, audio_encoding, attention_mask)` → `del embeddings_processor`

Our version skips the load/delete since models are permanent.

**Step 1: Add encode_text method**

```python
def encode_text(self, prompts: list[str]) -> list[EmbeddingsProcessorOutput]:
    hub = self.encoder_hub
    results = []
    for prompt in prompts:
        hidden_states, attention_mask = hub.text_encoder.encode(prompt)
        result = hub.embeddings_processor.process_hidden_states(hidden_states, attention_mask)
        results.append(result)
    return results
```

**Step 2: Add transfer helper**

```python
def _to_denoiser(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(self.denoiser_hub.device)

def _transfer_contexts(self, contexts: list[EmbeddingsProcessorOutput]) -> list[EmbeddingsProcessorOutput]:
    target = self.denoiser_hub.device
    return [
        EmbeddingsProcessorOutput(
            video_encoding=ctx.video_encoding.to(target),
            audio_encoding=ctx.audio_encoding.to(target) if ctx.audio_encoding is not None else None,
            attention_mask=ctx.attention_mask.to(target),
        )
        for ctx in contexts
    ]
```

**Step 3: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: add text encoding and cross-GPU context transfer"
```

---

## Task 3: Implement image/audio conditioning + latent state init

**Files:**
- Modify: `split_model_manager.py`

**Context:** Image conditioning (i2v), audio encoding (a2v), and video encoding (retake) all use VAE encoders on GPU:0. Encoded tensors transfer to GPU:1. Stage 2 of pro/a2v re-encodes images at full resolution using the video_encoder DUPLICATE on GPU:1.

**Key references:**
- `combined_image_conditionings()` in helpers.py lines 89-123
- `image_conditionings_by_replacing_latent()` in helpers.py lines 126-153
- `vae_encode_audio()` for A2V audio encoding
- `_encode_video_for_retake()` in retake.py
- `PipelineComponents` in ltx_core types.py (holds device, dtype, patchifiers)
- `torch.Generator(device=denoiser_device)` for noise creation

**Step 1: Add image conditioning on encoder hub (GPU:0)**

Encodes conditioning images, transfers results to GPU:1:
```python
def encode_image_conditionings(self, images, height, width, num_frames):
    # Use self.encoder_hub.video_encoder on GPU:0
    # Transfer conditioning items to GPU:1
    ...
```

**Step 2: Add stage 2 image conditioning on denoiser hub (GPU:1)**

Re-encodes at full resolution using the DUPLICATE video_encoder on GPU:1:
```python
def encode_stage2_image_conditionings(self, images, height, width, num_frames):
    # Use self.denoiser_hub.video_encoder on GPU:1
    ...
```

**Step 3: Add audio encoding for A2V on GPU:0**

```python
def encode_audio(self, audio_path, duration):
    # decode_audio_from_file() + vae_encode_audio() with encoder_hub.audio_encoder
    # Transfer encoded_audio_latent to GPU:1
    ...
```

**Step 4: Add video encoding for retake on GPU:0**

```python
def encode_video_for_retake(self, video_path, ...):
    # Use encoder_hub.video_encoder to encode input video
    # Apply TemporalRegionMask conditioning
    # Transfer initial_video_latent to GPU:1
    ...
```

**Step 5: Create PipelineComponents + torch.Generator on denoiser device**

```python
# In load_all() or as needed:
self._generator = torch.Generator(device=denoiser_device)
# PipelineComponents created with denoiser_device
```

**Step 6: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: add image/audio/video conditioning on encoder hub with GPU transfer"
```

---

## Task 4: Implement denoising for all pipeline types

**Files:**
- Modify: `split_model_manager.py`

**Context:** Four distinct denoising paths:

1. **Fast (distilled):** `simple_denoising_func` + `DISTILLED_SIGMA_VALUES` (8 steps) + `STAGE_2_DISTILLED_SIGMA_VALUES` (4 steps). Single context, no CFG. Uses distilled transformer for BOTH stages.

2. **Pro (ti2vid two-stage):** Stage 1: `multi_modal_guider_factory_denoising_func` + `LTX2Scheduler(steps=N)` (30 steps). Stage 2: `simple_denoising_func` + `STAGE_2_DISTILLED_SIGMA_VALUES` (3 steps). Needs dev transformer for stage 1, dev+LoRA transformer for stage 2.

3. **A2V:** Similar to pro but uses `denoise_video_only()` (audio frozen in stage 1, denoised in stage 2). Returns ORIGINAL audio (no VAE decode).

4. **Retake:** Single-stage. Uses `multi_modal_guider_denoising_func` (not factory variant). TemporalRegionMask conditioning. No spatial upsampler.

**Key references:**
- `denoise_audio_video()` in samplers.py — standard euler denoising loop
- `denoise_video_only()` — A2V variant (audio frozen)
- `simple_denoising_func()` in helpers.py lines 315-328
- `multi_modal_guider_factory_denoising_func()` in helpers.py lines 468-502
- `multi_modal_guider_denoising_func()` — retake variant
- `upsample_video()` — needs video_encoder.per_channel_statistics (use GPU:1 duplicate)
- `TilingConfig.default()` — pass to VAE decode

**Step 1: Implement fast model denoising**

```python
@torch.inference_mode()
def _run_fast(self, prompt, width, height, num_frames, fps, seed, generate_audio, images):
    self._ensure_transformer("distilled")
    # 1. encode_text([prompt]) on GPU:0, transfer to GPU:1
    # 2. encode image conditionings if any on GPU:0, transfer
    # 3. create noise with Generator(device=denoiser_device).manual_seed(seed)
    # 4. stage 1: simple_denoising_func + DISTILLED_SIGMA_VALUES (8 steps)
    # 5. upsample_video using denoiser_hub.video_encoder.per_channel_statistics
    # 6. re-encode images at full res using denoiser_hub.video_encoder
    # 7. stage 2: simple_denoising_func + STAGE_2_DISTILLED_SIGMA_VALUES (4 steps)
    # 8. vae_decode_video + vae_decode_audio with TilingConfig
    # 9. encode to MP4 bytes
    ...
```

**Step 2: Implement pro model denoising**

```python
@torch.inference_mode()
def _run_pro(self, prompt, width, height, num_frames, fps, seed, generate_audio, images):
    self._ensure_transformer("dev")
    params = detect_params(config.DEV_CHECKPOINT)
    # 1. encode_text([prompt, DEFAULT_NEGATIVE_PROMPT]) on GPU:0, transfer
    # 2. encode image conditionings on GPU:0, transfer
    # 3. create noise on GPU:1
    # 4. stage 1: multi_modal_guider_factory_denoising_func + LTX2Scheduler(steps=params.num_inference_steps)
    # 5. upsample_video using denoiser_hub.video_encoder.per_channel_statistics
    # 6. re-encode images at full res using denoiser_hub.video_encoder
    # 7. _ensure_transformer("dev_lora")
    # 8. stage 2: simple_denoising_func + STAGE_2_DISTILLED_SIGMA_VALUES
    #    Pass noise_scale=distilled_sigmas[0], initial_video_latent, initial_audio_latent
    # 9. _ensure_transformer("dev")  # restore for next request
    # 10. vae_decode + encode MP4
    ...
```

**Step 3: Implement A2V denoising**

```python
@torch.inference_mode()
def _run_a2v(self, prompt, audio_path, image_path, width, height, num_frames, fps, seed):
    self._ensure_transformer("dev")
    # 1. encode_text([prompt, DEFAULT_NEGATIVE_PROMPT]) on GPU:0, transfer
    # 2. encode_audio on GPU:0, transfer encoded_audio_latent
    # 3. optionally encode image conditioning on GPU:0, transfer
    # 4. stage 1: denoise_video_only() — audio latent frozen (denoise_mask=0)
    #    Uses multi_modal_guider, but audio guider has empty params
    # 5. upsample + re-encode images
    # 6. _ensure_transformer("dev_lora")
    # 7. stage 2: denoise_video_only() with simple_denoising_func
    # 8. _ensure_transformer("dev")
    # 9. vae_decode_video (but NOT audio — return original decoded_audio waveform)
    # 10. encode to MP4 with original audio
    ...
```

**Step 4: Implement retake denoising**

```python
@torch.inference_mode()
def _run_retake(self, video_path, start_time, duration, mode, prompt, seed):
    self._ensure_transformer("dev")  # retake uses dev checkpoint
    # 1. get_videostream_metadata(video_path) for fps, num_frames, width, height
    # 2. encode_video_for_retake on GPU:0 → initial_video_latent, transfer
    # 3. encode audio from video on GPU:0 → initial_audio_latent, transfer
    # 4. encode_text([prompt, negative_prompt]) on GPU:0, transfer
    # 5. Apply TemporalRegionMask to video_state and audio_state
    # 6. Set regenerate_video/regenerate_audio bools from mode:
    #    "replace_audio_and_video" → both True
    #    "replace_video" / "replace_video_only" → video=True, audio=False
    #    "replace_audio" → video=False, audio=True
    # 7. Single-stage denoising: multi_modal_guider_denoising_func (NOT factory variant)
    #    No spatial upsampler. No LoRA swap.
    # 8. vae_decode_video + vae_decode_audio
    # 9. encode to MP4
    ...
```

**Step 5: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: implement all denoising paths (fast/pro/a2v/retake)"
```

---

## Task 5: Wire up async API and server integration

**Files:**
- Modify: `split_model_manager.py`
- Modify: `server.py`
- Modify: `config.py`

**Context:** Expose the same async API as current PipelineManager so server.py endpoints work with minimal changes.

**Step 1: Add async wrapper methods**

```python
async def generate_text_to_video(self, prompt, model, width, height, num_frames, fps, seed, generate_audio=True):
    async with self.denoiser_hub.lock:
        loop = asyncio.get_running_loop()
        if model == "ltx-2-3-fast":
            return await loop.run_in_executor(None, self._run_fast, prompt, width, height, num_frames, fps, seed, generate_audio, [])
        else:
            return await loop.run_in_executor(None, self._run_pro, prompt, width, height, num_frames, fps, seed, generate_audio, [])

async def generate_image_to_video(self, prompt, image_path, model, width, height, num_frames, fps, seed, generate_audio=True):
    images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]
    async with self.denoiser_hub.lock:
        loop = asyncio.get_running_loop()
        if model == "ltx-2-3-fast":
            return await loop.run_in_executor(None, self._run_fast, prompt, width, height, num_frames, fps, seed, generate_audio, images)
        else:
            return await loop.run_in_executor(None, self._run_pro, prompt, width, height, num_frames, fps, seed, generate_audio, images)

async def generate_audio_to_video(self, prompt, audio_path, image_path, model, width, height, num_frames, fps, seed):
    async with self.denoiser_hub.lock:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._run_a2v, prompt, audio_path, image_path, width, height, num_frames, fps, seed)

async def retake(self, video_path, start_time, duration, mode, prompt, seed):
    async with self.denoiser_hub.lock:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._run_retake, video_path, start_time, duration, mode, prompt, seed)
```

**Step 2: Add config toggle**

```python
# config.py
USE_SPLIT_GPU = len(GPU_DEVICES) >= 2
```

**Step 3: Update server.py**

```python
# Replace manager creation
if config.USE_SPLIT_GPU:
    from split_model_manager import SplitModelManager
    manager = SplitModelManager()
else:
    from pipeline_manager import PipelineManager
    manager = PipelineManager()

# Replace readiness checks: `if not manager.workers:` → `if not manager.is_ready:`
# Add `is_ready` property to PipelineManager too: `return bool(self.workers)`
```

**Step 4: Move shared helpers out of pipeline_manager.py**

`_snap_to_multiple`, `_resolution_to_dims`, `_duration_to_frames` are used by server.py. Move to a shared `utils.py` or keep importing from pipeline_manager.py.

**Step 5: Commit**

```bash
git add split_model_manager.py server.py config.py pipeline_manager.py
git commit -m "feat: integrate SplitModelManager with server and config toggle"
```

---

## Task 6: GPU smoke test — all endpoints

**Files:** None (manual testing)

**Step 1: Start server**
```bash
./run.sh
```

**Step 2: Test health**
```bash
curl http://localhost:8090/health
```

**Step 3: Test text-to-video fast (exercises distilled checkpoint swap)**
```bash
curl -s -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cat walking","model":"ltx-2-3-fast","resolution":"1920x1080","duration":2.0,"fps":24.0}' \
  -o /tmp/test_split_fast.mp4 -w "HTTP %{http_code}, Size: %{size_download}\n"
ffprobe -v quiet -print_format json -show_streams /tmp/test_split_fast.mp4
```

**Step 4: Test text-to-video pro (exercises dev→dev_lora→dev swap)**
```bash
curl -s -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A dog on a beach","model":"ltx-2-3-pro","resolution":"1920x1080","duration":2.0,"fps":24.0,"generate_audio":true}' \
  -o /tmp/test_split_pro.mp4 -w "HTTP %{http_code}, Size: %{size_download}\n"
```

**Step 5: Test upload + i2v**
```bash
# Upload image, then call /v1/image-to-video
```

**Step 6: Test upload + a2v (exercises denoise_video_only + original audio passthrough)**
```bash
# Upload audio (+ optional image), then call /v1/audio-to-video
```

**Step 7: Test retake (exercises TemporalRegionMask + single-stage)**
```bash
# Upload video, then call /v1/retake
```

**Step 8: Compare timing vs old PipelineManager**

Toggle `USE_SPLIT_GPU = False` in config.py, restart, re-run. Compare wall-clock times. Key metrics:
- Time to first frame (model loading overhead)
- Total generation time
- GPU memory usage (`nvidia-smi`)

**Step 9: Commit**
```bash
git commit -m "test: validate dual-GPU split — all endpoints passing"
```

---

## Notes

### Files to study before implementing
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py` — encode_prompts(), combined_image_conditionings(), all denoising funcs
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py` — euler_denoising_loop(), denoise_video_only()
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py` — pro flow reference
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/distilled.py` — fast flow reference
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py` — a2v flow reference
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/retake.py` — retake flow reference
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py` — ModelLedger API
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-core/src/ltx_core/types.py` — PipelineComponents, LatentState, Modality

### Risk: 88GB peak during transformer swap
Load-then-swap holds both old (44GB) and new (44GB) transformers briefly. This leaves ~8GB margin on 96GB GPU. If activation memory from a previous request hasn't fully freed, this could OOM. Mitigation: `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` before loading new transformer. Fallback: delete-first pattern with try/except.

### Risk: Denoising function internals
The denoising functions (`simple_denoising_func`, `multi_modal_guider_factory_denoising_func`) are closures that capture the transformer reference. They need the transformer on GPU:1, which it already is. But study the exact closure signatures carefully — don't guess parameters.

### Invariant: Transformer state on lock release
After every generation, the transformer MUST be restored to a known state. Pro model swaps to dev_lora for stage 2, then back to dev. If restore fails, log error and set `_transformer_state` to reflect actual state. Next request will trigger appropriate swap.
