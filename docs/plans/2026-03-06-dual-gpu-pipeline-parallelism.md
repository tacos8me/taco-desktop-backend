# Dual-GPU Pipeline Parallelism Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split LTX pipeline execution across two GPUs — text encoding on GPU:0, denoising on GPU:1 — to eliminate model loading churn and keep both GPUs utilized.

**Architecture:** GPU:0 holds Gemma 12B text encoder + VAE encoders permanently. GPU:1 holds the 22B transformer + spatial upsampler + decoders permanently. Text embeddings (~8MB) transfer once per generation via `.to()`. Stage 2 LoRA swap requires transformer reload (~2-3s from NVMe, <1% of pro model gen time).

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
  - LTX-2.3 22B transformer (permanent, 44GB)
  - Spatial upsampler (permanent, 0.24GB)
  - Video VAE decoder (permanent, ~0.16GB)
  - Audio VAE decoder (permanent, ~0.05GB)
  - Vocoder (permanent, ~0.04GB)
```

### Validated Properties
- All 4 pipelines (t2v, i2v, a2v, retake) have clean split points
- Only ~8MB of embeddings cross the GPU boundary, once per generation
- No bidirectional communication during denoising
- Retake is single-stage, fully sequential, compatible
- LoRA fusion is permanent (no unfuse) — stage 2 requires transformer reload
- Local code matches upstream Lightricks/LTX-2 main (as of 2026-03-05)
- Safetensors loading is selective per component (not full 43GB checkpoint)

### VRAM Budget
| GPU | Weights | Activations | Total | Headroom |
|-----|---------|-------------|-------|----------|
| GPU:0 | 24.3 GB | 3-4 GB | ~28 GB | 68 GB |
| GPU:1 | 44.5 GB | 3-5 GB | ~49 GB | 47 GB |

---

## Task 1: Create SplitModelManager class

**Files:**
- Create: `split_model_manager.py`
- Test: `tests/test_split_model_manager.py`

**Context:** Replace the current `PipelineManager` which loads full pipeline objects per GPU. The new `SplitModelManager` loads individual model components onto specific GPUs and keeps them resident.

**Step 1: Write the SplitModelManager skeleton**

```python
"""Split-GPU model manager for LTX-2 video generation.

Keeps text encoder + VAE encoders on GPU:0 (encoder_device),
transformer + decoders on GPU:1 (denoiser_device). Models stay
resident — no per-request loading/unloading.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import torch
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.utils.model_ledger import ModelLedger

import config

logger = logging.getLogger(__name__)


@dataclass
class EncoderHub:
    """Models resident on the encoder GPU."""
    device: torch.device
    text_encoder: object  # GemmaTextEncoder
    embeddings_processor: object  # EmbeddingsProcessor
    video_encoder: object  # VideoEncoder
    audio_encoder: object  # AudioEncoder


@dataclass
class DenoiserHub:
    """Models resident on the denoiser GPU."""
    device: torch.device
    transformer: object  # X0Model
    spatial_upsampler: object  # LatentUpsampler
    video_decoder: object  # VideoDecoder
    audio_decoder: object  # AudioDecoder
    vocoder: object  # Vocoder
    lock: asyncio.Lock  # Only one generation at a time on the denoiser


class SplitModelManager:
    """Loads models once across two GPUs and dispatches inference."""

    def __init__(self) -> None:
        self.encoder_hub: EncoderHub | None = None
        self.denoiser_hub: DenoiserHub | None = None

    def load_all(self) -> None:
        """Load encoder models on GPU:0, denoiser models on GPU:1."""
        encoder_device = torch.device(config.GPU_DEVICES[0])
        denoiser_device = torch.device(config.GPU_DEVICES[1])

        # Build a ModelLedger for each device to extract components
        encoder_ledger = ModelLedger(
            dtype=torch.bfloat16,
            device=encoder_device,
            checkpoint_path=config.DEV_CHECKPOINT,
            gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
            loras=(),
        )
        denoiser_ledger = ModelLedger(
            dtype=torch.bfloat16,
            device=denoiser_device,
            checkpoint_path=config.DEV_CHECKPOINT,
            gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
            loras=(),
        )

        logger.info("Loading encoder hub on %s ...", encoder_device)
        self.encoder_hub = EncoderHub(
            device=encoder_device,
            text_encoder=encoder_ledger.text_encoder(),
            embeddings_processor=encoder_ledger.embeddings_processor(),
            video_encoder=encoder_ledger.video_encoder(),
            audio_encoder=encoder_ledger.audio_encoder(),
        )

        logger.info("Loading denoiser hub on %s ...", denoiser_device)
        self.denoiser_hub = DenoiserHub(
            device=denoiser_device,
            transformer=denoiser_ledger.transformer(),
            spatial_upsampler=denoiser_ledger.spatial_upsampler(),
            video_decoder=denoiser_ledger.video_decoder(),
            audio_decoder=denoiser_ledger.audio_decoder(),
            vocoder=denoiser_ledger.vocoder(),
            lock=asyncio.Lock(),
        )
        logger.info("All models loaded.")
```

**Step 2: Write a basic test**

```python
# tests/test_split_model_manager.py
"""Smoke test for SplitModelManager — import and init only (no GPU needed)."""

def test_split_model_manager_imports():
    from split_model_manager import SplitModelManager
    mgr = SplitModelManager()
    assert mgr.encoder_hub is None
    assert mgr.denoiser_hub is None
```

**Step 3: Run test**

Run: `uv run --no-sync pytest tests/test_split_model_manager.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add split_model_manager.py tests/test_split_model_manager.py
git commit -m "feat: add SplitModelManager skeleton with dual-GPU hubs"
```

---

## Task 2: Implement text encoding on encoder hub

**Files:**
- Modify: `split_model_manager.py`

**Context:** Extract the `encode_prompts()` logic from ltx-pipelines helpers.py but use our pre-loaded text encoder and embeddings processor instead of loading/deleting them each time.

**Step 1: Add encode_text method**

The key insight: `encode_prompts()` in helpers.py loads the text encoder from ModelLedger, encodes, deletes it, then loads the embeddings processor, processes, deletes it. We skip those load/delete steps since our models are permanent.

```python
def encode_text(
    self,
    prompts: list[str],
) -> list[EmbeddingsProcessorOutput]:
    """Encode prompts on encoder_device using resident text encoder."""
    hub = self.encoder_hub
    # Encode with Gemma (returns hidden_states, attention_mask)
    encoded = [hub.text_encoder.encode(p) for p in prompts]
    # Process through embeddings processor
    results = []
    for hidden_states, attention_mask in encoded:
        result = hub.embeddings_processor(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        results.append(result)
    return results
```

Note: The actual `encode_prompts()` function in helpers.py (lines 48-86) should be studied closely. It calls `text_encoder.encode()` then `embeddings_processor.create_embeddings()`. Match that exact interface — don't guess the method signatures.

**Step 2: Add transfer_to_denoiser helper**

```python
def _transfer_contexts(
    self,
    contexts: list[EmbeddingsProcessorOutput],
) -> list[EmbeddingsProcessorOutput]:
    """Move embedding tensors from encoder_device to denoiser_device."""
    target = self.denoiser_hub.device
    transferred = []
    for ctx in contexts:
        transferred.append(EmbeddingsProcessorOutput(
            video_encoding=ctx.video_encoding.to(target),
            audio_encoding=ctx.audio_encoding.to(target) if ctx.audio_encoding is not None else None,
            attention_mask=ctx.attention_mask.to(target),
        ))
    return transferred
```

**Step 3: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: add text encoding and cross-GPU context transfer"
```

---

## Task 3: Implement image/audio conditioning on encoder hub

**Files:**
- Modify: `split_model_manager.py`

**Context:** Image conditioning (for i2v) and audio encoding (for a2v) use the VAE encoders on GPU:0. The encoded latents then transfer to GPU:1 for denoising.

**Step 1: Add encode_image_conditioning method**

Study `combined_image_conditionings()` in helpers.py (lines 89-123) and `image_conditionings_by_replacing_latent()` (lines 126-153). These use `video_encoder` to encode conditioning images. Our version uses the resident video encoder instead of loading from ModelLedger.

```python
def encode_image_conditioning(
    self,
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    num_frames: int,
) -> list:  # Returns conditioning items
    """Encode conditioning images on encoder_device."""
    hub = self.encoder_hub
    # Use resident video_encoder — same logic as helpers.py combined_image_conditionings()
    # but without the load/delete cycle
    ...
```

**Step 2: Add encode_audio method for A2V**

```python
def encode_audio(
    self,
    audio_path: str,
    duration: float,
) -> torch.Tensor:
    """Encode audio file to latent on encoder_device."""
    hub = self.encoder_hub
    # decode_audio_from_file() + vae_encode_audio() with resident audio_encoder
    ...
```

**Step 3: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: add image/audio conditioning encoding on encoder hub"
```

---

## Task 4: Implement denoising on denoiser hub

**Files:**
- Modify: `split_model_manager.py`

**Context:** This is the core — run the denoising loop using the resident transformer on GPU:1. The transformer stays loaded between requests. For stage 2 (pro model), we reload with distilled LoRA.

**Step 1: Study the denoising flow**

Read `ti2vid_two_stages.py` lines 137-177 (stage 1) and 200-235 (stage 2). The key calls are:
- `denoise_audio_video()` from samplers.py — runs the euler denoising loop
- `simple_denoising_func()` or `multi_modal_guider_factory_denoising_func()` from helpers.py

**Step 2: Implement stage 1 denoising**

```python
@torch.inference_mode()
def _denoise_stage1(
    self,
    video_state: LatentState,
    audio_state: LatentState,
    v_context_p: torch.Tensor,
    v_context_n: torch.Tensor | None,
    a_context_p: torch.Tensor | None,
    a_context_n: torch.Tensor | None,
    num_inference_steps: int,
    video_guider_params: dict,
    audio_guider_params: dict,
    model: str,  # "ltx-2-3-fast" or "ltx-2-3-pro"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run stage 1 denoising on denoiser_device."""
    hub = self.denoiser_hub
    # Use resident transformer — no load/delete
    # Build sigma schedule, create denoising func, run euler loop
    ...
```

**Step 3: Implement stage 2 with LoRA reload**

For pro model, stage 2 needs the transformer with distilled LoRA. Since LoRA fusion is permanent, we must reload:

```python
def _reload_transformer_with_lora(self) -> None:
    """Reload transformer with distilled LoRA for stage 2."""
    del self.denoiser_hub.transformer
    torch.cuda.synchronize(self.denoiser_hub.device)
    torch.cuda.empty_cache()

    distilled_lora = LoraPathStrengthAndSDOps(
        path=config.DISTILLED_LORA, strength=1.0,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=self.denoiser_hub.device,
        checkpoint_path=config.DEV_CHECKPOINT,
        gemma_root_path=config.GEMMA_ROOT,
        spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
        loras=(distilled_lora,),
    )
    self.denoiser_hub.transformer = ledger.transformer()

def _restore_base_transformer(self) -> None:
    """Restore base transformer (no LoRA) after stage 2."""
    del self.denoiser_hub.transformer
    torch.cuda.synchronize(self.denoiser_hub.device)
    torch.cuda.empty_cache()

    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=self.denoiser_hub.device,
        checkpoint_path=config.DEV_CHECKPOINT,
        gemma_root_path=config.GEMMA_ROOT,
        spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
        loras=(),
    )
    self.denoiser_hub.transformer = ledger.transformer()
```

**Step 4: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: add stage 1/2 denoising with LoRA swap on denoiser hub"
```

---

## Task 5: Implement VAE decoding + video encoding on denoiser hub

**Files:**
- Modify: `split_model_manager.py`

**Context:** After denoising, decode latents to video/audio and encode to MP4. All decoder models are on GPU:1.

**Step 1: Add decode_and_encode method**

```python
def _decode_to_video(
    self,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    num_frames: int,
    fps: float,
    include_audio: bool = True,
) -> bytes:
    """Decode latents and encode to MP4 bytes."""
    hub = self.denoiser_hub
    # vae_decode_video() with resident video_decoder
    # vae_decode_audio() with resident audio_decoder + vocoder
    # encode_video() to temp file, read bytes
    ...
```

**Step 2: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: add VAE decoding and video encoding on denoiser hub"
```

---

## Task 6: Wire up full generation flows

**Files:**
- Modify: `split_model_manager.py`

**Context:** Combine all the pieces into complete async generation methods that match the current PipelineManager API.

**Step 1: Implement generate_text_to_video**

```python
async def generate_text_to_video(
    self,
    prompt: str,
    model: str,
    width: int, height: int,
    num_frames: int, fps: float,
    seed: int,
    generate_audio: bool = True,
) -> bytes:
    async with self.denoiser_hub.lock:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._run_t2v, prompt, model, width, height,
            num_frames, fps, seed, generate_audio,
        )
```

**Step 2: Implement _run_t2v (blocking)**

```python
@torch.inference_mode()
def _run_t2v(self, prompt, model, width, height, num_frames, fps, seed, generate_audio):
    # 1. Text encode on GPU:0
    contexts = self.encode_text([prompt] if model == "ltx-2-3-fast" else [prompt, DEFAULT_NEGATIVE_PROMPT])
    contexts = self._transfer_contexts(contexts)

    # 2. Build noise + latent states on GPU:1
    ...

    # 3. Stage 1 denoise on GPU:1
    video_latent, audio_latent = self._denoise_stage1(...)

    # 4. Spatial upsample on GPU:1
    ...

    # 5. Stage 2 denoise (reload LoRA if pro)
    if model == "ltx-2-3-pro":
        self._reload_transformer_with_lora()
    video_latent, audio_latent = self._denoise_stage2(...)
    if model == "ltx-2-3-pro":
        self._restore_base_transformer()

    # 6. Decode + encode MP4
    return self._decode_to_video(video_latent, audio_latent, num_frames, fps, generate_audio)
```

**Step 3: Implement i2v, a2v, retake flows (same pattern)**

Each follows: encode on GPU:0 → transfer → denoise on GPU:1 → decode on GPU:1.

**Step 4: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: wire up full t2v/i2v/a2v/retake flows in SplitModelManager"
```

---

## Task 7: Integrate with server.py

**Files:**
- Modify: `server.py`
- Modify: `config.py`

**Context:** Replace `PipelineManager` with `SplitModelManager` in the FastAPI server. Keep old `pipeline_manager.py` as fallback.

**Step 1: Add config toggle**

```python
# config.py
USE_SPLIT_GPU = True  # Use SplitModelManager when True, PipelineManager when False
ENCODER_DEVICE = "cuda:0"
DENOISER_DEVICE = "cuda:2"
```

**Step 2: Update server.py lifespan**

```python
from split_model_manager import SplitModelManager
from pipeline_manager import PipelineManager

if config.USE_SPLIT_GPU:
    manager = SplitModelManager()
else:
    manager = PipelineManager()
```

Both classes expose the same async API (generate_text_to_video, etc.), so endpoints need no changes.

**Step 3: Commit**

```bash
git add server.py config.py
git commit -m "feat: integrate SplitModelManager with config toggle"
```

---

## Task 8: GPU smoke test

**Files:** None (manual testing)

**Step 1: Start server with split GPU mode**

```bash
./run.sh
```

**Step 2: Run health check**

```bash
curl http://localhost:8090/health
```

**Step 3: Test text-to-video (fast model)**

```bash
curl -s -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cat walking","model":"ltx-2-3-fast","resolution":"1920x1080","duration":2.0,"fps":24.0}' \
  -o /tmp/test_split_fast.mp4 -w "HTTP %{http_code}, Size: %{size_download}\n"
```

**Step 4: Test text-to-video (pro model — exercises LoRA swap)**

```bash
curl -s -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A dog on a beach","model":"ltx-2-3-pro","resolution":"1920x1080","duration":2.0,"fps":24.0,"generate_audio":true}' \
  -o /tmp/test_split_pro.mp4 -w "HTTP %{http_code}, Size: %{size_download}\n"
```

**Step 5: Test upload + i2v + a2v + retake**

Full endpoint coverage.

**Step 6: Compare timing vs old PipelineManager**

Toggle `USE_SPLIT_GPU = False`, restart, re-run same tests. Compare wall-clock times.

**Step 7: Commit results**

```bash
git commit -m "test: validate dual-GPU split pipeline — all endpoints passing"
```

---

## Notes

### Key files to study before implementing
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py` — encode_prompts(), combined_image_conditionings(), denoising funcs
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py` — euler_denoising_loop()
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py` — reference for full generation flow
- `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py` — ModelLedger API

### Risk: encode_prompts() internals
The `encode_prompts()` function in helpers.py has specific logic for how it calls the text encoder and embeddings processor. Don't rewrite from scratch — study the exact method calls and replicate them with pre-loaded models.

### Risk: Denoising function construction
The denoising functions (`simple_denoising_func`, `multi_modal_guider_factory_denoising_func`) are closures that capture the transformer reference. They need the transformer on the correct device. Since our transformer is already on GPU:1, this should work naturally.

### Risk: Spatial upsampler integration
The spatial upsampler runs between stage 1 and stage 2. It operates on latents (already on GPU:1). Study `upsample_video()` calls in the pipeline code.
