"""Flux 2 Dev image generation manager.

Loads Flux2Pipeline with FP8 layerwise casting on a dedicated GPU.
FP8 transformer (~32GB) + bf16 Mistral text encoder (~47GB) = ~79GB.
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import os
import time

import torch
from PIL import Image

import config

# Ensure HF uses our NVMe cache
os.environ.setdefault("HF_HOME", "/mnt/nvme-1/huggingface")

logger = logging.getLogger(__name__)



class FluxManager:
    """Manages Flux 2 pipeline lifecycle and inference on a dedicated GPU."""

    def __init__(self) -> None:
        self._pipe = None
        self._device = config.FLUX_DEVICE
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        """Load the Flux 2 pipeline with FP8 transformer."""
        from diffusers import Flux2Pipeline
        from diffusers.models import Flux2Transformer2DModel

        logger.info("Loading Flux2 pipeline on %s ...", self._device)
        t0 = time.monotonic()

        # FP8 layerwise casting: stores weights in float8_e4m3fn (~32GB)
        # but computes in bf16. Halves VRAM vs full bf16 (~64GB).
        transformer = Flux2Transformer2DModel.from_pretrained(
            config.FLUX_MODEL_REPO,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            cache_dir=config.HF_CACHE_DIR,
        )
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )

        pipe = Flux2Pipeline.from_pretrained(
            config.FLUX_MODEL_REPO,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            cache_dir=config.HF_CACHE_DIR,
        )
        pipe = pipe.to(self._device)
        self._pipe = pipe

        elapsed = time.monotonic() - t0
        logger.info("Flux2 pipeline loaded in %.1fs on %s", elapsed, self._device)

    def unload(self) -> None:
        """Free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            torch.cuda.synchronize(torch.device(self._device))
            torch.cuda.empty_cache()
            logger.info("Flux2 pipeline unloaded from %s", self._device)

    @torch.inference_mode()
    def _generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        callback_on_step_end: object = None,
    ) -> bytes:
        """Generate an image (txt2img) and return WEBP bytes."""
        generator = torch.Generator(device=self._device).manual_seed(seed)

        kwargs: dict = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if callback_on_step_end is not None:
            kwargs["callback_on_step_end"] = callback_on_step_end

        try:
            result = self._pipe(**kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            logger.exception("Flux generate OOM, unloading pipeline")
            self.unload()
            raise

        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=95)
        return buf.getvalue()

    @torch.inference_mode()
    def _img2img(
        self,
        prompt: str,
        image_path: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        callback_on_step_end: object = None,
    ) -> bytes:
        """Edit an image using Flux 2 Kontext reference latents."""
        generator = torch.Generator(device=self._device).manual_seed(seed)
        ref_image = Image.open(image_path).convert("RGB")

        kwargs: dict = dict(
            image=ref_image,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if callback_on_step_end is not None:
            kwargs["callback_on_step_end"] = callback_on_step_end

        try:
            result = self._pipe(**kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            logger.exception("Flux img2img OOM, unloading pipeline")
            self.unload()
            raise

        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=95)
        return buf.getvalue()

    # --- Async API ---

    async def generate_text_to_image(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 0,
        callback_on_step_end: object = None,
    ) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._generate, prompt, width, height,
                num_inference_steps, guidance_scale, seed,
                callback_on_step_end,
            )

    async def generate_image_to_image(
        self,
        prompt: str,
        image_path: str,
        width: int,
        height: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 0,
        callback_on_step_end: object = None,
    ) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._img2img, prompt, image_path, width, height,
                num_inference_steps, guidance_scale, seed,
                callback_on_step_end,
            )
