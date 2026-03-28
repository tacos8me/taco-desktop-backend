"""Flux image generation manager with model swapping.

Supports FLUX.2-dev (text-to-image, img2img) and FLUX.2-klein-9b-kv
(text-to-image, img2img, multi-reference editing). Models swap on demand
— evicts current pipeline and loads requested model if different.
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
    """Manages Flux pipelines with on-demand model swapping."""

    def __init__(self) -> None:
        self._pipe = None
        self._current_model: str | None = None
        self._device = config.FLUX_DEVICE
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._pipe is not None

    def load(self, model_name: str = "flux2-dev") -> None:
        """Load a Flux pipeline. Dev model uses fused Turbo LoRA + FP8 layerwise casting."""
        from diffusers.models import Flux2Transformer2DModel

        model_repo = config.FLUX_MODELS[model_name]
        logger.info("Loading %s (%s) on %s ...", model_name, model_repo, self._device)
        t0 = time.monotonic()

        _load_kw = dict(torch_dtype=torch.bfloat16, cache_dir=config.HF_CACHE_DIR, local_files_only=True)

        if model_name == "flux2-klein":
            from diffusers import Flux2KleinPipeline
            klein_ckpt, klein_snap = self._resolve_klein_checkpoint()
            transformer = Flux2Transformer2DModel.from_single_file(
                klein_ckpt, torch_dtype=torch.bfloat16,
                config=str(klein_snap / "transformer"),
            )
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )
            pipe = Flux2KleinPipeline.from_pretrained(
                str(klein_snap), transformer=transformer,
                torch_dtype=torch.bfloat16, local_files_only=True,
            )
        else:
            from diffusers import Flux2Pipeline
            pipe = Flux2Pipeline.from_pretrained(model_repo, **_load_kw)
            # Fuse Turbo LoRA into base weights, then apply FP8 casting
            try:
                pipe.load_lora_weights(
                    config.FLUX_TURBO_LORA,
                    weight_name=config.FLUX_TURBO_LORA_WEIGHT,
                    cache_dir=config.HF_CACHE_DIR,
                )
                pipe.fuse_lora()
                pipe.unload_lora_weights()
                logger.info("Turbo LoRA fused into base weights")
            except Exception:
                logger.warning("Turbo LoRA not available", exc_info=True)
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )

        pipe = pipe.to(self._device)
        self._pipe = pipe
        self._current_model = model_name

        elapsed = time.monotonic() - t0
        logger.info("%s loaded in %.1fs on %s", model_name, elapsed, self._device)

    def _resolve_dev_snapshot(self):
        """Find the FLUX.2-dev snapshot dir in HF cache (needed for transformer config)."""
        from pathlib import Path
        model_dir = Path(config.HF_CACHE_DIR) / "models--black-forest-labs--FLUX.2-dev" / "snapshots"
        for snap_dir in model_dir.iterdir():
            if (snap_dir / "transformer" / "config.json").exists():
                return snap_dir
        raise FileNotFoundError("FLUX.2-dev snapshot not found in HF cache")

    def _resolve_klein_checkpoint(self) -> tuple:
        """Find the Klein single-file checkpoint in the HF cache. Returns (ckpt_path, snapshot_dir)."""
        from pathlib import Path
        cache_dir = Path(config.HF_CACHE_DIR)
        model_dir = cache_dir / "models--black-forest-labs--FLUX.2-klein-9b-kv"
        for snap_dir in (model_dir / "snapshots").iterdir():
            ckpt = snap_dir / "flux-2-klein-9b-kv.safetensors"
            if ckpt.exists():
                return str(ckpt), snap_dir
        raise FileNotFoundError("Klein checkpoint not found in HF cache. Run: huggingface-cli download black-forest-labs/FLUX.2-klein-9b-kv --cache-dir /mnt/nvme-1/huggingface/hub")

    def unload(self) -> None:
        """Free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._current_model = None
            gc.collect()
            torch.cuda.synchronize(torch.device(self._device))
            torch.cuda.empty_cache()
            logger.info("Flux pipeline unloaded from %s", self._device)

    def ensure_model(self, model_name: str) -> None:
        """Swap model if needed. No-op if already loaded."""
        if self._current_model == model_name:
            return
        if self._pipe is not None:
            logger.info("Swapping Flux model: %s → %s", self._current_model, model_name)
            self.unload()
        self.load(model_name)

    def _to_webp(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=95)
        return buf.getvalue()

    @torch.inference_mode()
    def _generate(
        self, prompt: str, width: int, height: int,
        num_inference_steps: int, guidance_scale: float, seed: int,
        model: str = "flux2-dev", turbo: bool = False,
        callback_on_step_end: object = None,
    ) -> bytes:
        """Generate an image (txt2img) and return WEBP bytes."""
        self.ensure_model(model)
        generator = torch.Generator(device=self._device).manual_seed(seed)

        kwargs: dict = dict(
            prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, generator=generator,
        )
        if turbo and model == "flux2-dev":
            kwargs["sigmas"] = config.FLUX_TURBO_SIGMAS
            kwargs["num_inference_steps"] = 8
            kwargs["guidance_scale"] = 2.5
        if callback_on_step_end is not None:
            kwargs["callback_on_step_end"] = callback_on_step_end
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        try:
            result = self._pipe(**kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            logger.exception("Flux generate OOM, unloading pipeline")
            self.unload()
            raise

        return self._to_webp(result.images[0])

    @torch.inference_mode()
    def _img2img(
        self, prompt: str, image_path: str, width: int, height: int,
        num_inference_steps: int, guidance_scale: float, seed: int,
        model: str = "flux2-dev", turbo: bool = False,
        callback_on_step_end: object = None,
    ) -> bytes:
        """Edit an image using single reference."""
        self.ensure_model(model)
        generator = torch.Generator(device=self._device).manual_seed(seed)
        ref_image = Image.open(image_path).convert("RGB")

        kwargs: dict = dict(
            image=ref_image, prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, generator=generator,
        )
        if turbo and model == "flux2-dev":
            kwargs["sigmas"] = config.FLUX_TURBO_SIGMAS
            kwargs["num_inference_steps"] = 8
            kwargs["guidance_scale"] = 2.5
        if callback_on_step_end is not None:
            kwargs["callback_on_step_end"] = callback_on_step_end
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        try:
            result = self._pipe(**kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            logger.exception("Flux img2img OOM, unloading pipeline")
            self.unload()
            raise

        return self._to_webp(result.images[0])

    @torch.inference_mode()
    def _edit(
        self, prompt: str, image_paths: list[str], width: int, height: int,
        num_inference_steps: int = 4, guidance_scale: float = 4.0,
        seed: int = 0, callback_on_step_end: object = None,
    ) -> bytes:
        """Multi-reference image editing via Klein."""
        self.ensure_model("flux2-klein")
        generator = torch.Generator(device=self._device).manual_seed(seed)
        images = [Image.open(p).convert("RGB") for p in image_paths]

        kwargs: dict = dict(
            image=images, prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, generator=generator,
        )
        if callback_on_step_end is not None:
            kwargs["callback_on_step_end"] = callback_on_step_end

        try:
            result = self._pipe(**kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            logger.exception("Flux edit OOM, unloading pipeline")
            self.unload()
            raise

        return self._to_webp(result.images[0])

    # --- Async API ---

    async def generate_text_to_image(self, *, model: str = "flux2-dev", **kwargs) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self._generate(model=model, **kwargs),
            )

    async def generate_image_to_image(self, *, model: str = "flux2-dev", **kwargs) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self._img2img(model=model, **kwargs),
            )

    async def generate_image_edit(self, *, model: str = "flux2-klein", **kwargs) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self._edit(**kwargs),
            )
