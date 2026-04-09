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
from pathlib import Path

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
        self._current_lora: tuple[str, float] | None = None  # (path, strength)
        self._device = config.FLUX_DEVICE
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._pipe is not None

    def load(self, model_name: str = "flux2-dev", user_lora: tuple[str, float] | None = None) -> None:
        """Load a Flux pipeline with optional fused user LoRA + FP8 layerwise casting.

        LoRA fusion MUST happen before enable_layerwise_casting — PEFT cannot inject
        adapters into an already-FP8-cast transformer (diffusers issues #9514, #11648).
        """
        from diffusers.models import Flux2Transformer2DModel

        model_repo = config.FLUX_MODELS[model_name]
        lora_desc = f" + lora({Path(user_lora[0]).name}@{user_lora[1]:.2f})" if user_lora else ""
        logger.info("Loading %s (%s)%s on %s ...", model_name, model_repo, lora_desc, self._device)
        t0 = time.monotonic()

        _load_kw = dict(torch_dtype=torch.bfloat16, cache_dir=config.HF_CACHE_DIR, local_files_only=True)

        if model_name == "flux2-klein":
            from diffusers import Flux2KleinKVPipeline
            klein_ckpt, klein_snap = self._resolve_klein_checkpoint()
            transformer = Flux2Transformer2DModel.from_single_file(
                klein_ckpt, torch_dtype=torch.bfloat16,
                config=str(klein_snap / "transformer"),
            )
            pipe = Flux2KleinKVPipeline.from_pretrained(
                str(klein_snap), transformer=transformer,
                torch_dtype=torch.bfloat16, local_files_only=True,
            )
            if user_lora:
                self._fuse_user_lora(pipe, user_lora)
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
                skip_modules_pattern=["x_embedder", "context_embedder", "proj_out"],
            )
        else:
            from diffusers import Flux2Pipeline
            pipe = Flux2Pipeline.from_pretrained(model_repo, **_load_kw)
            if user_lora:
                self._fuse_user_lora(pipe, user_lora)
            # Clean FP8 on base weights — skip precision-critical input/output layers
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
                skip_modules_pattern=["x_embedder", "context_embedder", "proj_out"],
            )

        pipe = pipe.to(self._device)
        self._pipe = pipe
        self._current_model = model_name
        self._current_lora = user_lora

        elapsed = time.monotonic() - t0
        logger.info("%s loaded in %.1fs on %s", model_name, elapsed, self._device)

    @staticmethod
    def _fuse_user_lora(pipe, user_lora: tuple[str, float]) -> None:
        """Load a user LoRA, fuse it into the transformer, and drop the adapter.

        Must be called BEFORE `enable_layerwise_casting` — see load() docstring.
        """
        path, strength = user_lora
        pipe.load_lora_weights(path, adapter_name="user_lora")
        pipe.fuse_lora(adapter_names=["user_lora"], lora_scale=strength)
        pipe.unload_lora_weights()

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
            self._current_lora = None
            gc.collect()
            torch.cuda.synchronize(torch.device(self._device))
            torch.cuda.empty_cache()
            logger.info("Flux pipeline unloaded from %s", self._device)

    def ensure_model(self, model_name: str, user_lora: tuple[str, float] | None = None) -> None:
        """Swap model if needed. Cache key is (model_name, user_lora) — any change forces reload."""
        if self._current_model == model_name and self._current_lora == user_lora:
            return
        if self._pipe is not None:
            logger.info(
                "Swapping Flux pipeline: %s/%s → %s/%s",
                self._current_model, self._current_lora, model_name, user_lora,
            )
            self.unload()
        self.load(model_name, user_lora=user_lora)

    def _to_webp(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=95)
        return buf.getvalue()

    @torch.inference_mode()
    def _generate(
        self, prompt: str, width: int, height: int,
        num_inference_steps: int, guidance_scale: float, seed: int,
        model: str = "flux2-dev", turbo: bool = False,
        lora_path: str | None = None, lora_strength: float = 1.0,
        callback_on_step_end: object = None,
    ) -> bytes:
        """Generate an image (txt2img) and return WEBP bytes."""
        user_lora = (lora_path, lora_strength) if lora_path else None
        self.ensure_model(model, user_lora=user_lora)
        generator = torch.Generator(device=self._device).manual_seed(seed)

        kwargs: dict = dict(
            prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        if model == "flux2-klein":
            # Klein KV pipeline doesn't accept guidance_scale (distilled, no CFG)
            pass
        else:
            kwargs["guidance_scale"] = guidance_scale
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

        if model == "flux2-klein":
            gc.collect()
            torch.cuda.empty_cache()

        return self._to_webp(result.images[0])

    @torch.inference_mode()
    def _img2img(
        self, prompt: str, image_path: str, width: int, height: int,
        num_inference_steps: int, guidance_scale: float, seed: int,
        model: str = "flux2-dev", turbo: bool = False,
        lora_path: str | None = None, lora_strength: float = 1.0,
        callback_on_step_end: object = None,
    ) -> bytes:
        """Edit an image using single reference."""
        user_lora = (lora_path, lora_strength) if lora_path else None
        self.ensure_model(model, user_lora=user_lora)
        generator = torch.Generator(device=self._device).manual_seed(seed)
        ref_image = Image.open(image_path).convert("RGB")

        kwargs: dict = dict(
            image=ref_image, prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        if model == "flux2-klein":
            pass
        else:
            kwargs["guidance_scale"] = guidance_scale
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

        if model == "flux2-klein":
            gc.collect()
            torch.cuda.empty_cache()

        return self._to_webp(result.images[0])

    @torch.inference_mode()
    def _edit(
        self, prompt: str, image_paths: list[str], width: int, height: int,
        num_inference_steps: int = 4, guidance_scale: float = 4.0,
        seed: int = 0,
        lora_path: str | None = None, lora_strength: float = 1.0,
        callback_on_step_end: object = None,
    ) -> bytes:
        """Multi-reference image editing via Klein."""
        user_lora = (lora_path, lora_strength) if lora_path else None
        self.ensure_model("flux2-klein", user_lora=user_lora)
        generator = torch.Generator(device=self._device).manual_seed(seed)
        images = [Image.open(p).convert("RGB") for p in image_paths]

        kwargs: dict = dict(
            image=images, prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        # Klein KV pipeline doesn't accept guidance_scale
        if callback_on_step_end is not None:
            kwargs["callback_on_step_end"] = callback_on_step_end
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        try:
            result = self._pipe(**kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            logger.exception("Flux edit OOM, unloading pipeline")
            self.unload()
            raise

        gc.collect()
        torch.cuda.empty_cache()

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
