"""Flux image generation manager with model swapping.

Supports FLUX.2-dev (text-to-image, img2img) and FLUX.2-klein-9b-kv
(text-to-image, img2img, multi-reference editing). Models swap on demand
— evicts current pipeline and loads requested model if different.

Precision: full bf16 throughout (matches ComfyUI default). FP8 layerwise
casting was dropped in v1.1.1 after diagnosing screendoor/dithering
artifacts traced to the `fuse_lora → enable_layerwise_casting(fp8)`
interaction (diffusers PR #10685, Flux issue #406).

VRAM budget on a 96 GB Blackwell:
- Flux 2 Dev: transformer ~60.4 GB + text encoder (Mistral-3.2-24B) ~45.2 GB
  + VAE ~0.3 GB = ~105.9 GB bf16. Does NOT fit all-resident, so we
  `enable_model_cpu_offload()` which pages components CPU↔GPU on demand.
  Denoising peak: ~75 GB. Prompt encode peak: ~49 GB.
- Flux 2 Klein KV: ~32 GB total bf16, fits comfortably. No offload.

LoRA handling: adapter mode (NOT fused). `load_lora_weights(adapter_name=
"user_lora")` attaches the adapter to the bf16 transformer; changing
strength is a free runtime call `pipe.set_adapters(["user_lora"],
[strength])` — no pipeline reload needed. Only LoRA file changes or
model switches trigger a full reload.
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

import config

# Ensure HF uses our NVMe cache
os.environ.setdefault("HF_HOME", "/mnt/nvme-1/huggingface")

logger = logging.getLogger(__name__)


@contextmanager
def _timed(label: str):
    """Log wall-clock elapsed for a block (phase-boundary timing)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s: %.2fs", label, time.perf_counter() - t0)


def _emit_phase(phase_sink: Callable | None, progress: float, phase: str) -> None:
    """Emit a (progress, phase) update to a caller-supplied callback."""
    if phase_sink is not None:
        try:
            phase_sink(progress, phase=phase)
        except TypeError:
            phase_sink(progress)


class FluxLoraError(ValueError):
    """Raised when a user LoRA cannot be attached as an adapter to the current model.

    Typically a client error — the LoRA's tensor shapes, keys, or format
    don't match the target pipeline. Handlers surface this as HTTP 422.
    """


# LoRA adapter name used for the single user LoRA slot. Kept as a module-level
# constant so generate methods can reference it when calling set_adapters /
# disable_lora without magic-stringing "user_lora" in multiple places.
USER_LORA_ADAPTER = "user_lora"


class FluxManager:
    """Manages Flux pipelines with on-demand model swapping."""

    def __init__(self) -> None:
        self._pipe = None
        self._current_model: str | None = None
        self._current_lora_path: str | None = None  # adapter identity only; strength is runtime
        self._device = config.FLUX_DEVICE
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._pipe is not None

    def load(self, model_name: str = "flux2-dev", user_lora_path: str | None = None) -> None:
        """Load a Flux pipeline in full bf16 with optional adapter-mode user LoRA.

        Dev uses `enable_model_cpu_offload` to page the text encoder CPU↔GPU around
        prompt encoding (bf16 all-resident is 105.9 GB, exceeds 96 GB). Klein fits
        fully resident in bf16 (~32 GB) and skips offload.

        LoRAs are attached as adapters — NOT fused. Strength is applied at inference
        time via `pipe.set_adapters([USER_LORA_ADAPTER], [strength])`, so the cache
        key is `(model_name, user_lora_path)` and strength changes are free.

        On failure, leaves the manager in a clean unloaded state (self._pipe = None)
        and frees any partially allocated GPU/CPU memory so the next request can retry.
        """
        from diffusers.models import Flux2Transformer2DModel

        model_repo = config.FLUX_MODELS[model_name]
        lora_desc = f" + adapter({Path(user_lora_path).name})" if user_lora_path else ""
        logger.info("Loading %s (%s)%s on %s ...", model_name, model_repo, lora_desc, self._device)
        t0 = time.monotonic()

        _load_kw = dict(torch_dtype=torch.bfloat16, cache_dir=config.HF_CACHE_DIR, local_files_only=True)

        pipe = None
        try:
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
                if user_lora_path:
                    self._load_user_lora_adapter(pipe, user_lora_path)
                # VAE: force float32 — AutoencoderKLFlux2._decode() silently ignores
                # its own force_upcast=True config flag. Decoder ResBlocks + GroupNorm
                # + nearest-neighbor upsample accumulate visible precision artifacts
                # in bf16. Extra VRAM cost is ~320 MB (the VAE's 321 MB params doubled).
                self._force_vae_fp32(pipe)
                # Klein fits in full bf16 on 96 GB — keep all components resident.
                pipe = pipe.to(self._device)
            else:
                from diffusers import Flux2Pipeline
                pipe = Flux2Pipeline.from_pretrained(model_repo, **_load_kw)
                if user_lora_path:
                    self._load_user_lora_adapter(pipe, user_lora_path)
                # VAE: force float32 before the offload hooks attach, so the VAE
                # component enters the offload manager already at fp32 and stays
                # fp32 across every CPU↔GPU page.
                self._force_vae_fp32(pipe)
                # Dev bf16 is 105.9 GB all-resident; use model_cpu_offload to page
                # text encoder, transformer, VAE between CPU and GPU as needed.
                # NB: do NOT call pipe.to(device) before enable_model_cpu_offload —
                # the offload hooks manage device placement themselves.
                pipe.enable_model_cpu_offload(device=self._device)

            self._pipe = pipe
            self._current_model = model_name
            self._current_lora_path = user_lora_path
        except Exception:
            # Drop any partially loaded state so the next request can retry cleanly.
            logger.exception("Flux load failed for %s%s; releasing partial state", model_name, lora_desc)
            self._pipe = None
            self._current_model = None
            self._current_lora_path = None
            del pipe
            try:
                gc.collect()
                torch.cuda.synchronize(torch.device(self._device))
                torch.cuda.empty_cache()
            except Exception:
                pass
            raise

        elapsed = time.monotonic() - t0
        logger.info("%s loaded in %.1fs on %s", model_name, elapsed, self._device)

    @staticmethod
    def _force_vae_fp32(pipe) -> None:
        """Run the VAE in full float32 regardless of pipeline dtype.

        Why: `AutoencoderKLFlux2._decode()` silently ignores the `force_upcast=True`
        config flag. Its ResBlocks + GroupNorm + nearest-neighbor Upsample2D cascade
        accumulates visible precision artifacts in bf16 (the fp32 fallback inside
        `Upsample2D.forward` only triggers on `torch.__version__ < "2.1"` — we're on
        2.11, so it's dead code). Running the VAE in fp32 is the correct fix.

        The pipeline still produces bf16 latents from the transformer denoising
        loop, so we also install a forward pre-hook on `post_quant_conv` — the
        first Conv2d in the decode path — to upcast incoming latents to fp32.
        Without that hook, the fp32 Conv2d rejects bf16 inputs with
        "Input type and bias type should be the same".
        """
        pipe.vae.to(torch.float32)

        def _upcast_input_hook(_module, args):
            if args and isinstance(args[0], torch.Tensor) and args[0].dtype != torch.float32:
                return (args[0].to(torch.float32),) + tuple(args[1:])
            return None  # no modification

        # post_quant_conv is the entry point of _decode; tiled_decode also
        # routes through it (one hook covers both tiled and non-tiled paths).
        if pipe.vae.post_quant_conv is not None:
            pipe.vae.post_quant_conv.register_forward_pre_hook(_upcast_input_hook)

    @staticmethod
    def _load_user_lora_adapter(pipe, path: str) -> None:
        """Attach a user LoRA to the pipeline as a named adapter (no fusion).

        The strength is applied at inference time via `set_adapters([name], [scale])`
        — NOT at load time. Raises FluxLoraError (a ValueError subclass) when the
        LoRA is incompatible with the target pipeline (wrong tensor dimensions,
        e.g. Dev-trained LoRA applied to Klein; missing keys; malformed safetensors).
        Handlers surface this as HTTP 422.
        """
        try:
            pipe.load_lora_weights(path, adapter_name=USER_LORA_ADAPTER)
        except (RuntimeError, KeyError, ValueError) as exc:
            # RuntimeError: torch state_dict size mismatch (wrong dims for model)
            # KeyError:     missing/extra keys during PEFT adapter injection
            # ValueError:   safetensors parse error, metadata mismatch
            raise FluxLoraError(
                f"LoRA at {Path(path).name} is incompatible with the current model: {exc}"
            ) from exc

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
            self._current_lora_path = None
            gc.collect()
            torch.cuda.synchronize(torch.device(self._device))
            torch.cuda.empty_cache()
            logger.info("Flux pipeline unloaded from %s", self._device)

    def ensure_model(self, model_name: str, user_lora_path: str | None = None) -> None:
        """Swap model if needed. Cache key is (model_name, user_lora_path).

        Strength is NOT part of the cache key — it's applied at inference time
        via `pipe.set_adapters([...], [strength])`, so changing strength never
        triggers a reload. Only model changes or LoRA file changes reload.
        """
        if self._current_model == model_name and self._current_lora_path == user_lora_path:
            return
        if self._pipe is not None:
            logger.info(
                "Swapping Flux pipeline: %s/%s → %s/%s",
                self._current_model, self._current_lora_path, model_name, user_lora_path,
            )
            self.unload()
        self.load(model_name, user_lora_path=user_lora_path)

    def _apply_lora_strength(self, lora_path: str | None, lora_strength: float) -> None:
        """Apply LoRA strength at inference time without reloading.

        Called from every generate method AFTER ensure_model() but BEFORE the
        pipeline __call__. Idempotent: safe to call with the same args across
        multiple requests. When lora_path is None, disables any active adapter.
        """
        if self._pipe is None:
            return
        if lora_path:
            # Adapter is guaranteed loaded by ensure_model() above.
            try:
                self._pipe.set_adapters([USER_LORA_ADAPTER], [lora_strength])
            except Exception:
                logger.exception("set_adapters failed for strength=%s", lora_strength)
                raise
        else:
            # No LoRA requested — disable any stale adapter from a previous request.
            try:
                self._pipe.disable_lora()
            except Exception:
                # Pipelines without any loaded adapter raise — ignore.
                pass

    def _to_webp(self, image: Image.Image) -> bytes:
        """Encode PIL image as LOSSLESS WEBP (VP8L).

        Lossy WEBP (VP8) uses YUV 4:2:0 chroma subsampling by default, which
        introduces grid/screendoor patterns on smooth gradients and flat color
        regions — exactly where users notice them first on Flux outputs (glass
        reflections, wood grain, sky). VP8L encodes RGB directly with no color
        space conversion, preserving the exact pixels the transformer produced.

        File size is ~2-3x larger than VP8 quality=95, but still smaller than
        PNG in most cases and acceptable for an image generation API.

        `method=6` selects the slowest/highest-compression encoding path;
        lossless WEBP encoding speed scales with image size but is still
        dominated by the ~20-step denoising work anyway.
        """
        buf = io.BytesIO()
        image.save(buf, format="WEBP", lossless=True, quality=100, method=6)
        return buf.getvalue()

    @torch.inference_mode()
    def _generate(
        self, prompt: str, width: int, height: int,
        num_inference_steps: int, guidance_scale: float, seed: int,
        model: str = "flux2-dev", turbo: bool = False,
        lora_path: str | None = None, lora_strength: float = 1.0,
        callback_on_step_end: object = None,
        phase_sink: Callable | None = None,
        turbo_steps: int = 8, turbo_guidance: float = 2.5,
    ) -> bytes:
        """Generate an image (txt2img) and return WEBP bytes."""
        self.ensure_model(model, user_lora_path=lora_path)
        self._apply_lora_strength(lora_path, lora_strength)
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
            kwargs["num_inference_steps"] = turbo_steps
            kwargs["guidance_scale"] = turbo_guidance

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

        # Pipeline returns with VAE decode already done — only WEBP encode remains.
        _emit_phase(phase_sink, 0.95, "encoding")
        with _timed("flux_webp_encode model=%s" % model):
            return self._to_webp(result.images[0])

    @torch.inference_mode()
    def _img2img(
        self, prompt: str, image_path: str, width: int, height: int,
        num_inference_steps: int, guidance_scale: float, seed: int,
        model: str = "flux2-dev", turbo: bool = False,
        lora_path: str | None = None, lora_strength: float = 1.0,
        callback_on_step_end: object = None,
        phase_sink: Callable | None = None,
        turbo_steps: int = 8, turbo_guidance: float = 2.5,
    ) -> bytes:
        """Edit an image using single reference."""
        self.ensure_model(model, user_lora_path=lora_path)
        self._apply_lora_strength(lora_path, lora_strength)
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
            kwargs["num_inference_steps"] = turbo_steps
            kwargs["guidance_scale"] = turbo_guidance

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

        _emit_phase(phase_sink, 0.95, "encoding")
        with _timed("flux_webp_encode model=%s" % model):
            return self._to_webp(result.images[0])

    @torch.inference_mode()
    def _edit(
        self, prompt: str, image_paths: list[str], width: int, height: int,
        num_inference_steps: int = 4, guidance_scale: float = 4.0,
        seed: int = 0,
        model: str = "flux2-klein",
        lora_path: str | None = None, lora_strength: float = 1.0,
        callback_on_step_end: object = None,
        phase_sink: Callable | None = None,
    ) -> bytes:
        """Multi-reference image editing via Dev or Klein."""
        self.ensure_model(model, user_lora_path=lora_path)
        self._apply_lora_strength(lora_path, lora_strength)
        generator = torch.Generator(device=self._device).manual_seed(seed)
        images = [Image.open(p).convert("RGB") for p in image_paths]

        kwargs: dict = dict(
            image=images, prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        # Klein KV pipeline is distilled and doesn't accept guidance_scale.
        if model != "flux2-klein":
            kwargs["guidance_scale"] = guidance_scale
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

        _emit_phase(phase_sink, 0.95, "encoding")
        with _timed("flux_webp_encode model=%s-edit" % model):
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
                None, lambda: self._edit(model=model, **kwargs),
            )
