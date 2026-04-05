"""Split-GPU model manager for LTX-2 video generation.

Dual-denoiser architecture: GPU:0 holds shared text encoder + embeddings
processor + audio encoder AND its own transformer + decoders. GPU:1 holds
only transformer + decoders. Text encoding is serialized on GPU:0; denoising
runs concurrently on both GPUs. Each GPU manages its own transformer swap
state independently.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from ltx_core.components.diffusion_steps import EulerDiffusionStep, Res2sDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuider,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, decode_video as vae_decode_video, get_video_chunks_number
from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig
from ltx_core.tools import VideoLatentShape
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    DISTILLED_SIGMA_VALUES,
    LTX_2_3_HQ_PARAMS,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from ltx_pipelines.retake import TemporalRegionMask
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    denoise_video_only,
    encode_prompts,
    multi_modal_guider_denoising_func,
    multi_modal_guider_factory_denoising_func,
    noise_audio_state,
    noise_video_state,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    encode_video,
    get_videostream_metadata,
    load_video_conditioning,
)
from ltx_pipelines.utils.model_ledger import ModelLedger
from ltx_pipelines.utils.samplers import (
    euler_denoising_loop,
    gradient_estimating_euler_denoising_loop,
    res2s_audio_video_denoising_loop,
)

_USE_GE_EULER = os.environ.get("USE_GE_EULER", "").lower() in ("1", "true", "yes")
from ltx_pipelines.utils.types import PipelineComponents

import config

logger = logging.getLogger(__name__)

# Pre-compute dev checkpoint params once at import time (avoids repeated disk I/O)
_DEV_PARAMS = detect_params(config.DEV_CHECKPOINT)

# skip_step=0: full STG on every step. skip_step=1 saved 33% NFE but created
# temporal oscillation in guidance signal, contributing to ghost trails during fast motion.
from dataclasses import replace as _replace

# ---------------------------------------------------------------------------
# CachingModelLedger — returns pre-loaded models instead of disk I/O
# ---------------------------------------------------------------------------


class CachingModelLedger:
    """Drop-in for ModelLedger that returns cached model instances.

    When pipeline code calls ``ledger.transformer()`` then ``del transformer``,
    the cached reference keeps the model alive on GPU.  ``cleanup_memory()``
    frees allocator cache but not the weights themselves.

    Models not pre-cached (decoders, upsampler, vocoder) are loaded on-demand
    from ``_source_ledger`` to keep baseline VRAM low.
    """

    def __init__(self, device: torch.device, cache: dict[str, object],
                 source_ledger: object = None) -> None:
        self.device = device
        self.dtype = torch.bfloat16
        self._cache = cache
        self._source_ledger = source_ledger

    def _lazy(self, key: str, loader_name: str):
        val = self._cache.get(key)
        if val is None and self._source_ledger is not None:
            logger.info("Lazy-loading %s on %s", key, self.device)
            val = getattr(self._source_ledger, loader_name)()
            self._cache[key] = val
        return val

    def text_encoder(self):
        return self._cache["text_encoder"]

    def gemma_embeddings_processor(self):
        return self._cache["embeddings_processor"]

    def video_encoder(self):
        return self._lazy("video_encoder", "video_encoder")

    def audio_encoder(self):
        return self._cache["audio_encoder"]

    def transformer(self):
        return self._cache.get("transformer")

    def spatial_upsampler(self):
        return self._lazy("spatial_upsampler", "spatial_upsampler")

    def video_decoder(self):
        return self._lazy("video_decoder", "video_decoder")

    def audio_decoder(self):
        return self._lazy("audio_decoder", "audio_decoder")

    def vocoder(self):
        return self._lazy("vocoder", "vocoder")


# ---------------------------------------------------------------------------
# DenoiserWorker — per-GPU denoiser state
# ---------------------------------------------------------------------------


@dataclass
class DenoiserWorker:
    """Per-GPU worker with its own transformer, decoders, and swap state."""
    device: torch.device
    ledger: CachingModelLedger
    components: PipelineComponents
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    transformer_state: str = ""
    _user_lora: tuple[str, float] | None = None
    transformer: object = None
    cache: dict[str, object] = field(default_factory=dict)
    _model_ledger: object = None  # ModelLedger for lazy-loading decoders/upsampler

    def evict_transformer(self) -> None:
        """Remove transformer from GPU to free VRAM for heavy operations like VAE encode."""
        if self.transformer is not None:
            logger.info("Worker %s: evicting transformer (%s) to free VRAM", self.device, self.transformer_state)
            self.transformer = None
            self.cache["transformer"] = None
            self.ledger._cache["transformer"] = None
            self.transformer_state = ""
            self._user_lora = None
            import gc; gc.collect()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    def ensure_transformer(self, state: str, user_lora: tuple[str, float] | None = None) -> None:
        """Swap transformer checkpoint on this worker's GPU.

        Args:
            state: Base transformer state (dev, distilled, dev_lora, etc.)
            user_lora: Optional (path, strength) for a user-supplied LoRA to fuse
                       alongside any preset LoRAs for this state.
        """
        if self.transformer_state == state and self._user_lora == user_lora:
            return

        logger.info("Worker %s: swapping transformer %s -> %s (user_lora=%s)", self.device, self.transformer_state, state, user_lora)
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
        elif state == "dev_lora_025":
            distilled_lora = LoraPathStrengthAndSDOps(
                path=config.DISTILLED_LORA, strength=0.25,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
            checkpoint, loras = config.DEV_CHECKPOINT, (distilled_lora,)
        elif state == "dev_lora_050":
            distilled_lora = LoraPathStrengthAndSDOps(
                path=config.DISTILLED_LORA, strength=0.5,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
            checkpoint, loras = config.DEV_CHECKPOINT, (distilled_lora,)
        elif state == "dev_lora_020":
            distilled_lora = LoraPathStrengthAndSDOps(
                path=config.DISTILLED_LORA, strength=0.2,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
            checkpoint, loras = config.DEV_CHECKPOINT, (distilled_lora,)
        else:
            raise ValueError(f"Unknown transformer state: {state}")

        # Append user-supplied LoRA if provided
        if user_lora:
            path, strength = user_lora
            loras = loras + (LoraPathStrengthAndSDOps(
                path=path, strength=strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ),)

        # Free old transformer BEFORE loading new one to avoid OOM
        # Must clear BOTH self.transformer AND ledger._cache (separate dicts!)
        old = self.transformer
        self.transformer = None
        self.cache["transformer"] = None
        self.ledger._cache["transformer"] = None
        del old
        import gc; gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        ledger = ModelLedger(
            dtype=torch.bfloat16, device=self.device,
            checkpoint_path=checkpoint, gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER, loras=loras,
        )
        new_transformer = ledger.transformer()
        self.transformer = new_transformer
        self.cache["transformer"] = new_transformer
        self.ledger._cache["transformer"] = new_transformer
        self.transformer_state = state
        self._user_lora = user_lora
        logger.info("Worker %s: transformer now %s", self.device, state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DECODE_TILING = TilingConfig(
    spatial_config=None,
    temporal_config=TemporalTilingConfig(tile_size_in_frames=128, tile_overlap_in_frames=32),
)

# Skip tiling for videos ≤257 frames (~10s at 24fps). Single-pass decode = no tile
# boundary artifacts at all. cuDNN 9.20 fixes the conv3d workspace bug so this fits
# in ~90GB available after transformer eviction. Only videos >10s need tiling.
SHORT_VIDEO_THRESHOLD = 257

# Gradient estimating euler: momentum-accelerated sampler, potentially 30→20 steps
# Enable via USE_GE_EULER=1 in .env for A/B testing
_euler_loop = gradient_estimating_euler_denoising_loop if _USE_GE_EULER else euler_denoising_loop

# Stage 2: 5 steps instead of upstream's 3. More steps resolve fast motion better —
# 3 steps from 91% noise can't fully reconstruct temporal detail during fast motion.
# Original: [0.909375, 0.725, 0.421875, 0.0] (3 steps)
_STAGE_2_SIGMAS = [0.909375, 0.727, 0.546, 0.364, 0.182, 0.0]  # 5 steps


def _get_decode_tiling(num_frames: int) -> TilingConfig | None:
    """Skip tiling for short videos to avoid temporal boundary artifacts."""
    return None if num_frames <= SHORT_VIDEO_THRESHOLD else DECODE_TILING


def _decode_video_fp32(latent: torch.Tensor, decoder, tiling, generator) -> Iterator[torch.Tensor]:
    """Decode video in bfloat16 (standard)."""
    yield from vae_decode_video(latent, decoder, tiling, generator)


def _video_to_bytes(video: Iterator[torch.Tensor], fps: float, audio: Audio, num_frames: int, *, include_audio: bool = True) -> bytes:
    # Can't use BytesIO here: encode_video calls av.open(path, mode="w") without
    # format= kwarg, so PyAV can't infer the container format from a file-like object.
    video_chunks_number = get_video_chunks_number(num_frames, _get_decode_tiling(num_frames))
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        encode_video(
            video=video,
            fps=int(fps),
            audio=audio if include_audio else None,
            output_path=tmp_path,
            video_chunks_number=video_chunks_number,
        )
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SplitModelManager
# ---------------------------------------------------------------------------


class SplitModelManager:
    """Dual-denoiser manager: GPU:0 = shared encoder + denoiser, GPU:1 = denoiser.

    Text encoding serialized on GPU:0. Both GPUs denoise concurrently.
    """

    def __init__(self) -> None:
        self._workers: list[DenoiserWorker] = []
        self._encoder_device: torch.device | None = None
        self._encoder_ledger: CachingModelLedger | None = None
        # No explicit encode lock needed — CUDA serializes ops on the default
        # stream per-device. Text encoding (0.4s) may wait at most 1 denoising
        # step (~1.5s) when GPU:0 is concurrently denoising. Acceptable tradeoff.

    @property
    def is_ready(self) -> bool:
        return len(self._workers) > 0

    def evict_all(self) -> None:
        """Free all GPU memory — evict transformer + encoder hub on all workers."""
        for worker in self._workers:
            worker.evict_transformer()
            # Also evict decoders/encoders in worker cache
            for key in list(worker.cache.keys()):
                worker.cache[key] = None
        self._workers.clear()
        if self._encoder_ledger is not None:
            for key in list(self._encoder_ledger._cache.keys()):
                self._encoder_ledger._cache[key] = None
            self._encoder_ledger = None
        gc.collect()
        for device_name in config.GPU_DEVICES:
            torch.cuda.synchronize(torch.device(device_name))
            torch.cuda.empty_cache()
        logger.info("All LTX models evicted from GPU")

    def load_all(self) -> None:
        self._workers.clear()
        devices = [torch.device(d) for d in config.GPU_DEVICES]

        # --- Shared encoder hub on GPU:0 ---
        self._encoder_device = devices[0]
        logger.info("Loading shared encoder hub on %s ...", devices[0])
        enc_ledger = ModelLedger(
            dtype=torch.bfloat16, device=devices[0],
            checkpoint_path=config.DEV_CHECKPOINT,
            gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
            loras=(),
        )
        enc_video_encoder = enc_ledger.video_encoder()  # kept for retake conditioning
        encoder_cache = {
            "text_encoder": enc_ledger.text_encoder(),
            "embeddings_processor": enc_ledger.gemma_embeddings_processor(),
            "video_encoder": enc_video_encoder,
            "audio_encoder": enc_ledger.audio_encoder(),
        }
        self._encoder_ledger = CachingModelLedger(devices[0], encoder_cache)

        # --- Denoiser worker on each GPU ---
        # Only pre-load transformer + video_encoder on GPU. Decoders/upsampler
        # are loaded on-demand (after transformer is evicted, freeing ~44GB).
        # This keeps baseline at ~50GB instead of ~70GB, leaving room for inference.
        for device in devices:
            logger.info("Loading denoiser worker on %s ...", device)
            den_ledger = ModelLedger(
                dtype=torch.bfloat16, device=device,
                checkpoint_path=config.DEV_CHECKPOINT,
                gemma_root_path=config.GEMMA_ROOT,
                spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
                loras=(),
            )
            transformer = den_ledger.transformer()
            # GPU:0 reuses encoder hub's video_encoder to avoid ~5GB duplication
            vid_enc = enc_video_encoder if device == devices[0] else den_ledger.video_encoder()
            cache = {
                "transformer": transformer,
                "video_encoder": vid_enc,
            }
            worker = DenoiserWorker(
                device=device,
                ledger=CachingModelLedger(device, cache, source_ledger=den_ledger),
                _model_ledger=den_ledger,
                components=PipelineComponents(dtype=torch.bfloat16, device=device),
                transformer_state="dev",
                transformer=transformer,
                cache=cache,
            )
            self._workers.append(worker)
            logger.info("Denoiser worker ready on %s", device)

        logger.info("All models loaded: %d workers, encoder on %s", len(self._workers), self._encoder_device)

    # --- Worker acquisition ---

    async def _acquire_worker(self) -> DenoiserWorker:
        """Wait for and return the first unlocked worker."""
        while True:
            for worker in self._workers:
                if not worker.lock.locked():
                    await worker.lock.acquire()
                    return worker
            await asyncio.sleep(0.05)

    # --- Context transfer ---

    def _contexts_to_device(self, contexts, target: torch.device):
        """Move EmbeddingsProcessorOutput tensors to target device."""
        from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
        return [
            EmbeddingsProcessorOutput(
                video_encoding=ctx.video_encoding.to(target),
                audio_encoding=ctx.audio_encoding.to(target) if ctx.audio_encoding is not None else None,
                attention_mask=ctx.attention_mask.to(target),
            )
            for ctx in contexts
        ]

    # --- Generation flows ---

    @staticmethod
    def _wrap_denoise(denoise_fn, on_progress, total_steps, offset=0.0, scale=1.0):
        """Wrap a denoise_fn to report progress on each step."""
        step_count = [0]  # mutable counter for closure

        def wrapped(*args, **kwargs):
            result = denoise_fn(*args, **kwargs)
            step_count[0] += 1
            p = offset + min(step_count[0] / max(total_steps, 1), 1.0) * scale
            on_progress(min(p, 0.99))
            return result
        return wrapped

    @torch.inference_mode()
    def _run_t2v(
        self, worker: DenoiserWorker, prompt: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool,
        on_progress=None, user_lora=None, enhance_prompt: bool = False,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        is_fast = model == "ltx-2-3-fast"

        # For fast model: start with dev+low LoRA for first pass (Reddit audio fix)
        worker.ensure_transformer("dev_lora_020" if is_fast else "dev", user_lora=user_lora)

        # Text encoding on GPU:0 (shared encoder)
        if is_fast:
            (ctx_p,) = encode_prompts([prompt], self._encoder_ledger, enhance_first_prompt=enhance_prompt)
            (ctx_p,) = self._contexts_to_device([ctx_p], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger, enhance_first_prompt=enhance_prompt)
            ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Stage 1: half-resolution denoising
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = worker.ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=[], height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        transformer = worker.ledger.transformer()

        if is_fast:
            sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device, dtype=torch.float32)
            split_at = 4  # Reddit audio fix: split schedule at step 4
            s1_steps = len(sigmas) - 1

            def denoising_loop(sigmas, video_state, audio_state, stepper):
                nonlocal transformer
                # Pass 1: dev + LoRA 0.2, simple denoise (no CFG — saves 2-3x memory)
                sigmas_1 = sigmas[:split_at + 1]
                dfn_1 = simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer)
                if on_progress:
                    dfn_1 = self._wrap_denoise(dfn_1, on_progress, s1_steps, offset=0.0, scale=0.35)
                video_state, audio_state = euler_denoising_loop(sigmas=sigmas_1, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn_1)

                # Free pass 1 model + activations before loading pass 2
                del dfn_1
                transformer = None
                worker.evict_transformer()
                gc.collect()
                torch.cuda.empty_cache()

                # Swap to distilled for pass 2
                worker.ensure_transformer("distilled", user_lora=user_lora)
                transformer = worker.ledger.transformer()

                # Pass 2: distilled with simple denoise (last 4 steps)
                sigmas_2 = sigmas[split_at:]
                dfn_2 = simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer)
                if on_progress:
                    dfn_2 = self._wrap_denoise(dfn_2, on_progress, s1_steps, offset=0.35, scale=0.35)
                return euler_denoising_loop(sigmas=sigmas_2, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn_2)
        else:
            params = _DEV_PARAMS
            empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
            sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=params.num_inference_steps).to(dtype=torch.float32, device=device)
            s1_steps = len(sigmas) - 1

            def denoising_loop(sigmas, video_state, audio_state, stepper):
                dfn = multi_modal_guider_factory_denoising_func(
                    video_guider_factory=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n),
                    audio_guider_factory=create_multimodal_guider_factory(params=params.audio_guider_params, negative_context=a_context_n),
                    v_context=v_context_p, a_context=a_context_p, transformer=transformer)
                if on_progress:
                    dfn = self._wrap_denoise(dfn, on_progress, s1_steps, offset=0.0, scale=0.7)
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_shape, conditionings=stage_1_cond, noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
            components=worker.components, dtype=dtype, device=device,
        )

        # Free transformer (~22GB) before upsample — not needed for upsample_video
        stage_1_latent = video_state.latent[:1]
        stage_1_audio_latent = audio_state.latent
        del video_state, audio_state, stage_1_cond, sigmas, transformer, denoising_loop
        cleanup_memory()

        upscaled = upsample_video(latent=stage_1_latent, video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        del stage_1_latent

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=[], height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            worker.ensure_transformer("dev_lora", user_lora=user_lora)

        transformer = worker.ledger.transformer()
        distilled_sigmas = torch.tensor(_STAGE_2_SIGMAS, device=device, dtype=torch.float32)
        s2_steps = len(distilled_sigmas) - 1

        def stage2_loop(sigmas, video_state, audio_state, stepper):
            dfn = simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer)
            if on_progress:
                dfn = self._wrap_denoise(dfn, on_progress, s2_steps, offset=0.7, scale=0.25)
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=worker.components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=stage_1_audio_latent,
        )

        # Free everything before decode — evict transformer + cached models to reclaim VRAM
        video_latent = video_state.latent
        audio_latent = audio_state.latent
        del video_state, audio_state, stage_2_cond, upscaled, stage_1_audio_latent
        del transformer, stage2_loop, distilled_sigmas, video_encoder
        worker.evict_transformer()
        for k in ("spatial_upsampler", "video_encoder"):
            worker.cache[k] = None
            worker.ledger._cache[k] = None
        gc.collect()
        torch.cuda.empty_cache()

        # Decode (decoders lazy-load here, with transformer/upsampler/encoder freed)
        decoded_video = _decode_video_fp32(video_latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        decoded_audio = vae_decode_audio(audio_latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        return _video_to_bytes(decoded_video, fps, decoded_audio, num_frames, include_audio=generate_audio)

    @torch.inference_mode()
    def _run_t2v_hq(
        self, worker: DenoiserWorker, prompt: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool,
        on_progress=None, user_lora=None, enhance_prompt: bool = False,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        hq_params = LTX_2_3_HQ_PARAMS

        worker.ensure_transformer("dev_lora_025", user_lora=user_lora)

        # Text encoding on GPU:0 (shared encoder)
        ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger, enhance_first_prompt=enhance_prompt)
        ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = Res2sDiffusionStep()

        # Stage 1: half-resolution denoising with res2s sampler
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = worker.ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=[], height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        transformer = worker.ledger.transformer()

        empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
        sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=hq_params.num_inference_steps).to(dtype=torch.float32, device=device)
        # res2s: 2 NFE per step + 1 final
        s1_nfe = 2 * hq_params.num_inference_steps + 1

        def denoising_loop(sigmas, video_state, audio_state, stepper):
            dfn = multi_modal_guider_denoising_func(
                video_guider=MultiModalGuider(params=hq_params.video_guider_params, negative_context=v_context_n),
                audio_guider=MultiModalGuider(params=hq_params.audio_guider_params, negative_context=a_context_n),
                v_context=v_context_p, a_context=a_context_p, transformer=transformer,
            )
            if on_progress:
                dfn = self._wrap_denoise(dfn, on_progress, s1_nfe, offset=0.0, scale=0.7)
            return res2s_audio_video_denoising_loop(
                sigmas=sigmas, video_state=video_state, audio_state=audio_state,
                stepper=stepper, denoise_fn=dfn,
            )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_shape, conditionings=stage_1_cond, noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
            components=worker.components, dtype=dtype, device=device,
        )

        # Free transformer before upsample — not needed for upsample_video
        stage_1_latent = video_state.latent[:1]
        stage_1_audio_latent = audio_state.latent
        del video_state, audio_state, stage_1_cond, sigmas, transformer, denoising_loop
        cleanup_memory()

        upscaled = upsample_video(latent=stage_1_latent, video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        del stage_1_latent

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=[], height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        worker.ensure_transformer("dev_lora_050", user_lora=user_lora)
        transformer = worker.ledger.transformer()
        distilled_sigmas = torch.tensor(_STAGE_2_SIGMAS, device=device, dtype=torch.float32)
        # res2s stage 2: 2 NFE per step + 1 final (3 distilled steps = 2 actual steps)
        s2_nfe = 2 * (len(distilled_sigmas) - 1) + 1

        def stage2_loop(sigmas, video_state, audio_state, stepper):
            dfn = simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer)
            if on_progress:
                dfn = self._wrap_denoise(dfn, on_progress, s2_nfe, offset=0.7, scale=0.25)
            return res2s_audio_video_denoising_loop(
                sigmas=sigmas, video_state=video_state, audio_state=audio_state,
                stepper=stepper, denoise_fn=dfn,
            )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=worker.components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=stage_1_audio_latent,
        )

        # Free everything before decode — evict transformer + cached models to reclaim VRAM
        video_latent = video_state.latent
        audio_latent = audio_state.latent
        del video_state, audio_state, stage_2_cond, upscaled, stage_1_audio_latent
        del transformer, stage2_loop, distilled_sigmas, video_encoder
        worker.evict_transformer()
        for k in ("spatial_upsampler", "video_encoder"):
            worker.cache[k] = None
            worker.ledger._cache[k] = None
        gc.collect()
        torch.cuda.empty_cache()

        # Decode (decoders lazy-load here, with transformer/upsampler/encoder freed)
        decoded_video = _decode_video_fp32(video_latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        decoded_audio = vae_decode_audio(audio_latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        return _video_to_bytes(decoded_video, fps, decoded_audio, num_frames, include_audio=generate_audio)

    @torch.inference_mode()
    def _run_i2v(
        self, worker: DenoiserWorker, prompt: str, keyframes: list[dict], model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool,
        on_progress=None, user_lora=None, enhance_prompt: bool = False,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        is_fast = model == "ltx-2-3-fast"
        images = [
            ImageConditioningInput(path=kf["image_path"], frame_idx=kf["frame_index"], strength=kf["strength"])
            for kf in keyframes
        ]

        worker.ensure_transformer("distilled" if is_fast else "dev", user_lora=user_lora)

        # Text encoding on GPU:0 (shared encoder)
        if is_fast:
            (ctx_p,) = encode_prompts([prompt], self._encoder_ledger, enhance_first_prompt=enhance_prompt)
            (ctx_p,) = self._contexts_to_device([ctx_p], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger, enhance_first_prompt=enhance_prompt)
            ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Stage 1
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = worker.ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=images, height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)
        transformer = worker.ledger.transformer()

        if is_fast:
            sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device, dtype=torch.float32)
            s1_steps = len(sigmas) - 1

            def denoising_loop(sigmas, video_state, audio_state, stepper):
                dfn = simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer)
                if on_progress:
                    dfn = self._wrap_denoise(dfn, on_progress, s1_steps, offset=0.0, scale=0.7)
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn)
        else:
            params = _DEV_PARAMS
            empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
            sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=params.num_inference_steps).to(dtype=torch.float32, device=device)
            s1_steps = len(sigmas) - 1

            def denoising_loop(sigmas, video_state, audio_state, stepper):
                dfn = multi_modal_guider_factory_denoising_func(
                    video_guider_factory=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n),
                    audio_guider_factory=create_multimodal_guider_factory(params=params.audio_guider_params, negative_context=a_context_n),
                    v_context=v_context_p, a_context=a_context_p, transformer=transformer)
                if on_progress:
                    dfn = self._wrap_denoise(dfn, on_progress, s1_steps, offset=0.0, scale=0.7)
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_shape, conditionings=stage_1_cond, noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
            components=worker.components, dtype=dtype, device=device,
        )

        # Free transformer before upsample — not needed for upsample_video
        stage_1_latent = video_state.latent[:1]
        stage_1_audio_latent = audio_state.latent
        del video_state, audio_state, stage_1_cond, sigmas, transformer, denoising_loop
        cleanup_memory()

        upscaled = upsample_video(latent=stage_1_latent, video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        del stage_1_latent

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=images, height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            worker.ensure_transformer("dev_lora", user_lora=user_lora)

        transformer = worker.ledger.transformer()
        distilled_sigmas = torch.tensor(_STAGE_2_SIGMAS, device=device, dtype=torch.float32)
        s2_steps = len(distilled_sigmas) - 1

        def stage2_loop(sigmas, video_state, audio_state, stepper):
            dfn = simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer)
            if on_progress:
                dfn = self._wrap_denoise(dfn, on_progress, s2_steps, offset=0.7, scale=0.25)
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=worker.components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=stage_1_audio_latent,
        )

        # Free everything before decode — evict transformer to reclaim ~22GB VRAM
        video_latent = video_state.latent
        audio_latent = audio_state.latent
        del video_state, audio_state, stage_2_cond, upscaled, stage_1_audio_latent
        del transformer, stage2_loop, distilled_sigmas
        worker.evict_transformer()
        for k in ("spatial_upsampler", "video_encoder"):
            worker.cache[k] = None
            worker.ledger._cache[k] = None
        gc.collect()
        torch.cuda.empty_cache()

        decoded_video = _decode_video_fp32(video_latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        decoded_audio = vae_decode_audio(audio_latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        return _video_to_bytes(decoded_video, fps, decoded_audio, num_frames, include_audio=generate_audio)

    @torch.inference_mode()
    def _run_a2v(
        self, worker: DenoiserWorker, prompt: str, audio_path: str, image_path: str | None,
        width: int, height: int, num_frames: int, fps: float, seed: int,
        on_progress=None, user_lora=None, enhance_prompt: bool = False,
        model: str = "ltx-2-3-pro",
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        is_fast = model == "ltx-2-3-fast"

        worker.ensure_transformer("distilled" if is_fast else "dev", user_lora=user_lora)

        # Text encoding on GPU:0 (shared encoder)
        if is_fast:
            (ctx_p,) = encode_prompts([prompt], self._encoder_ledger, enhance_first_prompt=enhance_prompt)
            (ctx_p,) = self._contexts_to_device([ctx_p], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger, enhance_first_prompt=enhance_prompt)
            ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Audio encoding on GPU:0, then transfer to worker device
        decoded_audio = decode_audio_from_file(audio_path, self._encoder_device, max_duration=num_frames / fps)
        encoded_audio_latent = vae_encode_audio(decoded_audio, self._encoder_ledger.audio_encoder())
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / fps, channels=8, mel_bins=16)
        encoded_audio_latent = encoded_audio_latent[:, :, :audio_shape.frames].to(device)

        images: list[ImageConditioningInput] = []
        if image_path is not None:
            images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Stage 1: video-only denoising (audio frozen)
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = worker.ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=images, height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)
        transformer = worker.ledger.transformer()

        if is_fast:
            sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device, dtype=torch.float32)
            s1_steps = len(sigmas) - 1

            def stage1_loop(sigmas, video_state, audio_state, stepper):
                dfn = simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer)
                if on_progress:
                    dfn = self._wrap_denoise(dfn, on_progress, s1_steps, offset=0.0, scale=0.7)
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn)
        else:
            params = _DEV_PARAMS
            empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
            sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=params.num_inference_steps).to(dtype=torch.float32, device=device)
            s1_steps = len(sigmas) - 1

            def stage1_loop(sigmas, video_state, audio_state, stepper):
                dfn = multi_modal_guider_factory_denoising_func(
                    video_guider_factory=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n),
                    audio_guider_factory=create_multimodal_guider_factory(params=MultiModalGuiderParams(), negative_context=None),
                    v_context=v_context_p, a_context=a_context_p, transformer=transformer)
                if on_progress:
                    dfn = self._wrap_denoise(dfn, on_progress, s1_steps, offset=0.0, scale=0.7)
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn)

        video_state = denoise_video_only(
            output_shape=stage_1_shape, conditionings=stage_1_cond, noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=stage1_loop,
            components=worker.components, dtype=dtype, device=device,
            initial_audio_latent=encoded_audio_latent,
        )

        # Free transformer before upsample — not needed for upsample_video
        stage_1_latent = video_state.latent[:1]
        del video_state, stage_1_cond, sigmas, transformer, stage1_loop
        cleanup_memory()

        upscaled = upsample_video(latent=stage_1_latent, video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        del stage_1_latent

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=images, height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            worker.ensure_transformer("dev_lora", user_lora=user_lora)
        transformer = worker.ledger.transformer()
        distilled_sigmas = torch.tensor(_STAGE_2_SIGMAS, device=device, dtype=torch.float32)
        s2_steps = len(distilled_sigmas) - 1

        def stage2_loop(sigmas, video_state, audio_state, stepper):
            dfn = simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer)
            if on_progress:
                dfn = self._wrap_denoise(dfn, on_progress, s2_steps, offset=0.7, scale=0.25)
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn)

        video_state = denoise_video_only(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=worker.components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=encoded_audio_latent,
        )

        # Free everything before decode — evict transformer to reclaim ~22GB VRAM
        video_latent = video_state.latent
        del video_state, stage_2_cond, upscaled
        del transformer, stage2_loop, distilled_sigmas
        worker.evict_transformer()
        for k in ("spatial_upsampler", "video_encoder"):
            worker.cache[k] = None
            worker.ledger._cache[k] = None
        gc.collect()
        torch.cuda.empty_cache()

        # Decode video but return ORIGINAL audio (a2v passthrough)
        decoded_video = _decode_video_fp32(video_latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        original_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)
        return _video_to_bytes(decoded_video, fps, original_audio, num_frames)

    @torch.inference_mode()
    def _run_retake(
        self, worker: DenoiserWorker, video_path: str, start_time: float, duration: float,
        mode: str, prompt: str, seed: int,
        on_progress=None, user_lora=None,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        end_time = start_time + duration
        regenerate_video = mode in ("replace_audio_and_video", "replace_video", "replace_video_only")
        regenerate_audio = mode in ("replace_audio_and_video", "replace_audio")

        # Get video metadata
        fps_vid, num_frames, vid_width, vid_height = get_videostream_metadata(video_path)

        # Text encoding on GPU:0 (shared encoder)
        params = _DEV_PARAMS
        ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
        ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Evict transformer (~44GB) to make room for VAE encode (~46GB intermediates)
        worker.evict_transformer()
        gc.collect()

        # Encode input video on GPU:0, transfer to worker device
        video_encoder_enc = self._encoder_ledger.video_encoder()
        video_conditioning = load_video_conditioning(video_path, height=vid_height, width=vid_width, frame_cap=num_frames, dtype=dtype, device=self._encoder_device)
        initial_video_latent = video_encoder_enc(video_conditioning.to(self._encoder_device, dtype=dtype)).to(device)
        del video_conditioning
        cleanup_memory()

        # Encode audio from video (may be None if no audio track)
        audio_in = decode_audio_from_file(video_path, self._encoder_device, max_duration=num_frames / fps_vid)
        initial_audio_latent = None
        output_shape = VideoPixelShape(batch=1, frames=num_frames, width=vid_width, height=vid_height, fps=fps_vid)
        if audio_in is not None:
            audio_encoder = self._encoder_ledger.audio_encoder()
            initial_audio_latent = vae_encode_audio(audio_in, audio_encoder).to(device)
            # Trim/pad audio latent to match expected frame count from output shape
            expected_frames = AudioLatentShape.from_video_pixel_shape(output_shape).frames
            actual_frames = initial_audio_latent.shape[2]
            if actual_frames > expected_frames:
                initial_audio_latent = initial_audio_latent[:, :, :expected_frames, :]
            elif actual_frames < expected_frames:
                pad = torch.zeros(
                    initial_audio_latent.shape[0], initial_audio_latent.shape[1],
                    expected_frames - actual_frames, initial_audio_latent.shape[3],
                    device=initial_audio_latent.device, dtype=initial_audio_latent.dtype,
                )
                initial_audio_latent = torch.cat([initial_audio_latent, pad], dim=2)

        # Reload transformer now that VAE encode is done
        worker.ensure_transformer("dev", user_lora=user_lora)

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(output_shape).to_torch_shape())
        sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=params.num_inference_steps).to(dtype=torch.float32, device=device)
        total_steps = len(sigmas) - 1

        transformer = worker.ledger.transformer()

        # Build denoising function with guiders (retake is single-stage, maps 0-0.95)
        def retake_loop(sigmas, video_state, audio_state, stepper):
            dfn = multi_modal_guider_denoising_func(
                video_guider=MultiModalGuider(params=params.video_guider_params, negative_context=v_context_n),
                audio_guider=MultiModalGuider(params=params.audio_guider_params, negative_context=a_context_n),
                v_context=v_context_p, a_context=a_context_p, transformer=transformer)
            if on_progress:
                dfn = self._wrap_denoise(dfn, on_progress, total_steps, offset=0.0, scale=0.95)
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=dfn)

        # Build temporal conditionings: mask=1 inside [start, end) for regen, mask=0 for preserve
        video_conditionings = [
            TemporalRegionMask(
                start_time=start_time if regenerate_video else 0.0,
                end_time=end_time if regenerate_video else 0.0,
                fps=fps_vid,
            )
        ]
        audio_conditionings = []
        if audio_in is not None:
            audio_conditionings = [
                TemporalRegionMask(
                    start_time=start_time if regenerate_audio else 0.0,
                    end_time=end_time if regenerate_audio else 0.0,
                    fps=fps_vid,
                )
            ]

        # Noise video and audio states separately with per-modality temporal masks
        video_state, video_tools = noise_video_state(
            output_shape=output_shape, noiser=noiser,
            conditionings=video_conditionings,
            components=worker.components, dtype=dtype, device=device,
            initial_latent=initial_video_latent,
        )
        audio_state, audio_tools = noise_audio_state(
            output_shape=output_shape, noiser=noiser,
            conditionings=audio_conditionings,
            components=worker.components, dtype=dtype, device=device,
            initial_latent=initial_audio_latent,
        )

        video_state, audio_state = retake_loop(sigmas, video_state, audio_state, stepper)

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        # Evict transformer (~22GB) before VAE decode to avoid OOM
        del transformer
        worker.evict_transformer()
        gc.collect()

        decoded_video = _decode_video_fp32(video_state.latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        decoded_audio = vae_decode_audio(audio_state.latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        return _video_to_bytes(decoded_video, fps_vid, decoded_audio, num_frames)

    # --- Async API (matches PipelineManager interface) ---

    async def generate_text_to_video(
        self, prompt: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool = True,
        on_progress=None, lora_path: str | None = None, lora_strength: float = 1.0,
        enhance_prompt: bool = False,
    ) -> bytes:
        worker = await self._acquire_worker()
        user_lora = (lora_path, lora_strength) if lora_path else None
        try:
            loop = asyncio.get_running_loop()
            if model == "ltx-2-3-hq":
                return await loop.run_in_executor(
                    None, self._run_t2v_hq, worker, prompt, width, height,
                    num_frames, fps, seed, generate_audio, on_progress, user_lora,
                    enhance_prompt,
                )
            return await loop.run_in_executor(
                None, self._run_t2v, worker, prompt, model, width, height,
                num_frames, fps, seed, generate_audio, on_progress, user_lora,
                enhance_prompt,
            )
        finally:
            worker.lock.release()

    async def generate_image_to_video(
        self, prompt: str, keyframes: list[dict], model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool = True,
        on_progress=None, lora_path: str | None = None, lora_strength: float = 1.0,
        enhance_prompt: bool = False,
    ) -> bytes:
        worker = await self._acquire_worker()
        user_lora = (lora_path, lora_strength) if lora_path else None
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_i2v, worker, prompt, keyframes, model, width, height,
                num_frames, fps, seed, generate_audio, on_progress, user_lora,
                enhance_prompt,
            )
        finally:
            worker.lock.release()

    async def generate_audio_to_video(
        self, prompt: str, audio_path: str, image_path: str | None,
        model: str, width: int, height: int, num_frames: int, fps: float, seed: int,
        on_progress=None, lora_path: str | None = None, lora_strength: float = 1.0,
        enhance_prompt: bool = False,
    ) -> bytes:
        worker = await self._acquire_worker()
        user_lora = (lora_path, lora_strength) if lora_path else None
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_a2v, worker, prompt, audio_path, image_path,
                width, height, num_frames, fps, seed, on_progress, user_lora,
                enhance_prompt, model,
            )
        finally:
            worker.lock.release()

    async def retake(
        self, video_path: str, start_time: float, duration: float,
        mode: str, prompt: str, seed: int,
        on_progress=None, lora_path: str | None = None, lora_strength: float = 1.0,
    ) -> bytes:
        worker = await self._acquire_worker()
        user_lora = (lora_path, lora_strength) if lora_path else None
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_retake, worker, video_path, start_time, duration,
                mode, prompt, seed, on_progress, user_lora,
            )
        finally:
            worker.lock.release()
