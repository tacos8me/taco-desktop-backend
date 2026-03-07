"""Split-GPU model manager for LTX-2 video generation.

Dual-denoiser architecture: GPU:0 holds shared text encoder + embeddings
processor + audio encoder AND its own transformer + decoders. GPU:1 holds
only transformer + decoders. Text encoding is serialized on GPU:0; denoising
runs concurrently on both GPUs. Each GPU manages its own transformer swap
state independently.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    denoise_video_only,
    encode_prompts,
    multi_modal_guider_denoising_func,
    multi_modal_guider_factory_denoising_func,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    encode_video,
    get_videostream_metadata,
    load_video_conditioning,
)
from ltx_pipelines.utils.model_ledger import ModelLedger
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import PipelineComponents

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CachingModelLedger — returns pre-loaded models instead of disk I/O
# ---------------------------------------------------------------------------


class CachingModelLedger:
    """Drop-in for ModelLedger that returns cached model instances.

    When pipeline code calls ``ledger.transformer()`` then ``del transformer``,
    the cached reference keeps the model alive on GPU.  ``cleanup_memory()``
    frees allocator cache but not the weights themselves.
    """

    def __init__(self, device: torch.device, cache: dict[str, object]) -> None:
        self.device = device
        self.dtype = torch.bfloat16
        self._cache = cache

    def text_encoder(self):
        return self._cache["text_encoder"]

    def gemma_embeddings_processor(self):
        return self._cache["embeddings_processor"]

    def video_encoder(self):
        return self._cache["video_encoder"]

    def audio_encoder(self):
        return self._cache["audio_encoder"]

    def transformer(self):
        return self._cache["transformer"]

    def spatial_upsampler(self):
        return self._cache["spatial_upsampler"]

    def video_decoder(self):
        return self._cache["video_decoder"]

    def audio_decoder(self):
        return self._cache["audio_decoder"]

    def vocoder(self):
        return self._cache["vocoder"]


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
    transformer: object = None
    cache: dict[str, object] = field(default_factory=dict)

    def ensure_transformer(self, state: str) -> None:
        """Swap transformer checkpoint on this worker's GPU."""
        if self.transformer_state == state:
            return

        logger.info("Worker %s: swapping transformer %s -> %s", self.device, self.transformer_state, state)
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

        # Free old transformer BEFORE loading new one to avoid OOM
        # (both ~22GB; GPU:0 has ~72GB baseline, can't hold two)
        old = self.transformer
        self.transformer = None
        self.cache["transformer"] = None
        del old
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
        self.transformer_state = state
        logger.info("Worker %s: transformer now %s", self.device, state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _video_to_bytes(video: Iterator[torch.Tensor], fps: float, audio: Audio, num_frames: int, *, include_audio: bool = True) -> bytes:
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
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

    def load_all(self) -> None:
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
                "spatial_upsampler": den_ledger.spatial_upsampler(),
                "video_decoder": den_ledger.video_decoder(),
                "audio_decoder": den_ledger.audio_decoder(),
                "vocoder": den_ledger.vocoder(),
            }
            worker = DenoiserWorker(
                device=device,
                ledger=CachingModelLedger(device, cache),
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

    @torch.inference_mode()
    def _run_t2v(
        self, worker: DenoiserWorker, prompt: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        is_fast = model == "ltx-2-3-fast"

        worker.ensure_transformer("distilled" if is_fast else "dev")

        # Text encoding on GPU:0 (shared encoder)
        if is_fast:
            (ctx_p,) = encode_prompts([prompt], self._encoder_ledger)
            (ctx_p,) = self._contexts_to_device([ctx_p], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
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
            sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)

            def denoising_loop(sigmas, video_state, audio_state, stepper):
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                    denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))
        else:
            params = detect_params(config.DEV_CHECKPOINT)
            sigmas = LTX2Scheduler().execute(steps=params.num_inference_steps).to(dtype=torch.float32, device=device)

            def denoising_loop(sigmas, video_state, audio_state, stepper):
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                    denoise_fn=multi_modal_guider_factory_denoising_func(
                        video_guider_factory=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n),
                        audio_guider_factory=create_multimodal_guider_factory(params=params.audio_guider_params, negative_context=a_context_n),
                        v_context=v_context_p, a_context=a_context_p, transformer=transformer))

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_shape, conditionings=stage_1_cond, noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
            components=worker.components, dtype=dtype, device=device,
        )

        # Stage 2: upsample + refine
        upscaled = upsample_video(latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        stage_1_audio_latent = audio_state.latent
        del video_state, audio_state, stage_1_cond, sigmas, transformer, denoising_loop
        cleanup_memory()

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=[], height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            worker.ensure_transformer("dev_lora")

        transformer = worker.ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)

        def stage2_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=worker.components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=stage_1_audio_latent,
        )

        # Save stage 2 latents, free everything before potential transformer swap
        video_latent = video_state.latent
        audio_latent = audio_state.latent
        del video_state, audio_state, stage_2_cond, upscaled, stage_1_audio_latent
        del transformer, stage2_loop, distilled_sigmas
        cleanup_memory()

        if not is_fast:
            worker.ensure_transformer("dev")

        # Decode
        decoded_video = vae_decode_video(video_latent, worker.ledger.video_decoder(), TilingConfig.default(), generator)
        decoded_audio = vae_decode_audio(audio_latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        return _video_to_bytes(decoded_video, fps, decoded_audio, num_frames, include_audio=generate_audio)

    @torch.inference_mode()
    def _run_i2v(
        self, worker: DenoiserWorker, prompt: str, image_path: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        is_fast = model == "ltx-2-3-fast"
        images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]

        worker.ensure_transformer("distilled" if is_fast else "dev")

        # Text encoding on GPU:0 (shared encoder)
        if is_fast:
            (ctx_p,) = encode_prompts([prompt], self._encoder_ledger)
            (ctx_p,) = self._contexts_to_device([ctx_p], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
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
            sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)
            def denoising_loop(sigmas, video_state, audio_state, stepper):
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                    denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))
        else:
            params = detect_params(config.DEV_CHECKPOINT)
            sigmas = LTX2Scheduler().execute(steps=params.num_inference_steps).to(dtype=torch.float32, device=device)
            def denoising_loop(sigmas, video_state, audio_state, stepper):
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                    denoise_fn=multi_modal_guider_factory_denoising_func(
                        video_guider_factory=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n),
                        audio_guider_factory=create_multimodal_guider_factory(params=params.audio_guider_params, negative_context=a_context_n),
                        v_context=v_context_p, a_context=a_context_p, transformer=transformer))

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_shape, conditionings=stage_1_cond, noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
            components=worker.components, dtype=dtype, device=device,
        )

        # Stage 2
        upscaled = upsample_video(latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        stage_1_audio_latent = audio_state.latent
        del video_state, audio_state, stage_1_cond, sigmas, transformer, denoising_loop
        cleanup_memory()

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=images, height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            worker.ensure_transformer("dev_lora")

        transformer = worker.ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)
        def stage2_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=worker.components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=stage_1_audio_latent,
        )

        # Save stage 2 latents, free everything before potential transformer swap
        video_latent = video_state.latent
        audio_latent = audio_state.latent
        del video_state, audio_state, stage_2_cond, upscaled, stage_1_audio_latent
        del transformer, stage2_loop, distilled_sigmas
        cleanup_memory()

        if not is_fast:
            worker.ensure_transformer("dev")

        decoded_video = vae_decode_video(video_latent, worker.ledger.video_decoder(), TilingConfig.default(), generator)
        decoded_audio = vae_decode_audio(audio_latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        return _video_to_bytes(decoded_video, fps, decoded_audio, num_frames, include_audio=generate_audio)

    @torch.inference_mode()
    def _run_a2v(
        self, worker: DenoiserWorker, prompt: str, audio_path: str, image_path: str | None,
        width: int, height: int, num_frames: int, fps: float, seed: int,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16

        worker.ensure_transformer("dev")

        # Text encoding on GPU:0 (shared encoder)
        ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
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
        params = detect_params(config.DEV_CHECKPOINT)

        # Stage 1: video-only denoising (audio frozen)
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = worker.ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=images, height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)
        transformer = worker.ledger.transformer()

        sigmas = LTX2Scheduler().execute(steps=params.num_inference_steps).to(dtype=torch.float32, device=device)

        def stage1_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=multi_modal_guider_factory_denoising_func(
                    video_guider_factory=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n),
                    audio_guider_factory=create_multimodal_guider_factory(params=MultiModalGuiderParams(), negative_context=None),
                    v_context=v_context_p, a_context=a_context_p, transformer=transformer))

        video_state = denoise_video_only(
            output_shape=stage_1_shape, conditionings=stage_1_cond, noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=stage1_loop,
            components=worker.components, dtype=dtype, device=device,
            initial_audio_latent=encoded_audio_latent,
        )

        # Stage 2: refine with distilled LoRA
        upscaled = upsample_video(latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        del video_state, stage_1_cond, sigmas, transformer, stage1_loop
        cleanup_memory()

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=images, height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        worker.ensure_transformer("dev_lora")
        transformer = worker.ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)

        def stage2_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))

        video_state = denoise_video_only(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=worker.components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=encoded_audio_latent,
        )

        # Save stage 2 latent, free everything before transformer swap
        video_latent = video_state.latent
        del video_state, stage_2_cond, upscaled
        del transformer, stage2_loop, distilled_sigmas
        cleanup_memory()

        worker.ensure_transformer("dev")

        # Decode video but return ORIGINAL audio (a2v passthrough)
        decoded_video = vae_decode_video(video_latent, worker.ledger.video_decoder(), TilingConfig.default(), generator)
        original_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)
        return _video_to_bytes(decoded_video, fps, original_audio, num_frames)

    @torch.inference_mode()
    def _run_retake(
        self, worker: DenoiserWorker, video_path: str, start_time: float, duration: float,
        mode: str, prompt: str, seed: int,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        end_time = start_time + duration
        regenerate_video = mode in ("replace_audio_and_video", "replace_video", "replace_video_only")
        regenerate_audio = mode in ("replace_audio_and_video", "replace_audio")

        worker.ensure_transformer("dev")

        # Get video metadata
        fps_vid, num_frames, vid_width, vid_height = get_videostream_metadata(video_path)

        # Text encoding on GPU:0 (shared encoder)
        params = detect_params(config.DEV_CHECKPOINT)
        ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
        ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Encode input video on GPU:0, transfer to worker device
        video_encoder_enc = self._encoder_ledger.video_encoder()
        video_conditioning = load_video_conditioning(video_path, height=vid_height, width=vid_width)
        initial_video_latent = video_encoder_enc(video_conditioning.to(self._encoder_device, dtype=dtype)).to(device)

        # Encode audio from video on GPU:0, transfer to worker device
        decoded_audio = decode_audio_from_file(video_path, self._encoder_device, max_duration=num_frames / fps_vid)
        audio_encoder = self._encoder_ledger.audio_encoder()
        initial_audio_latent = vae_encode_audio(decoded_audio, audio_encoder).to(device)

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        sigmas = LTX2Scheduler().execute(steps=params.num_inference_steps).to(dtype=torch.float32, device=device)

        transformer = worker.ledger.transformer()

        # Build denoising function with guiders
        def retake_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n).build(sigmas[0]),
                    audio_guider=create_multimodal_guider_factory(params=params.audio_guider_params, negative_context=a_context_n).build(sigmas[0]),
                    v_context=v_context_p, a_context=a_context_p, transformer=transformer))

        output_shape = VideoPixelShape(batch=1, frames=num_frames, width=vid_width, height=vid_height, fps=fps_vid)

        video_state, audio_state = denoise_audio_video(
            output_shape=output_shape, conditionings=[], noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=retake_loop,
            components=worker.components, dtype=dtype, device=device,
            initial_video_latent=initial_video_latent,
            initial_audio_latent=initial_audio_latent,
        )

        decoded_video = vae_decode_video(video_state.latent, worker.ledger.video_decoder(), TilingConfig.default(), generator)
        decoded_audio = vae_decode_audio(audio_state.latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        return _video_to_bytes(decoded_video, fps_vid, decoded_audio, num_frames)

    # --- Async API (matches PipelineManager interface) ---

    async def generate_text_to_video(
        self, prompt: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool = True,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_t2v, worker, prompt, model, width, height,
                num_frames, fps, seed, generate_audio,
            )
        finally:
            worker.lock.release()

    async def generate_image_to_video(
        self, prompt: str, image_path: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool = True,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_i2v, worker, prompt, image_path, model, width, height,
                num_frames, fps, seed, generate_audio,
            )
        finally:
            worker.lock.release()

    async def generate_audio_to_video(
        self, prompt: str, audio_path: str, image_path: str | None,
        model: str, width: int, height: int, num_frames: int, fps: float, seed: int,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_a2v, worker, prompt, audio_path, image_path,
                width, height, num_frames, fps, seed,
            )
        finally:
            worker.lock.release()

    async def retake(
        self, video_path: str, start_time: float, duration: float,
        mode: str, prompt: str, seed: int,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_retake, worker, video_path, start_time, duration,
                mode, prompt, seed,
            )
        finally:
            worker.lock.release()
