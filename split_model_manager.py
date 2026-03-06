"""Split-GPU model manager for LTX-2 video generation.

Keeps text encoder + VAE encoders on GPU:0 (encoder_device),
transformer + decoders on GPU:1 (denoiser_device). Models stay
resident — no per-request loading/unloading except transformer
checkpoint swaps between dev/distilled variants.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import Iterator
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
    """Loads models once across two GPUs and dispatches inference.

    GPU:0 (encoder): Gemma text encoder, embeddings processor, VAE encoders
    GPU:1 (denoiser): Transformer (swappable), spatial upsampler, VAE decoders, vocoder
    Video encoder duplicated on both GPUs.
    """

    def __init__(self) -> None:
        self._encoder_device: torch.device | None = None
        self._denoiser_device: torch.device | None = None
        self._encoder_ledger: CachingModelLedger | None = None
        self._denoiser_ledger: CachingModelLedger | None = None
        self._transformer_state: str = ""
        self._lock: asyncio.Lock | None = None
        self._components: PipelineComponents | None = None
        # Keep direct references for swap logic
        self._transformer = None
        self._denoiser_cache: dict[str, object] = {}

    @property
    def is_ready(self) -> bool:
        return self._encoder_ledger is not None and self._denoiser_ledger is not None

    def load_all(self) -> None:
        encoder_device = torch.device(config.GPU_DEVICES[0])
        denoiser_device = torch.device(config.GPU_DEVICES[1])
        self._encoder_device = encoder_device
        self._denoiser_device = denoiser_device

        # --- Encoder hub (GPU:0) ---
        logger.info("Loading encoder hub on %s ...", encoder_device)
        enc_ledger = ModelLedger(
            dtype=torch.bfloat16, device=encoder_device,
            checkpoint_path=config.DEV_CHECKPOINT,
            gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
            loras=(),
        )
        encoder_cache = {
            "text_encoder": enc_ledger.text_encoder(),
            "embeddings_processor": enc_ledger.gemma_embeddings_processor(),
            "video_encoder": enc_ledger.video_encoder(),
            "audio_encoder": enc_ledger.audio_encoder(),
        }
        self._encoder_ledger = CachingModelLedger(encoder_device, encoder_cache)

        # --- Denoiser hub (GPU:1) ---
        logger.info("Loading denoiser hub on %s ...", denoiser_device)
        den_ledger = ModelLedger(
            dtype=torch.bfloat16, device=denoiser_device,
            checkpoint_path=config.DEV_CHECKPOINT,
            gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
            loras=(),
        )
        self._transformer = den_ledger.transformer()
        self._denoiser_cache = {
            "transformer": self._transformer,
            "video_encoder": den_ledger.video_encoder(),
            "spatial_upsampler": den_ledger.spatial_upsampler(),
            "video_decoder": den_ledger.video_decoder(),
            "audio_decoder": den_ledger.audio_decoder(),
            "vocoder": den_ledger.vocoder(),
        }
        self._denoiser_ledger = CachingModelLedger(denoiser_device, self._denoiser_cache)
        self._transformer_state = "dev"

        self._components = PipelineComponents(dtype=torch.bfloat16, device=denoiser_device)
        self._lock = asyncio.Lock()
        logger.info("All models loaded. Transformer: dev")

    # --- Transformer swap ---

    def _ensure_transformer(self, state: str) -> None:
        if self._transformer_state == state:
            return

        logger.info("Swapping transformer: %s -> %s", self._transformer_state, state)
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

        # Load new transformer FIRST (error safety)
        ledger = ModelLedger(
            dtype=torch.bfloat16, device=self._denoiser_device,
            checkpoint_path=checkpoint, gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER, loras=loras,
        )
        new_transformer = ledger.transformer()

        old = self._transformer
        self._transformer = new_transformer
        self._denoiser_cache["transformer"] = new_transformer
        self._transformer_state = state
        del old
        torch.cuda.synchronize(self._denoiser_device)
        torch.cuda.empty_cache()
        logger.info("Transformer swapped to %s", state)

    # --- Context transfer ---

    def _contexts_to_denoiser(self, contexts):
        """Move EmbeddingsProcessorOutput tensors from encoder to denoiser device."""
        from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
        target = self._denoiser_device
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
        self, prompt: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool,
    ) -> bytes:
        device = self._denoiser_device
        dtype = torch.bfloat16
        is_fast = model == "ltx-2-3-fast"

        # Ensure correct transformer
        self._ensure_transformer("distilled" if is_fast else "dev")

        # Text encoding on GPU:0
        if is_fast:
            (ctx_p,) = encode_prompts([prompt], self._encoder_ledger)
            (ctx_p,) = self._contexts_to_denoiser([ctx_p])
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
            ctx_p, ctx_n = self._contexts_to_denoiser([ctx_p, ctx_n])
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Stage 1: half-resolution denoising
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = self._denoiser_ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=[], height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        transformer = self._denoiser_ledger.transformer()

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
            components=self._components, dtype=dtype, device=device,
        )

        # Stage 2: upsample + refine
        upscaled = upsample_video(latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=self._denoiser_ledger.spatial_upsampler())
        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=[], height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            self._ensure_transformer("dev_lora")
            transformer = self._denoiser_ledger.transformer()

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)

        def stage2_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=self._components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=audio_state.latent,
        )

        if not is_fast:
            self._ensure_transformer("dev")

        # Decode
        decoded_video = vae_decode_video(video_state.latent, self._denoiser_ledger.video_decoder(), TilingConfig.default(), generator)
        decoded_audio = vae_decode_audio(audio_state.latent, self._denoiser_ledger.audio_decoder(), self._denoiser_ledger.vocoder())
        return _video_to_bytes(decoded_video, fps, decoded_audio, num_frames, include_audio=generate_audio)

    @torch.inference_mode()
    def _run_i2v(
        self, prompt: str, image_path: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool,
    ) -> bytes:
        device = self._denoiser_device
        dtype = torch.bfloat16
        is_fast = model == "ltx-2-3-fast"
        images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]

        self._ensure_transformer("distilled" if is_fast else "dev")

        # Text encoding on GPU:0
        if is_fast:
            (ctx_p,) = encode_prompts([prompt], self._encoder_ledger)
            (ctx_p,) = self._contexts_to_denoiser([ctx_p])
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
            ctx_p, ctx_n = self._contexts_to_denoiser([ctx_p, ctx_n])
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Stage 1
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = self._denoiser_ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=images, height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)
        transformer = self._denoiser_ledger.transformer()

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
            components=self._components, dtype=dtype, device=device,
        )

        # Stage 2
        upscaled = upsample_video(latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=self._denoiser_ledger.spatial_upsampler())
        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=images, height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            self._ensure_transformer("dev_lora")
            transformer = self._denoiser_ledger.transformer()

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)
        def stage2_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=self._components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=audio_state.latent,
        )

        if not is_fast:
            self._ensure_transformer("dev")

        decoded_video = vae_decode_video(video_state.latent, self._denoiser_ledger.video_decoder(), TilingConfig.default(), generator)
        decoded_audio = vae_decode_audio(audio_state.latent, self._denoiser_ledger.audio_decoder(), self._denoiser_ledger.vocoder())
        return _video_to_bytes(decoded_video, fps, decoded_audio, num_frames, include_audio=generate_audio)

    @torch.inference_mode()
    def _run_a2v(
        self, prompt: str, audio_path: str, image_path: str | None,
        width: int, height: int, num_frames: int, fps: float, seed: int,
    ) -> bytes:
        device = self._denoiser_device
        dtype = torch.bfloat16

        self._ensure_transformer("dev")

        # Text encoding on GPU:0
        ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
        ctx_p, ctx_n = self._contexts_to_denoiser([ctx_p, ctx_n])
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Audio encoding on GPU:0, then transfer
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
        video_encoder = self._denoiser_ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=images, height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)
        transformer = self._denoiser_ledger.transformer()

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
            components=self._components, dtype=dtype, device=device,
            initial_audio_latent=encoded_audio_latent,
        )

        # Stage 2: refine with distilled LoRA
        upscaled = upsample_video(latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=self._denoiser_ledger.spatial_upsampler())
        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=images, height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        self._ensure_transformer("dev_lora")
        transformer = self._denoiser_ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)

        def stage2_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))

        video_state = denoise_video_only(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=self._components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=encoded_audio_latent,
        )

        self._ensure_transformer("dev")

        # Decode video but return ORIGINAL audio (a2v passthrough)
        decoded_video = vae_decode_video(video_state.latent, self._denoiser_ledger.video_decoder(), TilingConfig.default(), generator)
        original_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)
        return _video_to_bytes(decoded_video, fps, original_audio, num_frames)

    @torch.inference_mode()
    def _run_retake(
        self, video_path: str, start_time: float, duration: float,
        mode: str, prompt: str, seed: int,
    ) -> bytes:
        device = self._denoiser_device
        dtype = torch.bfloat16
        end_time = start_time + duration
        regenerate_video = mode in ("replace_audio_and_video", "replace_video", "replace_video_only")
        regenerate_audio = mode in ("replace_audio_and_video", "replace_audio")

        self._ensure_transformer("dev")

        # Get video metadata
        fps_vid, num_frames, vid_width, vid_height = get_videostream_metadata(video_path)

        # Text encoding on GPU:0
        params = detect_params(config.DEV_CHECKPOINT)
        ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
        ctx_p, ctx_n = self._contexts_to_denoiser([ctx_p, ctx_n])
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Encode input video on GPU:0, transfer to GPU:1
        video_encoder_enc = self._encoder_ledger.video_encoder()
        video_conditioning = load_video_conditioning(video_path, height=vid_height, width=vid_width)
        initial_video_latent = video_encoder_enc(video_conditioning.to(self._encoder_device, dtype=dtype)).to(device)

        # Encode audio from video on GPU:0, transfer
        decoded_audio = decode_audio_from_file(video_path, self._encoder_device, max_duration=num_frames / fps_vid)
        audio_encoder = self._encoder_ledger.audio_encoder()
        initial_audio_latent = vae_encode_audio(decoded_audio, audio_encoder).to(device)

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        sigmas = LTX2Scheduler().execute(steps=params.num_inference_steps).to(dtype=torch.float32, device=device)

        transformer = self._denoiser_ledger.transformer()

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
            components=self._components, dtype=dtype, device=device,
            initial_video_latent=initial_video_latent,
            initial_audio_latent=initial_audio_latent,
        )

        decoded_video = vae_decode_video(video_state.latent, self._denoiser_ledger.video_decoder(), TilingConfig.default(), generator)
        decoded_audio = vae_decode_audio(audio_state.latent, self._denoiser_ledger.audio_decoder(), self._denoiser_ledger.vocoder())
        return _video_to_bytes(decoded_video, fps_vid, decoded_audio, num_frames)

    # --- Async API (matches PipelineManager interface) ---

    async def generate_text_to_video(
        self, prompt: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool = True,
    ) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_t2v, prompt, model, width, height,
                num_frames, fps, seed, generate_audio,
            )

    async def generate_image_to_video(
        self, prompt: str, image_path: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool = True,
    ) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_i2v, prompt, image_path, model, width, height,
                num_frames, fps, seed, generate_audio,
            )

    async def generate_audio_to_video(
        self, prompt: str, audio_path: str, image_path: str | None,
        model: str, width: int, height: int, num_frames: int, fps: float, seed: int,
    ) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_a2v, prompt, audio_path, image_path,
                width, height, num_frames, fps, seed,
            )

    async def retake(
        self, video_path: str, start_time: float, duration: float,
        mode: str, prompt: str, seed: int,
    ) -> bytes:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_retake, video_path, start_time, duration,
                mode, prompt, seed,
            )
