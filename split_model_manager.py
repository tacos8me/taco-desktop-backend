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
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch
from dataclasses import replace as _replace
from tqdm import tqdm

# --- ltx_core (stable across upstream updates) ---
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep, Res2sDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuider,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator, X0Model
from ltx_core.model.transformer.compiling import COMPILE_TRANSFORMER, modify_sd_ops_for_compilation
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
    get_video_chunks_number,
)
from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_core.tools import AudioLatentTools, VideoLatentShape, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_core.utils import find_matching_file

# --- ltx_pipelines (new API) ---
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    DISTILLED_SIGMA_VALUES,
    DISTILLED_SIGMAS,
    LTX_2_3_HQ_PARAMS,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMAS,
    detect_params,
)
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    combined_image_conditionings,
    create_noised_state,
    generate_enhanced_prompt,
    post_process_latent,
)
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    decode_video_from_file,
    encode_video,
    get_videostream_metadata,
    video_preprocess,
)
from ltx_pipelines.utils.samplers import (
    euler_denoising_loop,
    gradient_estimating_euler_denoising_loop,
    res2s_audio_video_denoising_loop,
)
from ltx_pipelines.utils.types import LatentState, PipelineComponents

_USE_GE_EULER = os.environ.get("USE_GE_EULER", "").lower() in ("1", "true", "yes")

import config

logger = logging.getLogger(__name__)

# Pre-compute dev checkpoint params once at import time (avoids repeated disk I/O)
_DEV_PARAMS = detect_params(config.DEV_CHECKPOINT)

# skip_step=0: full STG on every step. skip_step=1 saved 33% NFE but created
# temporal oscillation in guidance signal, contributing to ghost trails during fast motion.

# ---------------------------------------------------------------------------
# CachingModelFactory — cached models with lazy-load from SingleGPUModelBuilder
# ---------------------------------------------------------------------------


class CachingModelFactory:
    """Server-optimized model cache using SingleGPUModelBuilder for lazy-loading.

    Pre-loaded models (text encoder, transformer, etc.) are stored in ``_cache``
    and returned directly. Decoders/upsampler are built on-demand via
    ``SingleGPUModelBuilder`` to keep baseline VRAM low.
    """

    def __init__(self, device: torch.device, cache: dict[str, object],
                 builders: dict[str, SingleGPUModelBuilder] | None = None) -> None:
        self.device = device
        self.dtype = torch.bfloat16
        self._cache = cache
        self._builders = builders or {}

    def _lazy(self, key: str):
        val = self._cache.get(key)
        if val is None and key in self._builders:
            logger.info("Lazy-loading %s on %s", key, self.device)
            val = self._builders[key].build(device=self.device, dtype=self.dtype).to(self.device).eval()
            self._cache[key] = val
        return val

    def text_encoder(self):
        return self._cache.get("text_encoder")

    def gemma_embeddings_processor(self):
        return self._cache.get("embeddings_processor")

    def video_encoder(self):
        return self._lazy("video_encoder")

    def audio_encoder(self):
        return self._cache.get("audio_encoder")

    def transformer(self):
        return self._cache.get("transformer")

    def spatial_upsampler(self):
        return self._lazy("spatial_upsampler")

    def video_decoder(self):
        return self._lazy("video_decoder")

    def audio_decoder(self):
        return self._lazy("audio_decoder")

    def vocoder(self):
        return self._lazy("vocoder")


# ---------------------------------------------------------------------------
# DenoiserWorker — per-GPU denoiser state
# ---------------------------------------------------------------------------


@dataclass
class DenoiserWorker:
    """Per-GPU worker with its own transformer, decoders, and swap state."""
    device: torch.device
    ledger: CachingModelFactory
    components: PipelineComponents
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    transformer_state: str = ""
    _user_lora: tuple[str, float] | None = None
    transformer: object = None
    cache: dict[str, object] = field(default_factory=dict)

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

        # Build transformer via SingleGPUModelBuilder (replaces ModelLedger)
        sd_ops = LTXV_MODEL_COMFY_RENAMING_MAP
        module_ops: tuple = ()
        build_loras = tuple(loras)
        if config.ENABLE_TORCH_COMPILE:
            module_ops = (COMPILE_TRANSFORMER,)
            sd_ops = modify_sd_ops_for_compilation(sd_ops)
            build_loras = tuple(
                LoraPathStrengthAndSDOps(
                    path=l.path, strength=l.strength,
                    sd_ops=modify_sd_ops_for_compilation(l.sd_ops) if l.sd_ops is not None else l.sd_ops,
                )
                for l in loras
            )

        builder = SingleGPUModelBuilder(
            model_path=checkpoint,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=sd_ops,
            loras=build_loras,
            module_ops=module_ops,
        )
        new_transformer = X0Model(builder.build(device=self.device, dtype=torch.bfloat16)).to(self.device).eval()
        self.transformer = new_transformer
        self.cache["transformer"] = new_transformer
        self.ledger._cache["transformer"] = new_transformer
        self.transformer_state = state
        self._user_lora = user_lora
        logger.info("Worker %s: transformer now %s", self.device, state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use upstream default tiling config — the tiling implementation was rewritten in
# the block-based refactor (cosine S-curve blending, weight accumulation). Our old
# config (128 frames / 32 overlap / no spatial) produced checkerboard artifacts
# with the new implementation.
DECODE_TILING = TilingConfig.default()

# Skip tiling for videos ≤257 frames (~10s at 24fps). Single-pass decode = no tile
# boundary artifacts at all. cuDNN 9.20 fixes the conv3d workspace bug so this fits
# in ~90GB available after transformer eviction. Only videos >10s need tiling.
SHORT_VIDEO_THRESHOLD = 257

# Gradient estimating euler: momentum-accelerated sampler, potentially 30→20 steps
# Enable via USE_GE_EULER=1 in .env for A/B testing
_euler_loop = gradient_estimating_euler_denoising_loop if _USE_GE_EULER else euler_denoising_loop

# ---------------------------------------------------------------------------
# Sampler configuration (toggled via /v1/system/sampler)
# ---------------------------------------------------------------------------
_use_cfg_pp: bool = True
_cfg_pp_eta_stage1: float = 1.0   # ancestral noise for distilled stage 1
_cfg_pp_eta_default: float = 0.0  # deterministic for guided/stage 2
_stage2_sigmas_override: list[float] | None = [0.85, 0.725, 0.4219, 0.0]


def _get_ancestral_step(sigma_from: float, sigma_to: float, eta: float = 1.0) -> tuple[float, float]:
    """Split sigma step into deterministic (sigma_down) + stochastic (sigma_up) parts."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def euler_cfg_pp_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper,  # DiffusionStepProtocol — unused, kept for interface compat
    transformer,
    denoiser,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[LatentState | None, LatentState | None]:
    """CFG++ Euler sampler ported from ComfyUI's sample_euler_ancestral_cfg_pp.

    For LTX-2 (CONST / flow-matching): alpha(sigma) = 1 - sigma.
    The key difference from standard Euler is the alpha-rescaled step formula
    which keeps the trajectory on the data manifold instead of drifting.
    """
    for step_idx in tqdm(range(len(sigmas) - 1)):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        # Get denoised prediction from the denoiser (handles CFG internally)
        denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_idx)

        # Post-process: blend with clean_latent via denoise_mask
        if video_state is not None and denoised_video is not None:
            denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        if audio_state is not None and denoised_audio is not None:
            denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        if sigma_next == 0:
            # Final step: jump straight to denoised
            if video_state is not None and denoised_video is not None:
                video_state = _replace(video_state, latent=denoised_video)
            if audio_state is not None and denoised_audio is not None:
                audio_state = _replace(audio_state, latent=denoised_audio)
        else:
            alpha_s = max(1.0 - sigma.item(), 1e-8)
            alpha_t = max(1.0 - sigma_next.item(), 1e-8)

            if eta > 0:
                sigma_down, sigma_up = _get_ancestral_step(
                    sigma.item() / alpha_s, sigma_next.item() / alpha_t, eta=eta,
                )
            else:
                sigma_down = sigma_next.item() / alpha_t
                sigma_up = 0.0

            video_state = _cfg_pp_step_state(video_state, denoised_video, sigma, alpha_s, alpha_t, sigma_down, sigma_up, eta, generator)
            audio_state = _cfg_pp_step_state(audio_state, denoised_audio, sigma, alpha_s, alpha_t, sigma_down, sigma_up, eta, generator)

    return (video_state, audio_state)


def _cfg_pp_step_state(
    state: LatentState | None,
    denoised: torch.Tensor | None,
    sigma: torch.Tensor,
    alpha_s: float,
    alpha_t: float,
    sigma_down: float,
    sigma_up: float,
    eta: float,
    generator: torch.Generator | None,
) -> LatentState | None:
    """Apply one CFG++ Euler step to a single modality."""
    if state is None or denoised is None:
        return state
    x = state.latent.float()
    d = denoised.float()
    # Noise direction: (x - alpha_s * denoised) / sigma
    noise_dir = (x - alpha_s * d) / sigma
    # Euler step with alpha rescaling
    x_next = alpha_t * d + alpha_t * sigma_down * noise_dir
    if eta > 0 and sigma_up > 0:
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x_next = x_next + alpha_t * noise * sigma_up
    return _replace(state, latent=x_next.to(state.latent.dtype))


def _get_decode_tiling(num_frames: int) -> TilingConfig | None:
    """Skip tiling for short videos to avoid temporal boundary artifacts."""
    return None if num_frames <= SHORT_VIDEO_THRESHOLD else DECODE_TILING


@contextmanager
def _timed(label: str):
    """Log wall-clock elapsed for a block. Used at post-denoise phase boundaries
    so we can measure where the v1.1.5 "stuck at 95%" tail actually goes —
    previously the VAE decode + ffmpeg encode ran silently for 10+ seconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s: %.2fs", label, time.perf_counter() - t0)


def _emit_phase(on_progress, progress: float, phase: str) -> None:
    """Emit a (progress, phase) update if a callback is present.

    Signature-flexible: our own `on_progress` closures accept `(float, str)`,
    but upstream ltx-pipelines may pass float-only callbacks through our
    `_wrap_denoise`. This helper only runs in taco-backend's own post-denoise
    code paths, so we know the callback accepts the phase kwarg."""
    if on_progress is not None:
        try:
            on_progress(progress, phase=phase)
        except TypeError:
            # Callback doesn't accept phase kwarg — degrade to progress-only.
            on_progress(progress)


def _decode_video_fp32(latent: torch.Tensor, decoder, tiling, generator) -> Iterator[torch.Tensor]:
    """Decode video via VideoDecoder.decode_video method."""
    yield from decoder.decode_video(latent, tiling, generator)


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
        result = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    # Release the torch caching allocator's held blocks from decode/encode.
    # Without this, 10-15 GB of freed 4K activation blocks linger in the
    # allocator cache and cause OOM on the NEXT generation.
    gc.collect()
    torch.cuda.empty_cache()
    return result


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
        self._encoder_ledger: CachingModelFactory | None = None
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
            for key in list(worker.cache.keys()):
                worker.cache[key] = None
            # Clear builder refs so weight tensors can be GC'd
            worker.ledger._builders.clear()
        self._workers.clear()
        if self._encoder_ledger is not None:
            for key in list(self._encoder_ledger._cache.keys()):
                self._encoder_ledger._cache[key] = None
            self._encoder_ledger._builders.clear()
            self._encoder_ledger = None
        gc.collect()
        for device_name in config.GPU_DEVICES:
            torch.cuda.synchronize(torch.device(device_name))
            torch.cuda.empty_cache()
        logger.info("All LTX models evicted from GPU")

    def load_all(self) -> None:
        self._workers.clear()
        devices = [torch.device(d) for d in config.GPU_DEVICES]
        checkpoint = config.DEV_CHECKPOINT

        # --- Shared encoder hub on GPU:0 ---
        self._encoder_device = devices[0]
        logger.info("Loading shared encoder hub on %s ...", devices[0])

        # Build encoder components directly via SingleGPUModelBuilder
        gemma_module_ops = module_ops_from_gemma_root(config.GEMMA_ROOT)
        model_folder = find_matching_file(config.GEMMA_ROOT, "model*.safetensors").parent
        gemma_weight_paths = tuple(str(p) for p in model_folder.rglob("*.safetensors"))

        text_encoder = SingleGPUModelBuilder(
            model_path=gemma_weight_paths,
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *gemma_module_ops),
        ).build(device=devices[0], dtype=torch.bfloat16).to(devices[0]).eval()

        embeddings_processor = SingleGPUModelBuilder(
            model_path=checkpoint,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
        ).build(device=devices[0], dtype=torch.bfloat16).to(devices[0]).eval()

        enc_video_encoder = SingleGPUModelBuilder(
            model_path=checkpoint,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=devices[0], dtype=torch.bfloat16).to(devices[0]).eval()

        audio_encoder = SingleGPUModelBuilder(
            model_path=checkpoint,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=devices[0], dtype=torch.bfloat16).to(devices[0]).eval()

        encoder_cache = {
            "text_encoder": text_encoder,
            "embeddings_processor": embeddings_processor,
            "video_encoder": enc_video_encoder,
            "audio_encoder": audio_encoder,
        }
        self._encoder_ledger = CachingModelFactory(devices[0], encoder_cache)

        # --- Denoiser worker on each GPU ---
        # Only pre-load transformer + video_encoder on GPU. Decoders/upsampler
        # are loaded on-demand (after transformer is evicted, freeing ~44GB).
        for device in devices:
            logger.info("Loading denoiser worker on %s ...", device)

            # Build transformer (with optional torch.compile)
            sd_ops = LTXV_MODEL_COMFY_RENAMING_MAP
            module_ops: tuple = ()
            if config.ENABLE_TORCH_COMPILE:
                module_ops = (COMPILE_TRANSFORMER,)
                sd_ops = modify_sd_ops_for_compilation(sd_ops)
            transformer = X0Model(
                SingleGPUModelBuilder(
                    model_path=checkpoint,
                    model_class_configurator=LTXModelConfigurator,
                    model_sd_ops=sd_ops,
                    module_ops=module_ops,
                ).build(device=device, dtype=torch.bfloat16)
            ).to(device).eval()

            # GPU:0 reuses encoder hub's video_encoder to avoid ~5GB duplication
            vid_enc = enc_video_encoder if device == devices[0] else SingleGPUModelBuilder(
                model_path=checkpoint,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            ).build(device=device, dtype=torch.bfloat16).to(device).eval()

            cache = {
                "transformer": transformer,
                "video_encoder": vid_enc,
            }
            # Lazy-load builders for components freed after decode (video_encoder,
            # spatial_upsampler evicted between stages; decoders loaded on demand).
            lazy_builders = {
                "video_encoder": SingleGPUModelBuilder(model_path=checkpoint, model_class_configurator=VideoEncoderConfigurator, model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER),
                "video_decoder": SingleGPUModelBuilder(model_path=checkpoint, model_class_configurator=VideoDecoderConfigurator, model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER),
                "audio_decoder": SingleGPUModelBuilder(model_path=checkpoint, model_class_configurator=AudioDecoderConfigurator, model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER),
                "vocoder": SingleGPUModelBuilder(model_path=checkpoint, model_class_configurator=VocoderConfigurator, model_sd_ops=VOCODER_COMFY_KEYS_FILTER),
                "spatial_upsampler": SingleGPUModelBuilder(model_path=config.SPATIAL_UPSAMPLER, model_class_configurator=LatentUpsamplerConfigurator),
            }
            worker = DenoiserWorker(
                device=device,
                ledger=CachingModelFactory(device, cache, builders=lazy_builders),
                components=PipelineComponents(dtype=torch.bfloat16, device=device),
                transformer_state="dev",
                transformer=transformer,
                cache=cache,
            )
            self._workers.append(worker)
            logger.info("Denoiser worker ready on %s", device)

        logger.info("All models loaded: %d workers, encoder on %s", len(self._workers), self._encoder_device)

    # --- Encoder hub paging (CPU ↔ GPU) ---

    def _page_encoder_to_cpu(self) -> None:
        """Move text encoder + audio encoder to CPU after prompt encoding.

        Frees ~25.5 GB on cuda:0 for denoising + VAE decode. The encoder
        is only needed during the ~2s encode_prompts() call; keeping it
        resident during the 30-60s denoising + decode wastes 25% of the
        96 GB budget and causes OOM on 4K (3840x2160) generations.
        """
        if self._encoder_ledger is None:
            return
        te = self._encoder_ledger._cache.get("text_encoder")
        ae = self._encoder_ledger._cache.get("audio_encoder")
        if te is not None:
            te.to("cpu")
        if ae is not None:
            ae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Encoder hub paged to CPU (freed ~25 GB on %s)", self._encoder_device)

    def _page_encoder_to_gpu(self) -> None:
        """Move text encoder + audio encoder back to GPU before encoding.

        Cost: ~2-3s via PCIe 5.0 (25 GB at 13-53 GB/s depending on
        pinned vs pageable). Acceptable since gen takes 30-60s total.
        """
        if self._encoder_ledger is None or self._encoder_device is None:
            return
        te = self._encoder_ledger._cache.get("text_encoder")
        ae = self._encoder_ledger._cache.get("audio_encoder")
        if te is not None:
            try:
                if next(te.parameters()).device.type == "cpu":
                    te.to(self._encoder_device)
            except StopIteration:
                pass
        if ae is not None:
            try:
                if next(ae.parameters()).device.type == "cpu":
                    ae.to(self._encoder_device)
            except StopIteration:
                pass

    # --- Worker acquisition ---

    async def _acquire_worker(self) -> DenoiserWorker:
        """Wait for and return the first unlocked worker."""
        if not self._workers:
            raise RuntimeError("No LTX workers available — call load_all() first")
        while True:
            for worker in self._workers:
                if not worker.lock.locked():
                    await worker.lock.acquire()
                    return worker
            await asyncio.sleep(0.05)

    # --- Context transfer ---

    def _contexts_to_device(self, contexts, target: torch.device):
        """Move EmbeddingsProcessorOutput tensors to target device."""
        return [
            EmbeddingsProcessorOutput(
                video_encoding=ctx.video_encoding.to(target),
                audio_encoding=ctx.audio_encoding.to(target) if ctx.audio_encoding is not None else None,
                attention_mask=ctx.attention_mask.to(target),
            )
            for ctx in contexts
        ]

    # --- Prompt encoding (replaces upstream encode_prompts) ---

    def _encode_prompts(self, prompts: list[str], *, enhance_first_prompt: bool = False, enhance_prompt_image: str | None = None, enhance_prompt_seed: int = 42) -> list[EmbeddingsProcessorOutput]:
        """Encode prompts through cached Gemma text encoder + embeddings processor."""
        text_encoder = self._encoder_ledger.text_encoder()
        if enhance_first_prompt:
            prompts = list(prompts)
            prompts[0] = generate_enhanced_prompt(text_encoder, prompts[0], image_path=enhance_prompt_image, seed=enhance_prompt_seed)
        raw_outputs = [text_encoder.encode(p) for p in prompts]
        torch.cuda.synchronize()
        embeddings_processor = self._encoder_ledger.gemma_embeddings_processor()
        return [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]

    # --- State creation helper (replaces denoise_audio_video / denoise_video_only) ---

    @staticmethod
    def _create_av_states(
        pixel_shape: VideoPixelShape, fps: float, noiser, dtype, device,
        video_conds=None, audio_conds=None,
        initial_video=None, initial_audio=None,
        video_noise_scale: float = 1.0, audio_noise_scale: float = 1.0,
        freeze_audio: bool = False,
    ):
        """Create noised video + audio latent states for denoising.

        Returns (video_state, audio_state, video_tools, audio_tools).
        """
        v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
        v_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)
        video_state = create_noised_state(
            v_tools, video_conds or [], noiser, dtype, device,
            noise_scale=video_noise_scale, initial_latent=initial_video,
        )

        a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        a_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
        # Frozen audio must have noise_scale=0.0 — the old denoise_video_only
        # explicitly zeroed it. Adding noise then freezing the mask means the
        # transformer sees noisy audio conditioning every step, corrupting video.
        effective_audio_noise = 0.0 if freeze_audio else audio_noise_scale
        audio_state = create_noised_state(
            a_tools, audio_conds or [], noiser, dtype, device,
            noise_scale=effective_audio_noise, initial_latent=initial_audio,
        )
        if freeze_audio:
            audio_state = _replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))

        return video_state, audio_state, v_tools, a_tools

    # --- Generation flows ---

    @staticmethod
    def _wrap_denoiser(denoiser, on_progress, total_steps, offset=0.0, scale=1.0):
        """Wrap a Denoiser to report progress on each step.

        Cap is 0.90: the top 10% of the progress bar is reserved for
        post-denoise phases (decoding, encoding, saving).
        """
        step_count = [0]

        class ProgressDenoiser:
            def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
                result = denoiser(transformer, video_state, audio_state, sigmas, step_index)
                step_count[0] += 1
                p = offset + min(step_count[0] / max(total_steps, 1), 1.0) * scale
                on_progress(min(p, 0.90))
                return result

        return ProgressDenoiser()

    @torch.inference_mode()
    def _run_t2v(
        self, worker: DenoiserWorker, prompt: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool,
        on_progress=None, user_lora=None, enhance_prompt: bool = False,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        is_fast = model == "ltx-2-3-fast"

        worker.ensure_transformer("distilled" if is_fast else "dev", user_lora=user_lora)

        # Text encoding on GPU:0 (shared encoder) — page in, encode, page out
        self._page_encoder_to_gpu()
        if is_fast:
            (ctx_p,) = self._encode_prompts([prompt], enhance_first_prompt=enhance_prompt)
            (ctx_p,) = self._contexts_to_device([ctx_p], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = self._encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], enhance_first_prompt=enhance_prompt)
            ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
        self._page_encoder_to_cpu()  # free ~25 GB for denoising + decode

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Stage 1: half-resolution denoising
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = worker.ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=[], height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        transformer = BatchSplitAdapter(worker.ledger.transformer(), max_batch_size=1)

        # Create initial states for stage 1
        video_state, audio_state, v_tools, a_tools = self._create_av_states(
            stage_1_shape, fps, noiser, dtype, device, video_conds=stage_1_cond,
        )

        if is_fast:
            sigmas = DISTILLED_SIGMAS.to(device=device, dtype=torch.float32)
            s1_steps = len(sigmas) - 1

            denoiser = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
            if on_progress:
                denoiser = self._wrap_denoiser(denoiser, on_progress, s1_steps, offset=0.0, scale=0.7)
            if _use_cfg_pp:
                loop_fn = partial(euler_cfg_pp_loop, eta=_cfg_pp_eta_stage1, generator=generator)
            else:
                loop_fn = euler_denoising_loop
            video_state, audio_state = loop_fn(
                sigmas=sigmas, video_state=video_state, audio_state=audio_state,
                stepper=stepper, transformer=transformer, denoiser=denoiser)
        else:
            params = _DEV_PARAMS
            empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
            sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=params.num_inference_steps).to(dtype=torch.float32, device=device)
            s1_steps = len(sigmas) - 1

            denoiser = FactoryGuidedDenoiser(
                v_context=v_context_p, a_context=a_context_p,
                video_guider_factory=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n),
                audio_guider_factory=create_multimodal_guider_factory(params=params.audio_guider_params, negative_context=a_context_n),
            )
            if on_progress:
                denoiser = self._wrap_denoiser(denoiser, on_progress, s1_steps, offset=0.0, scale=0.7)
            if _use_cfg_pp:
                loop_fn = partial(euler_cfg_pp_loop, eta=_cfg_pp_eta_default, generator=generator)
            else:
                loop_fn = euler_denoising_loop
            video_state, audio_state = loop_fn(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, transformer=transformer, denoiser=denoiser)

        # Post-process stage 1
        video_state = v_tools.clear_conditioning(video_state)
        video_state = v_tools.unpatchify(video_state)
        audio_state = a_tools.clear_conditioning(audio_state)
        audio_state = a_tools.unpatchify(audio_state)

        # Free transformer (~22GB) before upsample — not needed for upsample_video
        stage_1_latent = video_state.latent[:1]
        stage_1_audio_latent = audio_state.latent
        del video_state, audio_state, stage_1_cond, sigmas, transformer, v_tools, a_tools
        cleanup_memory()

        upscaled = upsample_video(latent=stage_1_latent, video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        del stage_1_latent

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=[], height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            worker.ensure_transformer("dev_lora", user_lora=user_lora)

        transformer = BatchSplitAdapter(worker.ledger.transformer(), max_batch_size=1)
        if _use_cfg_pp and _stage2_sigmas_override is not None:
            distilled_sigmas = torch.tensor(_stage2_sigmas_override, device=device, dtype=torch.float32)
        else:
            distilled_sigmas = STAGE_2_DISTILLED_SIGMAS.to(device=device, dtype=torch.float32)
        s2_steps = len(distilled_sigmas) - 1

        # Stage 2: create states with upscaled video as initial latent
        video_state, audio_state, v_tools, a_tools = self._create_av_states(
            stage_2_shape, fps, noiser, dtype, device,
            video_conds=stage_2_cond,
            video_noise_scale=float(distilled_sigmas[0]),
            audio_noise_scale=float(distilled_sigmas[0]),
            initial_video=upscaled, initial_audio=stage_1_audio_latent,
        )

        denoiser_s2 = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        if on_progress:
            denoiser_s2 = self._wrap_denoiser(denoiser_s2, on_progress, s2_steps, offset=0.7, scale=0.25)
        if _use_cfg_pp:
            loop_fn_s2 = partial(euler_cfg_pp_loop, eta=_cfg_pp_eta_default, generator=generator)
        else:
            loop_fn_s2 = euler_denoising_loop
        video_state, audio_state = loop_fn_s2(sigmas=distilled_sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, transformer=transformer, denoiser=denoiser_s2)

        # Post-process stage 2
        video_state = v_tools.clear_conditioning(video_state)
        video_state = v_tools.unpatchify(video_state)
        audio_state = a_tools.clear_conditioning(audio_state)
        audio_state = a_tools.unpatchify(audio_state)

        # Free everything before decode — evict transformer + cached models to reclaim VRAM
        video_latent = video_state.latent
        audio_latent = audio_state.latent
        del video_state, audio_state, stage_2_cond, upscaled, stage_1_audio_latent
        del transformer, denoiser_s2, distilled_sigmas, video_encoder, v_tools, a_tools
        worker.evict_transformer()
        for k in ("spatial_upsampler", "video_encoder"):
            worker.cache[k] = None
            worker.ledger._cache[k] = None
        gc.collect()
        torch.cuda.empty_cache()

        # Decode (decoders lazy-load here, with transformer/upsampler/encoder freed)
        _emit_phase(on_progress, 0.90, "decoding")
        decoded_video = _decode_video_fp32(video_latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        with _timed(f"audio_vae_decode job=t2v"):
            decoded_audio = vae_decode_audio(audio_latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        _emit_phase(on_progress, 0.95, "encoding")
        with _timed(f"video_decode+encode job=t2v"):
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

        # Text encoding on GPU:0 (shared encoder) — page in, encode, page out
        self._page_encoder_to_gpu()
        ctx_p, ctx_n = self._encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], enhance_first_prompt=enhance_prompt)
        ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
        self._page_encoder_to_cpu()  # free ~25 GB for denoising + decode

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = Res2sDiffusionStep()

        # Stage 1: half-resolution denoising with res2s sampler
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = worker.ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=[], height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        transformer = BatchSplitAdapter(worker.ledger.transformer(), max_batch_size=1)

        empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
        sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=hq_params.num_inference_steps).to(dtype=torch.float32, device=device)
        # res2s: 2 NFE per step + 1 final
        s1_nfe = 2 * hq_params.num_inference_steps + 1

        # Create initial states for stage 1
        video_state, audio_state, v_tools, a_tools = self._create_av_states(
            stage_1_shape, fps, noiser, dtype, device, video_conds=stage_1_cond,
        )

        denoiser = GuidedDenoiser(
            v_context=v_context_p, a_context=a_context_p,
            video_guider=MultiModalGuider(params=hq_params.video_guider_params, negative_context=v_context_n),
            audio_guider=MultiModalGuider(params=hq_params.audio_guider_params, negative_context=a_context_n),
        )
        if on_progress:
            denoiser = self._wrap_denoiser(denoiser, on_progress, s1_nfe, offset=0.0, scale=0.7)
        video_state, audio_state = res2s_audio_video_denoising_loop(
            sigmas=sigmas, video_state=video_state, audio_state=audio_state,
            stepper=stepper, transformer=transformer, denoiser=denoiser,
        )

        # Post-process stage 1
        video_state = v_tools.clear_conditioning(video_state)
        video_state = v_tools.unpatchify(video_state)
        audio_state = a_tools.clear_conditioning(audio_state)
        audio_state = a_tools.unpatchify(audio_state)

        # Free transformer before upsample — not needed for upsample_video
        stage_1_latent = video_state.latent[:1]
        stage_1_audio_latent = audio_state.latent
        del video_state, audio_state, stage_1_cond, sigmas, transformer, denoiser, v_tools, a_tools
        cleanup_memory()

        upscaled = upsample_video(latent=stage_1_latent, video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        del stage_1_latent

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=[], height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        worker.ensure_transformer("dev_lora_050", user_lora=user_lora)
        transformer = BatchSplitAdapter(worker.ledger.transformer(), max_batch_size=1)
        distilled_sigmas = STAGE_2_DISTILLED_SIGMAS.to(device=device, dtype=torch.float32)
        # res2s stage 2: 2 NFE per step + 1 final (3 distilled steps = 2 actual steps)
        s2_nfe = 2 * (len(distilled_sigmas) - 1) + 1

        # Stage 2: create states with upscaled video as initial latent
        video_state, audio_state, v_tools, a_tools = self._create_av_states(
            stage_2_shape, fps, noiser, dtype, device,
            video_conds=stage_2_cond,
            video_noise_scale=float(distilled_sigmas[0]),
            audio_noise_scale=float(distilled_sigmas[0]),
            initial_video=upscaled, initial_audio=stage_1_audio_latent,
        )

        denoiser_s2 = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        if on_progress:
            denoiser_s2 = self._wrap_denoiser(denoiser_s2, on_progress, s2_nfe, offset=0.7, scale=0.25)
        video_state, audio_state = res2s_audio_video_denoising_loop(
            sigmas=distilled_sigmas, video_state=video_state, audio_state=audio_state,
            stepper=stepper, transformer=transformer, denoiser=denoiser_s2,
        )

        # Post-process stage 2
        video_state = v_tools.clear_conditioning(video_state)
        video_state = v_tools.unpatchify(video_state)
        audio_state = a_tools.clear_conditioning(audio_state)
        audio_state = a_tools.unpatchify(audio_state)

        # Free everything before decode — evict transformer + cached models to reclaim VRAM
        video_latent = video_state.latent
        audio_latent = audio_state.latent
        del video_state, audio_state, stage_2_cond, upscaled, stage_1_audio_latent
        del transformer, denoiser_s2, distilled_sigmas, video_encoder, v_tools, a_tools
        worker.evict_transformer()
        for k in ("spatial_upsampler", "video_encoder"):
            worker.cache[k] = None
            worker.ledger._cache[k] = None
        gc.collect()
        torch.cuda.empty_cache()

        # Decode (decoders lazy-load here, with transformer/upsampler/encoder freed)
        _emit_phase(on_progress, 0.90, "decoding")
        decoded_video = _decode_video_fp32(video_latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        with _timed("audio_vae_decode job=t2v_hq"):
            decoded_audio = vae_decode_audio(audio_latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        _emit_phase(on_progress, 0.95, "encoding")
        with _timed("video_decode+encode job=t2v_hq"):
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

        # Text encoding on GPU:0 (shared encoder) — page in, encode, page out
        self._page_encoder_to_gpu()
        _enhance_image = images[0].path if images else None
        if is_fast:
            (ctx_p,) = self._encode_prompts([prompt], enhance_first_prompt=enhance_prompt, enhance_prompt_image=_enhance_image)
            (ctx_p,) = self._contexts_to_device([ctx_p], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = self._encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], enhance_first_prompt=enhance_prompt, enhance_prompt_image=_enhance_image)
            ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
        self._page_encoder_to_cpu()  # free ~25 GB for denoising + decode

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Stage 1
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = worker.ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=images, height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)
        transformer = BatchSplitAdapter(worker.ledger.transformer(), max_batch_size=1)

        if is_fast:
            sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device, dtype=torch.float32)
            s1_steps = len(sigmas) - 1
        else:
            params = _DEV_PARAMS
            empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
            sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=params.num_inference_steps).to(dtype=torch.float32, device=device)
            s1_steps = len(sigmas) - 1

        # Create initial states for stage 1
        video_state, audio_state, v_tools, a_tools = self._create_av_states(
            stage_1_shape, fps, noiser, dtype, device, video_conds=stage_1_cond,
        )

        if is_fast:
            denoiser = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        else:
            denoiser = FactoryGuidedDenoiser(
                v_context=v_context_p, a_context=a_context_p,
                video_guider_factory=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n),
                audio_guider_factory=create_multimodal_guider_factory(params=params.audio_guider_params, negative_context=a_context_n),
            )
        if on_progress:
            denoiser = self._wrap_denoiser(denoiser, on_progress, s1_steps, offset=0.0, scale=0.7)
        if _use_cfg_pp:
            # i2v/a2v: always eta=0 — ancestral noise destroys image/audio conditioning
            loop_fn = partial(euler_cfg_pp_loop, eta=0.0, generator=generator)
        else:
            loop_fn = euler_denoising_loop
        video_state, audio_state = loop_fn(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, transformer=transformer, denoiser=denoiser)

        # Post-process stage 1
        video_state = v_tools.clear_conditioning(video_state)
        video_state = v_tools.unpatchify(video_state)
        audio_state = a_tools.clear_conditioning(audio_state)
        audio_state = a_tools.unpatchify(audio_state)

        # Free transformer before upsample — not needed for upsample_video
        stage_1_latent = video_state.latent[:1]
        stage_1_audio_latent = audio_state.latent
        del video_state, audio_state, stage_1_cond, sigmas, transformer, denoiser, v_tools, a_tools
        cleanup_memory()

        upscaled = upsample_video(latent=stage_1_latent, video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        del stage_1_latent

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=images, height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            worker.ensure_transformer("dev_lora", user_lora=user_lora)

        transformer = BatchSplitAdapter(worker.ledger.transformer(), max_batch_size=1)
        if _use_cfg_pp and _stage2_sigmas_override is not None:
            distilled_sigmas = torch.tensor(_stage2_sigmas_override, device=device, dtype=torch.float32)
        else:
            distilled_sigmas = STAGE_2_DISTILLED_SIGMAS.to(device=device, dtype=torch.float32)
        s2_steps = len(distilled_sigmas) - 1

        # Stage 2: create states with upscaled video as initial latent
        video_state, audio_state, v_tools, a_tools = self._create_av_states(
            stage_2_shape, fps, noiser, dtype, device,
            video_conds=stage_2_cond,
            video_noise_scale=float(distilled_sigmas[0]),
            audio_noise_scale=float(distilled_sigmas[0]),
            initial_video=upscaled, initial_audio=stage_1_audio_latent,
        )

        denoiser_s2 = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        if on_progress:
            denoiser_s2 = self._wrap_denoiser(denoiser_s2, on_progress, s2_steps, offset=0.7, scale=0.25)
        if _use_cfg_pp:
            loop_fn_s2 = partial(euler_cfg_pp_loop, eta=_cfg_pp_eta_default, generator=generator)
        else:
            loop_fn_s2 = euler_denoising_loop
        video_state, audio_state = loop_fn_s2(sigmas=distilled_sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, transformer=transformer, denoiser=denoiser_s2)

        # Post-process stage 2
        video_state = v_tools.clear_conditioning(video_state)
        video_state = v_tools.unpatchify(video_state)
        audio_state = a_tools.clear_conditioning(audio_state)
        audio_state = a_tools.unpatchify(audio_state)

        # Free everything before decode — evict transformer to reclaim ~22GB VRAM
        video_latent = video_state.latent
        audio_latent = audio_state.latent
        del video_state, audio_state, stage_2_cond, upscaled, stage_1_audio_latent
        del transformer, denoiser_s2, distilled_sigmas, v_tools, a_tools
        worker.evict_transformer()
        for k in ("spatial_upsampler", "video_encoder"):
            worker.cache[k] = None
            worker.ledger._cache[k] = None
        gc.collect()
        torch.cuda.empty_cache()

        _emit_phase(on_progress, 0.90, "decoding")
        decoded_video = _decode_video_fp32(video_latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        with _timed("audio_vae_decode job=i2v"):
            decoded_audio = vae_decode_audio(audio_latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        _emit_phase(on_progress, 0.95, "encoding")
        with _timed("video_decode+encode job=i2v"):
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

        # Text encoding on GPU:0 (shared encoder) — page in, encode, page out
        self._page_encoder_to_gpu()
        _enhance_image = image_path
        if is_fast:
            (ctx_p,) = self._encode_prompts([prompt], enhance_first_prompt=enhance_prompt, enhance_prompt_image=_enhance_image)
            (ctx_p,) = self._contexts_to_device([ctx_p], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = self._encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], enhance_first_prompt=enhance_prompt, enhance_prompt_image=_enhance_image)
            ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
        self._page_encoder_to_cpu()  # free ~25 GB for denoising + decode

        # Audio encoding on GPU:0, then transfer to worker device
        decoded_audio = decode_audio_from_file(audio_path, self._encoder_device, max_duration=num_frames / fps)
        if decoded_audio is None:
            raise ValueError("No audio track found in the provided file")
        encoded_audio_latent = vae_encode_audio(decoded_audio, self._encoder_ledger.audio_encoder())
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / fps, channels=8, mel_bins=16)
        encoded_audio_latent = encoded_audio_latent[:, :, :audio_shape.frames].to(device)
        actual_frames = encoded_audio_latent.shape[2]
        if actual_frames < audio_shape.frames:
            pad = torch.zeros(
                encoded_audio_latent.shape[0], encoded_audio_latent.shape[1],
                audio_shape.frames - actual_frames, encoded_audio_latent.shape[3],
                device=encoded_audio_latent.device, dtype=encoded_audio_latent.dtype,
            )
            encoded_audio_latent = torch.cat([encoded_audio_latent, pad], dim=2)

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
        transformer = BatchSplitAdapter(worker.ledger.transformer(), max_batch_size=1)

        if is_fast:
            sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device, dtype=torch.float32)
            s1_steps = len(sigmas) - 1
        else:
            params = _DEV_PARAMS
            empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
            sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=params.num_inference_steps).to(dtype=torch.float32, device=device)
            s1_steps = len(sigmas) - 1

        # Create initial states for stage 1 (audio frozen)
        video_state, audio_state, v_tools, a_tools = self._create_av_states(
            stage_1_shape, fps, noiser, dtype, device, video_conds=stage_1_cond,
            initial_audio=encoded_audio_latent, freeze_audio=True,
        )

        if is_fast:
            denoiser = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        else:
            # Static guiders (not factory) — matches upstream A2VidPipelineTwoStage.
            # Factory guiders resolve per-step from sigma which weakens audio coupling
            # at low sigma values. Static guiders maintain consistent audio-video
            # guidance strength across all denoising steps.
            denoiser = GuidedDenoiser(
                v_context=v_context_p, a_context=a_context_p,
                video_guider=MultiModalGuider(params=params.video_guider_params, negative_context=v_context_n),
                audio_guider=MultiModalGuider(params=MultiModalGuiderParams()),
            )
        if on_progress:
            denoiser = self._wrap_denoiser(denoiser, on_progress, s1_steps, offset=0.0, scale=0.7)
        if _use_cfg_pp:
            # i2v/a2v: always eta=0 — ancestral noise destroys image/audio conditioning
            loop_fn = partial(euler_cfg_pp_loop, eta=0.0, generator=generator)
        else:
            loop_fn = euler_denoising_loop
        video_state, audio_state = loop_fn(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, transformer=transformer, denoiser=denoiser)

        # Post-process stage 1
        video_state = v_tools.clear_conditioning(video_state)
        video_state = v_tools.unpatchify(video_state)
        audio_state = a_tools.clear_conditioning(audio_state)
        audio_state = a_tools.unpatchify(audio_state)

        # Free transformer before upsample — not needed for upsample_video
        stage_1_latent = video_state.latent[:1]
        del video_state, audio_state, stage_1_cond, sigmas, transformer, denoiser, v_tools, a_tools
        cleanup_memory()

        upscaled = upsample_video(latent=stage_1_latent, video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        del stage_1_latent

        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=images, height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            worker.ensure_transformer("dev_lora", user_lora=user_lora)
        transformer = BatchSplitAdapter(worker.ledger.transformer(), max_batch_size=1)
        if _use_cfg_pp and _stage2_sigmas_override is not None:
            distilled_sigmas = torch.tensor(_stage2_sigmas_override, device=device, dtype=torch.float32)
        else:
            distilled_sigmas = STAGE_2_DISTILLED_SIGMAS.to(device=device, dtype=torch.float32)
        s2_steps = len(distilled_sigmas) - 1

        # Stage 2: create states with upscaled video as initial latent (audio frozen)
        video_state, audio_state, v_tools, a_tools = self._create_av_states(
            stage_2_shape, fps, noiser, dtype, device,
            video_conds=stage_2_cond,
            video_noise_scale=float(distilled_sigmas[0]),
            initial_video=upscaled, initial_audio=encoded_audio_latent,
            freeze_audio=True,
        )

        denoiser_s2 = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        if on_progress:
            denoiser_s2 = self._wrap_denoiser(denoiser_s2, on_progress, s2_steps, offset=0.7, scale=0.25)
        if _use_cfg_pp:
            loop_fn_s2 = partial(euler_cfg_pp_loop, eta=_cfg_pp_eta_default, generator=generator)
        else:
            loop_fn_s2 = euler_denoising_loop
        video_state, audio_state = loop_fn_s2(sigmas=distilled_sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, transformer=transformer, denoiser=denoiser_s2)

        # Post-process stage 2
        video_state = v_tools.clear_conditioning(video_state)
        video_state = v_tools.unpatchify(video_state)
        audio_state = a_tools.clear_conditioning(audio_state)
        audio_state = a_tools.unpatchify(audio_state)

        # Free everything before decode — evict transformer to reclaim ~22GB VRAM
        video_latent = video_state.latent
        del video_state, audio_state, stage_2_cond, upscaled
        del transformer, denoiser_s2, distilled_sigmas, v_tools, a_tools
        worker.evict_transformer()
        for k in ("spatial_upsampler", "video_encoder"):
            worker.cache[k] = None
            worker.ledger._cache[k] = None
        gc.collect()
        torch.cuda.empty_cache()

        # Decode video but return ORIGINAL audio (a2v passthrough)
        _emit_phase(on_progress, 0.90, "decoding")
        decoded_video = _decode_video_fp32(video_latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        original_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)
        _emit_phase(on_progress, 0.95, "encoding")
        with _timed("video_decode+encode job=a2v"):
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
        meta = get_videostream_metadata(video_path)
        fps_vid, num_frames, vid_width, vid_height = meta.fps, meta.frames, meta.width, meta.height

        # Text encoding on GPU:0 (shared encoder) — page in, encode, page out
        self._page_encoder_to_gpu()
        params = _DEV_PARAMS
        ctx_p, ctx_n = self._encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT])
        ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
        self._page_encoder_to_cpu()  # free ~25 GB for denoising + decode
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Evict transformer (~44GB) to make room for VAE encode (~46GB intermediates)
        worker.evict_transformer()
        gc.collect()

        # Encode input video on GPU:0, transfer to worker device
        video_encoder_enc = self._encoder_ledger.video_encoder()
        frame_gen = decode_video_from_file(path=video_path, device=self._encoder_device, max_duration=num_frames / fps_vid)
        video_conditioning = video_preprocess(frame_gen, vid_height, vid_width, dtype, self._encoder_device)
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

        transformer = BatchSplitAdapter(worker.ledger.transformer(), max_batch_size=1)

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

        # Create noised video and audio states with per-modality temporal masks
        v_shape = VideoLatentShape.from_pixel_shape(output_shape)
        v_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps_vid)
        video_state = create_noised_state(v_tools, video_conditionings, noiser, dtype, device, initial_latent=initial_video_latent)

        a_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
        a_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
        audio_state = create_noised_state(a_tools, audio_conditionings, noiser, dtype, device, initial_latent=initial_audio_latent)

        # Build denoiser with guiders (retake is single-stage, maps 0-0.95)
        denoiser = GuidedDenoiser(
            v_context=v_context_p, a_context=a_context_p,
            video_guider=MultiModalGuider(params=params.video_guider_params, negative_context=v_context_n),
            audio_guider=MultiModalGuider(params=params.audio_guider_params, negative_context=a_context_n),
        )
        if on_progress:
            denoiser = self._wrap_denoiser(denoiser, on_progress, total_steps, offset=0.0, scale=0.95)
        if _use_cfg_pp:
            loop_fn = partial(euler_cfg_pp_loop, eta=_cfg_pp_eta_default, generator=generator)
        else:
            loop_fn = euler_denoising_loop
        video_state, audio_state = loop_fn(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, transformer=transformer, denoiser=denoiser)

        video_state = v_tools.clear_conditioning(video_state)
        video_state = v_tools.unpatchify(video_state)
        audio_state = a_tools.clear_conditioning(audio_state)
        audio_state = a_tools.unpatchify(audio_state)

        # Evict transformer (~22GB) before VAE decode to avoid OOM
        del transformer
        worker.evict_transformer()
        gc.collect()

        _emit_phase(on_progress, 0.90, "decoding")
        decoded_video = _decode_video_fp32(video_state.latent, worker.ledger.video_decoder(), _get_decode_tiling(num_frames), generator)
        with _timed("audio_vae_decode job=retake"):
            decoded_audio = vae_decode_audio(audio_state.latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        _emit_phase(on_progress, 0.95, "encoding")
        with _timed("video_decode+encode job=retake"):
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
