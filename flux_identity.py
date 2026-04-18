"""Flux 2 Klein identity preservation — v1.8.0.

Port of [`capitan01R/ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer#identity-preservation-nodes)'s
two training-free hooks:

1. :class:`IdentityGuidance` — latent-space pull toward reference during
   denoising, applied in ``callback_on_step_end`` within a sampling-percentage
   window. Three modes: ``direct``, ``adaptive`` (cosine-weighted),
   ``channel_match``.

2. :class:`IdentityFeatureTransfer` — ``register_forward_hook`` on Flux 2
   Klein's self-attention (``Flux2Attention`` inside each double-stream
   ``Flux2TransformerBlock``). Separates reference tokens from generation
   tokens in the attention output sequence and blends gen features toward
   ref features. Three modes: ``cosine_pull``, ``topk_replace``,
   ``mean_transfer``.

Both hooks are entered via :func:`identity_session`, a context manager that:

- resolves the user-facing 3-preset API (``balanced | faithful | loose``) +
  overall ``strength`` into per-hook mode + scaled strength,
- installs the attention hooks on the transformer,
- yields a combined ``callback_on_step_end`` that also chains any existing
  callback (progress tracker),
- tears everything down in a ``finally`` so repeated requests against the
  long-lived ``FluxManager._pipe`` don't leak state.

KV-cache caveat: Klein KV includes reference tokens in the attention
sequence only on step 0 (subsequent steps use cached K/V). The feature
transfer hook therefore only meaningfully fires on step 0 — this is
detected by comparing attention-output sequence length against the known
generation-token count. IdentityGuidance fires on every step to keep
identity pressure beyond step 0.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------

GuidanceMode = Literal["direct", "adaptive", "channel_match"]
TransferMode = Literal["cosine_pull", "topk_replace", "mean_transfer"]
IdentityPreset = Literal["balanced", "faithful", "loose"]

# Fixed plugin defaults (not user-exposed in v1.8.0)
_GUIDANCE_START_PCT = 0.0
_GUIDANCE_END_PCT = 0.5
_TOP_K_PERCENT = 50.0
# Target the middle-plus of the double-block stack. Plugin defaults were
# blocks 8..23 of 24; we use fractional bounds so the mapping survives any
# Klein variant with different block counts.
_BLOCK_START_FRAC = 0.25
_BLOCK_END_FRAC = 0.88
# Transfer hook strength is scaled down from the user's overall strength
# to preserve the plugin's 0.50:0.15 guidance:transfer ratio at dial=0.5.
_TRANSFER_STRENGTH_RATIO = 0.3

# If the blend op raises this many times in a row, re-raise and let the
# pipeline abort. Silently passing the raw attention output through on every
# step would produce a plausible-looking but identity-free image, which is
# worse than a hard failure — the user would have no signal that the request
# didn't do what was asked.
_MAX_BLEND_FAILURES = 5


@dataclass
class GuidanceConfig:
    mode: GuidanceMode
    strength: float
    start_pct: float = _GUIDANCE_START_PCT
    end_pct: float = _GUIDANCE_END_PCT


@dataclass
class TransferConfig:
    mode: TransferMode
    strength: float
    block_start: int
    block_end: int
    top_k_pct: float = _TOP_K_PERCENT


_PRESET_MAP: dict[IdentityPreset, tuple[GuidanceMode, TransferMode]] = {
    "balanced": ("adaptive", "cosine_pull"),
    "faithful": ("direct", "topk_replace"),
    "loose": ("channel_match", "mean_transfer"),
}


def _resolve_identity_preset(
    preset: IdentityPreset, strength: float, num_double_blocks: int,
) -> tuple[GuidanceConfig, TransferConfig]:
    """Map user preset + overall strength into per-hook configs."""
    if preset not in _PRESET_MAP:
        raise ValueError(f"unknown identity_mode: {preset!r}")
    g_mode, t_mode = _PRESET_MAP[preset]

    guidance_s = float(strength)
    transfer_s = float(strength) * _TRANSFER_STRENGTH_RATIO

    if num_double_blocks <= 0:
        block_start, block_end = 0, 0
    else:
        block_start = max(0, int(num_double_blocks * _BLOCK_START_FRAC))
        block_end = min(num_double_blocks - 1, int(num_double_blocks * _BLOCK_END_FRAC))

    return (
        GuidanceConfig(mode=g_mode, strength=guidance_s),
        TransferConfig(
            mode=t_mode, strength=transfer_s,
            block_start=block_start, block_end=block_end,
        ),
    )


# ---------------------------------------------------------------------------
# IdentityGuidance — latent-space pull toward reference
# ---------------------------------------------------------------------------


class IdentityGuidance:
    """Pull denoised latents toward reference inside a sampling window.

    ``reference_latent`` should already match the target-generation latent
    shape when passed in (caller resizes the reference image before VAE
    encoding). Shape-mismatch via :py:meth:`torch.Tensor.reshape` is
    attempted only when the element count matches (i.e., pack vs. unpack
    permutation).
    """

    def __init__(self, reference_latent: torch.Tensor, config: GuidanceConfig):
        self.reference = reference_latent
        self.config = config
        self._matched_ref: torch.Tensor | None = None

    def _match_shape(self, target: torch.Tensor) -> torch.Tensor:
        if self._matched_ref is not None and self._matched_ref.shape == target.shape:
            return self._matched_ref
        ref = self.reference.to(target.device, dtype=target.dtype)
        if ref.shape == target.shape:
            self._matched_ref = ref
            return ref
        if ref.numel() == target.numel():
            ref = ref.reshape(target.shape).contiguous()
            self._matched_ref = ref
            return ref
        raise ValueError(
            f"IdentityGuidance: ref {tuple(ref.shape)} ({ref.numel()} el) "
            f"cannot be matched to target {tuple(target.shape)} ({target.numel()} el). "
            "Caller must resize reference to match edit target before VAE encode."
        )

    def apply(self, denoised: torch.Tensor, step_pct: float) -> torch.Tensor:
        cfg = self.config
        if cfg.strength <= 0.0:
            return denoised
        if step_pct < cfg.start_pct or step_pct > cfg.end_pct:
            return denoised
        ref = self._match_shape(denoised)

        if cfg.mode == "direct":
            return denoised + cfg.strength * (ref - denoised)

        if cfg.mode == "adaptive":
            B = denoised.shape[0]
            d_flat = denoised.reshape(B, -1).float()
            r_flat = ref.reshape(B, -1).float()
            d_n = F.normalize(d_flat, dim=-1)
            r_n = F.normalize(r_flat, dim=-1)
            sim = (d_n * r_n).sum(dim=-1).clamp(0.0, 1.0)  # [B]
            broadcast_shape = [B] + [1] * (denoised.ndim - 1)
            sim_bcast = sim.reshape(broadcast_shape).to(denoised.dtype)
            return denoised + (cfg.strength * sim_bcast) * (ref - denoised)

        if cfg.mode == "channel_match":
            # 4D spatial: per-channel stats over H,W (dims 2,3)
            # 3D packed:  per-dim stats over token axis (dim 1)
            stat_dims = (2, 3) if denoised.ndim == 4 else (1,)
            d_mean = denoised.mean(dim=stat_dims, keepdim=True)
            d_std = denoised.std(dim=stat_dims, keepdim=True) + 1e-6
            r_mean = ref.mean(dim=stat_dims, keepdim=True)
            r_std = ref.std(dim=stat_dims, keepdim=True) + 1e-6
            matched = (denoised - d_mean) / d_std * r_std + r_mean
            return denoised + cfg.strength * (matched - denoised)

        return denoised


# ---------------------------------------------------------------------------
# IdentityFeatureTransfer — attention-output steering
# ---------------------------------------------------------------------------


class IdentityFeatureTransfer:
    """Forward-hook on Flux 2 Klein's double-block self-attention.

    The hook observes the image-stream attention output ``[B, T_img, D]``.
    On steps where T_img > ``expected_gen_tokens``, reference tokens are
    present in the sequence (step 0 of Klein KV); the hook separates the
    leading ``T_img - expected_gen_tokens`` reference tokens from the
    trailing generation tokens and blends the generation side toward the
    reference side via the configured mode. On steps where T_img equals
    ``expected_gen_tokens`` (Klein KV cached path, steps 1+), the hook
    returns output unchanged.
    """

    def __init__(self, config: TransferConfig, expected_gen_tokens: int):
        self.config = config
        self.expected_gen_tokens = expected_gen_tokens
        self._handles: list = []
        self._ref_token_count_seen: int = 0  # diagnostic
        self._consec_failures: int = 0

    def _blend(self, ref: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        strength = cfg.strength
        if strength <= 0.0:
            return gen

        if cfg.mode == "cosine_pull":
            ref_n = F.normalize(ref.float(), dim=-1)
            gen_n = F.normalize(gen.float(), dim=-1)
            sim = torch.bmm(gen_n, ref_n.transpose(1, 2))  # [B, T_gen, T_ref]
            best_sim, best_idx = sim.max(dim=-1)
            B, T_gen = best_idx.shape
            best_ref = torch.gather(
                ref, 1, best_idx.unsqueeze(-1).expand(B, T_gen, ref.shape[-1]),
            )
            weight = best_sim.clamp(0.0, 1.0).unsqueeze(-1).to(gen.dtype)
            return gen + (strength * weight) * (best_ref.to(gen.dtype) - gen)

        if cfg.mode == "topk_replace":
            k_pct = cfg.top_k_pct / 100.0
            T_gen = gen.shape[1]
            k = max(1, int(T_gen * k_pct))
            ref_n = F.normalize(ref.float(), dim=-1)
            gen_n = F.normalize(gen.float(), dim=-1)
            sim = torch.bmm(gen_n, ref_n.transpose(1, 2))
            best_sim, best_idx = sim.max(dim=-1)  # [B, T_gen]
            # Top-K gen positions by similarity
            _, topk_positions = best_sim.topk(k, dim=-1)  # [B, k]
            result = gen.clone()
            B = gen.shape[0]
            for b in range(B):
                positions = topk_positions[b]  # [k]
                ref_idx = best_idx[b, positions]  # [k]
                ref_matches = torch.index_select(ref[b], 0, ref_idx)  # [k, D]
                old = result[b, positions]
                result[b, positions] = old + strength * (ref_matches.to(gen.dtype) - old)
            return result

        if cfg.mode == "mean_transfer":
            ref_mean = ref.mean(dim=1, keepdim=True)
            gen_mean = gen.mean(dim=1, keepdim=True)
            shift = (ref_mean - gen_mean).to(gen.dtype)
            return gen + strength * shift

        return gen

    def _hook_fn(self, module, args, output):
        # Flux2Attention.forward returns (attn_output, context_attn_output).
        if not isinstance(output, tuple) or len(output) != 2:
            return output
        attn_output, context_attn_output = output
        if not isinstance(attn_output, torch.Tensor) or attn_output.ndim != 3:
            return output

        T_total = attn_output.shape[1]
        if T_total <= self.expected_gen_tokens:
            # Steps 1+ in Klein KV: ref tokens are cached, not in sequence.
            return output
        ref_count = T_total - self.expected_gen_tokens
        if self._ref_token_count_seen == 0:
            self._ref_token_count_seen = ref_count
            logger.info(
                "[identity] transfer first hit: T_total=%d ref=%d gen=%d",
                T_total, ref_count, self.expected_gen_tokens,
            )
        ref_toks = attn_output[:, :ref_count, :]
        gen_toks = attn_output[:, ref_count:, :]
        try:
            gen_modified = self._blend(ref_toks, gen_toks)
        except Exception as exc:
            self._consec_failures += 1
            if self._consec_failures == 1:
                logger.warning(
                    "IdentityFeatureTransfer blend failed (ref=%s gen=%s): %s",
                    tuple(ref_toks.shape), tuple(gen_toks.shape), exc,
                )
            if self._consec_failures >= _MAX_BLEND_FAILURES:
                logger.error(
                    "IdentityFeatureTransfer: %d consecutive blend failures — "
                    "re-raising so identity_session tears down cleanly",
                    self._consec_failures,
                )
                raise
            return output
        else:
            self._consec_failures = 0
        if gen_modified is gen_toks:
            return output
        new_attn = torch.cat([ref_toks, gen_modified], dim=1)
        return (new_attn, context_attn_output)

    def install(self, transformer: torch.nn.Module) -> int:
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None or len(blocks) == 0:
            logger.warning("IdentityFeatureTransfer: transformer has no transformer_blocks; skipping")
            return 0
        cfg = self.config
        for i in range(cfg.block_start, cfg.block_end + 1):
            if i >= len(blocks):
                break
            block = blocks[i]
            attn_module = getattr(block, "attn", None)
            if attn_module is None:
                logger.warning("IdentityFeatureTransfer: block %d has no .attn; skipping", i)
                continue
            handle = attn_module.register_forward_hook(self._hook_fn)
            self._handles.append(handle)
        logger.info(
            "IdentityFeatureTransfer: installed %d hooks on double blocks %d..%d (mode=%s, strength=%.3f)",
            len(self._handles), cfg.block_start, cfg.block_end, cfg.mode, cfg.strength,
        )
        return len(self._handles)

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                logger.exception("hook remove failed")
        self._handles.clear()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@contextmanager
def identity_session(
    pipe,
    *,
    reference_latent: torch.Tensor,
    mode_preset: IdentityPreset,
    strength: float,
    num_inference_steps: int,
    expected_gen_tokens: int,
    base_callback=None,
):
    """Enter identity-preservation mode for one Klein pipeline call.

    Args:
        pipe: a ``Flux2KleinKVPipeline`` instance. Used only to reach
            ``pipe.transformer``.
        reference_latent: VAE-encoded reference (matched to the target
            generation latent shape by the caller). ``IdentityGuidance``
            pulls denoised latents toward this.
        mode_preset: one of ``"balanced" | "faithful" | "loose"``.
        strength: in [0, 1]. Scales both hooks proportionally (transfer
            strength is additionally scaled by 0.3 to keep the plugin's
            5:1.5 guidance:transfer ratio at dial=0.5).
        num_inference_steps: total steps, needed to compute per-step
            percentage for the guidance window.
        expected_gen_tokens: number of generation-side image tokens in the
            attention output sequence (i.e., without reference tokens).
            Hook fires only when observed sequence length is strictly
            greater than this. For Klein with VAE scale 8 and patch
            size 1, this is ``(height // 8) * (width // 8)``.
        base_callback: existing ``callback_on_step_end`` to chain (e.g.,
            progress tracker from :func:`make_flux_callback`). May be None.

    Yields:
        A single callable suitable for ``pipe(callback_on_step_end=...)``.

    On exit (including exception) all forward hooks are removed. Callers
    must not reuse the yielded callback after the context closes.
    """
    transformer = pipe.transformer
    num_double_blocks = len(getattr(transformer, "transformer_blocks", []))
    guidance_cfg, transfer_cfg = _resolve_identity_preset(
        mode_preset, strength, num_double_blocks,
    )
    guidance = IdentityGuidance(reference_latent, guidance_cfg)
    transfer = IdentityFeatureTransfer(transfer_cfg, expected_gen_tokens=expected_gen_tokens)

    def combined_callback(pipeline, step_idx, timestep, callback_kwargs):
        # Progress / cancellation first so cancellation short-circuits
        # identity work too.
        if base_callback is not None:
            try:
                result = base_callback(pipeline, step_idx, timestep, callback_kwargs)
                if isinstance(result, dict):
                    callback_kwargs = result
            except Exception:
                logger.exception("base_callback raised inside identity_session")
        latents = callback_kwargs.get("latents") if isinstance(callback_kwargs, dict) else None
        if isinstance(latents, torch.Tensor):
            step_pct = (step_idx + 1) / max(num_inference_steps, 1)
            try:
                corrected = guidance.apply(latents, step_pct)
                if corrected is not latents:
                    callback_kwargs["latents"] = corrected
            except Exception:
                logger.exception("IdentityGuidance.apply failed; passing through")
        return callback_kwargs

    installed = transfer.install(transformer)
    logger.info(
        "[identity] session ENTER preset=%s strength=%.2f guidance=(%s,%.2f) "
        "transfer=(%s,%.2f) hooks=%d",
        mode_preset, strength,
        guidance_cfg.mode, guidance_cfg.strength,
        transfer_cfg.mode, transfer_cfg.strength, installed,
    )
    try:
        yield combined_callback
    finally:
        transfer.remove()
        logger.info(
            "[identity] session EXIT — hooks removed, first-hit ref_tokens=%d",
            transfer._ref_token_count_seen,
        )
