"""NVFP4 dequantizer for ComfyUI-format safetensors.

Loads NVFP4-quantized Gemma weights, dequantizes to BF16, and writes a
standard safetensors file that the LTX ModelLedger can load directly.

NVFP4 format (per quantized linear layer):
  - weight: U8 [rows, cols/2] — two FP4 E2M1 values packed per byte
  - weight_scale: F8_E4M3 [rows, cols/16] — per-block scale (group_size=16)
  - weight_scale_2: F32 scalar — per-tensor global scale
  - comfy_quant: U8 bytes — JSON metadata (ignored)

Dequant formula: bf16 = fp4_value * block_scale * tensor_scale
"""

from __future__ import annotations

import logging
import struct
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# FP4 E2M1 absolute value lookup table (3-bit unsigned magnitude)
# bit layout: [sign(1) | exponent(2) | mantissa(1)]
_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _unpack_nvfp4_to_bf16(packed: torch.Tensor) -> torch.Tensor:
    """Unpack U8 tensor of packed FP4 E2M1 values to bfloat16.

    Each byte contains two FP4 values:
      - bits [3:0] (low nibble) → even-indexed element
      - bits [7:4] (high nibble) → odd-indexed element
    """
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)

    lut = _FP4_LUT

    lo_val = lut[lo & 0x07]
    lo_val = torch.where(lo > 7, -lo_val, lo_val)

    hi_val = lut[hi & 0x07]
    hi_val = torch.where(hi > 7, -hi_val, hi_val)

    # Interleave: [rows, cols/2] → [rows, cols]
    result = torch.stack([lo_val, hi_val], dim=-1).reshape(packed.shape[0], -1)
    return result.to(torch.bfloat16)


def dequantize_nvfp4_safetensors(src_path: str | Path, key_prefix: str = "") -> dict[str, torch.Tensor]:
    """Load an NVFP4 safetensors file and return a BF16 state dict.

    Identifies quantized layers by the presence of weight_scale/weight_scale_2
    companion tensors. Non-quantized tensors (LayerNorms, biases, protected
    layers) are passed through as-is.

    Args:
        key_prefix: Prefix to prepend to all output keys (e.g. "language_model."
            to match HF Gemma3ForConditionalGeneration key naming).
    """
    src_path = Path(src_path)
    logger.info("Loading NVFP4 safetensors from %s ...", src_path)

    sd = load_file(str(src_path), device="cpu")

    # Identify quantized layers by weight_scale companion keys
    # Key example: model.layers.10.mlp.down_proj.weight_scale → prefix = model.layers.10.mlp.down_proj
    quantized_prefixes = set()
    for k in sd:
        if k.endswith(".weight_scale"):
            prefix = k.removesuffix(".weight_scale")
            if f"{prefix}.weight_scale_2" in sd:
                quantized_prefixes.add(prefix)

    logger.info("Found %d quantized layers, %d total tensors", len(quantized_prefixes), len(sd))

    result: dict[str, torch.Tensor] = {}
    processed = set()

    for prefix in sorted(quantized_prefixes):
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.weight_scale"
        scale2_key = f"{prefix}.weight_scale_2"
        quant_key = f"{prefix}.comfy_quant"

        packed = sd[weight_key]  # U8 [rows, cols/2]
        block_scale_raw = sd[scale_key]  # F8_E4M3 [rows, cols/16]
        tensor_scale = sd[scale2_key]  # F32 scalar

        # Unpack FP4 → BF16
        unpacked = _unpack_nvfp4_to_bf16(packed)  # [rows, cols]

        # Upcast block scales to float32 for multiplication
        block_scale = block_scale_raw.to(torch.float32)

        # Expand block scales to match weight dims (repeat each scale 16x)
        block_scale_expanded = block_scale.repeat_interleave(16, dim=1)

        # Handle case where cols isn't perfectly divisible by 16
        if block_scale_expanded.shape[1] > unpacked.shape[1]:
            block_scale_expanded = block_scale_expanded[:, :unpacked.shape[1]]

        # Dequantize
        dequantized = (unpacked * block_scale_expanded * tensor_scale.to(torch.float32)).to(torch.bfloat16)
        result[key_prefix + weight_key] = dequantized

        processed.update([weight_key, scale_key, scale2_key, quant_key])

    # Pass through all non-quantized tensors
    for key, tensor in sd.items():
        if key not in processed:
            if key.endswith(".comfy_quant"):
                continue  # skip metadata blobs
            result[key_prefix + key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor

    logger.info("Dequantized %d layers → %d BF16 tensors", len(quantized_prefixes), len(result))
    return result


def convert_nvfp4_to_bf16(src_path: str | Path, dst_path: str | Path) -> None:
    """Convert an NVFP4 safetensors to a standard BF16 safetensors file."""
    sd = dequantize_nvfp4_safetensors(src_path)
    dst_path = Path(dst_path)
    logger.info("Saving BF16 safetensors to %s ...", dst_path)
    save_file(sd, str(dst_path))
    logger.info("Done. %d tensors saved (%.1f GB)", len(sd), dst_path.stat().st_size / 1e9)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.safetensors> <output.safetensors>")
        sys.exit(1)
    convert_nvfp4_to_bf16(sys.argv[1], sys.argv[2])
