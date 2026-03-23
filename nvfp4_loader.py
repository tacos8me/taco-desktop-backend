"""NVFP4 dequantizer for ComfyUI-format safetensors.

Loads NVFP4-quantized Gemma weights, dequantizes to BF16 via comfy-kitchen's
native CUDA kernel, and writes a standard safetensors file that the LTX
ModelLedger can load directly.

Requires: pip install comfy-kitchen
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def dequantize_nvfp4_safetensors(
    src_path: str | Path,
    key_prefix: str = "",
    device: str = "cuda:0",
) -> dict[str, torch.Tensor]:
    """Load an NVFP4 safetensors file and return a BF16 state dict.

    Uses comfy-kitchen's native CUDA dequantization kernel which correctly
    handles the swizzled block scale layout.

    Args:
        src_path: Path to the NVFP4 safetensors file.
        key_prefix: Prefix to prepend to all output keys (e.g. "language_model."
            to match HF Gemma3ForConditionalGeneration key naming).
        device: CUDA device for dequantization (tensors are moved to CPU after).
    """
    import comfy_kitchen as ck

    src_path = Path(src_path)
    logger.info("Loading NVFP4 safetensors from %s ...", src_path)

    sd = load_file(str(src_path), device=device)

    # Identify quantized layers by weight_scale companion keys
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

        packed = sd[weight_key]          # uint8 [rows, cols/2]
        block_scale = sd[scale_key]      # float8_e4m3fn [rows, cols/16] (swizzled)
        tensor_scale = sd[scale2_key]    # float32 scalar

        # Dequantize via comfy-kitchen native CUDA kernel
        dequantized = ck.dequantize_nvfp4(packed, tensor_scale, block_scale, torch.bfloat16)
        result[key_prefix + weight_key] = dequantized.cpu()

        processed.update([weight_key, scale_key, scale2_key, quant_key])

    # Pass through all non-quantized tensors
    for key, tensor in sd.items():
        if key not in processed:
            if key.endswith(".comfy_quant"):
                continue
            t = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            result[key_prefix + key] = t.cpu() if t.is_cuda else t

    logger.info("Dequantized %d layers → %d BF16 tensors", len(quantized_prefixes), len(result))
    return result


def convert_nvfp4_to_bf16(
    src_path: str | Path,
    dst_path: str | Path,
    key_prefix: str = "",
    merge_from: str | None = None,
) -> None:
    """Convert an NVFP4 safetensors to a standard BF16 safetensors file.

    Args:
        merge_from: Optional glob pattern for safetensors files to merge in
            (e.g. vision_tower weights from a standard checkpoint).
    """
    sd = dequantize_nvfp4_safetensors(src_path, key_prefix=key_prefix)

    if merge_from:
        import glob
        import safetensors
        for f in sorted(glob.glob(merge_from)):
            with safetensors.safe_open(f, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if k not in sd:
                        sd[k] = sf.get_tensor(k)
                        logger.debug("Merged key: %s", k)
        logger.info("Merged additional weights from %s", merge_from)

    dst_path = Path(dst_path)
    logger.info("Saving BF16 safetensors to %s (%d tensors) ...", dst_path, len(sd))
    save_file(sd, str(dst_path))
    logger.info("Done. %.1f GB", dst_path.stat().st_size / 1e9)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.safetensors> <output.safetensors> [--prefix PREFIX] [--merge-from GLOB]")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    prefix = ""
    merge = None
    args = sys.argv[3:]
    while args:
        if args[0] == "--prefix":
            prefix = args[1]
            args = args[2:]
        elif args[0] == "--merge-from":
            merge = args[1]
            args = args[2:]
        else:
            args = args[1:]
    convert_nvfp4_to_bf16(src, dst, key_prefix=prefix, merge_from=merge)
