# Models

[Back to README](../README.md)

## LTX Video Models (cuda:0)

LTX-2.3 powers all video generation -- text-to-video, image-to-video, audio-to-video, and temporal retake. Uses a 22B transformer with shared encoder hub (Gemma 3 12B text encoder + VAE + spatial upsampler).

| Model | ID | Steps | Speed (~5s 1080p) | VRAM | Best For |
|-------|-----|-------|-------------------|------|----------|
| Fast | `ltx-2-3-fast` | 8 | ~15 s | ~79 GB | Previews, rapid iteration |
| Pro | `ltx-2-3-pro` | 30 + 5 (stage 2) | ~65 s | ~79 GB | Production quality |
| HQ | `ltx-2-3-hq` | 15 + 5 (res2s) | ~90 s | ~79 GB | Final render, highest quality |

**Pipeline details:**
- **Fast**: Distilled transformer, 8 denoising steps, no CFG. Lowest latency.
- **Pro**: Dev transformer with 30 euler steps + CFG + STG, followed by 5 stage-2 refinement steps using `dev_lora`. Best balance of quality and speed.
- **HQ**: Dev + `distilled_lora@0.25` for 15 res2s steps, then `dev_lora@0.50` for stage 2. Highest fidelity but slowest.

**Common constraints:**
- Frame count snapped to `8k+1` (9, 17, 25, 33, 41, 49...)
- Resolution must be multiples of 64
- Duration: 0-30 seconds
- All models support LoRA (fusion mode -- different strengths require full transformer reload)

## Flux Image Models (cuda:0)

Flux 2 handles all image generation -- text-to-image, image-to-image, and multi-reference editing. Full bf16 precision (no FP8 quantization).

| Model | ID | Steps | Speed (1024x1024) | Precision | VRAM | LoRA |
|-------|-----|-------|-------------------|-----------|------|------|
| Dev | `flux2-dev` | 20-50 | ~50-90 s | bf16 | ~81 GB (CPU offload) | Yes |
| Dev Turbo | `flux2-dev` + `turbo:true` | 8 | ~25-35 s | bf16 | ~81 GB (CPU offload) | Yes |
| Klein KV | `flux2-klein` | 4 | ~3 s | bf16 | ~32 GB (resident) | Yes |

**Pipeline details:**
- **Dev**: `Flux2Pipeline` with `enable_model_cpu_offload`. Components page between CPU and GPU at call boundaries. Cannot be fully resident in bf16 (105.9 GB total > 96 GB). Highest image quality.
- **Dev Turbo**: Same pipeline as Dev but with 8-step custom sigma schedule (`FLUX_TURBO_SIGMAS`) and guidance_scale 2.5. Composable with LoRA -- attach the `flux2-turbo` folder-drop LoRA for full distilled fidelity.
- **Klein KV**: `Flux2KleinKVPipeline` loaded via `from_single_file()`. Full bf16 resident (~32 GB). 4-step distillation, ignores `guidance_scale` (no CFG). Ultra-fast for editing workflows.

**LoRA behavior:**
- Adapter mode (not fused) -- strength applied at inference time via `set_adapters()`
- Strength changes are free (~0 ms runtime operation)
- Model or LoRA file changes trigger full pipeline reload (~30-60 s for Dev)
- Cache key: `(model_name, lora_path)` -- strength is NOT in the key

**Why no FP8:** FP8 layerwise casting was removed in v1.1.1 after diagnosing screendoor/grid artifacts. The PEFT input-autocast hook forces compute back into FP8 when fused LoRA weights are present, creating structured dithering on non-standard grid points. ComfyUI also defaults to bf16 on high-VRAM hardware for the same reason.

## JoyAI Image Edit (cuda:1 sidecar)

Instruction-based single-image editing via an out-of-process sidecar.

| Property | Value |
|----------|-------|
| Model ID | `joyai-edit` |
| Steps | 30 |
| Latency | ~78 s (1024x1024) |
| VRAM | ~50 GB on cuda:1 |
| Port | `127.0.0.1:8092` |
| systemd unit | `joyai-sidecar.service` |
| Activation | `LOAD_JOYAI=1` in `.env` |

**Key constraints:**
- Exactly **1** `image_uri` required (single-image editing only)
- LoRA is **not supported** (returns 422)
- Prompts are plain English instructions -- the server wraps them in a chat template
- Falls back to `503 joyai_disabled` / `503 sidecar_unreachable` if unavailable; client should retry with `flux2-klein`
- Accessed via `/v1/image-edit` or `/v2/image-edit` with `model="joyai-edit"`

## ACE Music Generation (cuda:1 sidecar)

ACE Step (xl-base + LM) generates music from text prompts and lyrics.

| Property | Value |
|----------|-------|
| Architecture | xl-base + LM |
| Latency | ~2-10 s per track (varies by duration/steps) |
| VRAM | ~18 GB on cuda:1 |
| Port | `127.0.0.1:8001` |
| systemd unit | `ace-step.service` |
| Activation | `LOAD_ACE=1` in `.env` |
| Audio formats | mp3, flac, wav, wav32, opus, aac |

**Task types:**

| Task | Description | Source audio required |
|------|-------------|---------------------|
| `text2music` | Generate from text/lyrics | No |
| `cover` | Create a cover version | Yes |
| `repaint` | Repaint/restyle existing audio | Yes |
| `extract` | Extract stems from audio | Yes |
| `lego` | Lego-style audio manipulation | Yes |
| `complete` | Complete partial audio | Yes |

**Queue management:**
- Music queue cap: `MAX_MUSIC_PENDING` (default 5)
- Returns `429 music_queue_full` when exceeded
- Returns `503` when disabled or during turbo mode
- Phase for music jobs: `"generating"` (not `"denoising"`)

## Turbo Mode Throughput

When turbo mode is active (`POST /v1/system/turbo`), both GPUs are assigned to LTX with 2 concurrent denoiser workers. This doubles video generation throughput:

| Mode | Video workers | Video throughput | Image/Music |
|------|--------------|-----------------|-------------|
| Normal | 1 (cuda:0) | 1x | Available |
| Turbo | 2 (cuda:0 + cuda:1) | 2x | Unavailable (503) |

Turbo mode is intended for batch video processing. Entry takes ~20 s, exit ~15 s.

## Model Selection Guide

| Scenario | Recommended Model |
|----------|------------------|
| Quick image iteration / editing | `flux2-klein` (4 steps, ~3 s) |
| High-quality image generation | `flux2-dev` (50 steps, ~50-90 s) |
| Fast image generation | `flux2-dev` + `turbo:true` (8 steps, ~25-35 s) |
| Instruction-based image edit | `joyai-edit` (30 steps, ~78 s) |
| Video preview / iteration | `ltx-2-3-fast` (8 steps, ~15 s) |
| Production video | `ltx-2-3-pro` (35 steps, ~65 s) |
| Final render video | `ltx-2-3-hq` (20 steps, ~90 s) |
| Music generation | ACE `text2music` (~2-10 s) |
| Batch video processing | Enable turbo mode for 2x throughput |
