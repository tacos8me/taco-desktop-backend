# taco-backend

FastAPI inference server for LTX-2 video generation and Flux 2 image generation, built as the backend for taco-desktop.

## Architecture

The server runs two model managers on separate GPUs:

- **LTX-2 (SplitModelManager)** on `cuda:0` (~59GB): Gemma 3 12B text encoder, embeddings processor, VAE encoders/decoders, 22B transformer (hot-swapped between distilled/dev/dev_lora checkpoints), spatial upsampler, audio decoder, vocoder
- **Flux 2 (FluxManager)** on `cuda:1` (~79GB): Flux2Pipeline with FP8 layerwise casting (float8_e4m3fn storage, bf16 compute) + Mistral text encoder

A shared `asyncio.Lock` serializes all inference across both managers to prevent CUBLAS errors from concurrent FP8 CUDA operations.

## API Endpoints

### Video (LTX-2)

| Method | Path | Request Body | Response |
|--------|------|-------------|----------|
| POST | `/v1/text-to-video` | `{prompt, model, resolution, duration, fps, generate_audio?, camera_motion?}` | `video/mp4` |
| POST | `/v1/image-to-video` | `{prompt, image_uri, model, resolution, duration, fps, generate_audio?}` | `video/mp4` |
| POST | `/v1/audio-to-video` | `{prompt, audio_uri, image_uri?, model, resolution, duration?, fps?}` | `video/mp4` |
| POST | `/v1/retake` | `{video_uri, start_time, duration, mode, prompt?}` | `video/mp4` |

### Image (Flux 2)

| Method | Path | Request Body | Response |
|--------|------|-------------|----------|
| POST | `/v1/text-to-image` | `{prompt, model?, width?, height?, num_inference_steps?, guidance_scale?, seed?}` | `image/png` |
| POST | `/v1/image-to-image` | `{prompt, image_uri, model?, width?, height?, num_inference_steps?, guidance_scale?, seed?}` | `image/png` |

### Upload & Health

| Method | Path | Request Body | Response |
|--------|------|-------------|----------|
| GET | `/health` | -- | `{"status": "ok", "ltx": "ready", "flux": "ready"}` |
| POST | `/v1/upload` | -- | `{"upload_url", "storage_uri", "required_headers"}` |
| PUT | `/uploads/put/{id}` | raw bytes | `201 Created` |

## Upload Flow

Media files (images, audio, video) are passed to generation endpoints via `storage://` URIs:

1. `POST /v1/upload` -- returns an `upload_url` and a `storage_uri`
2. `PUT` raw bytes to the `upload_url`
3. Pass the `storage_uri` (e.g. `storage://abc123`) as `image_uri`, `audio_uri`, or `video_uri` in generation requests

Files are stored on disk in the `uploads/` directory, keyed by UUID.

## Models

| Name | Transformer | Stage 1 | Stage 2 | Notes |
|------|------------|---------|---------|-------|
| `ltx-2-3-fast` | distilled 22B | 8 steps, no CFG | 3 distilled refinement steps | Fastest (~15s) |
| `ltx-2-3-pro` | dev 22B | 30 steps with CFG | dev_lora 3 steps | Higher quality (~65s) |
| `flux2-dev` | Flux2 FP8 | 50 steps, guidance 4.0 | -- | Image generation |

All video generation uses a two-stage pipeline: half-resolution denoising followed by spatial upsampling and refinement at full resolution.

Retake mode values: `replace_audio_and_video`, `replace_video`, `replace_video_only`, `replace_audio`.

## Setup

**Requirements:** Python 3.13, `uv` package manager

```bash
# Install dependencies (cu130 torch for Blackwell GPUs)
uv sync
```

Key dependency notes:
- `torch`, `torchaudio`, `torchvision` pinned to cu130 index (sm_100 support for Blackwell)
- `ltx-core` and `ltx-pipelines` are editable path deps from `/mnt/nvme-1/repos/LTX-2/packages/`
- `diffusers` requires pre-release (`>=0.37.0.dev0`) for Flux2Pipeline support
- `uv run` must use `--no-sync` to avoid torch index resolution issues

## Running

```bash
./run.sh
```

`run.sh` sets:
- `LD_LIBRARY_PATH` -- adds cu130 nvidia libs from the venv (needed for nvrtc builtins)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` -- reduces CUDA memory fragmentation
- Runs `uvicorn server:app` on `0.0.0.0:8090`

## Configuration

All settings are in `config.py`:

| Setting | Value | Description |
|---------|-------|-------------|
| `LTX_DEVICE` | `cuda:0` | GPU for LTX-2 video generation |
| `FLUX_DEVICE` | `cuda:1` | GPU for Flux 2 image generation |
| `PORT` | `8090` | Server port |
| `CHECKPOINTS_DIR` | `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/` | LTX-2 model weights |
| `GEMMA_ROOT` | HF snapshot dir | Gemma 3 12B text encoder |
| `FLUX_MODEL_REPO` | `black-forest-labs/FLUX.2-dev` | Flux model identifier |
| `HF_CACHE_DIR` | `/mnt/nvme-1/huggingface/hub` | HuggingFace cache |
| `UPLOAD_DIR` | `./uploads` | Upload file storage |

## GPU Topology

PyTorch CUDA device indices do not match `nvidia-smi` bus order:

| PyTorch | nvidia-smi | GPU | Role |
|---------|-----------|-----|------|
| `cuda:0` | index 0 | RTX PRO 6000 96GB | LTX-2 |
| `cuda:1` | index 2 | RTX PRO 6000 96GB | Flux 2 |
| `cuda:2` | index 1 | RTX PRO 4000 24GB | unused |
