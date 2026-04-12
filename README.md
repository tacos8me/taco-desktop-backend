# taco-backend

Dual-GPU inference server for AI video, image, music generation, and image editing. Powers [noodle-i](https://i.noodlefinger.io) (image), [noodle-v](https://v.noodlefinger.io) (video), and [m.noodlefinger.io](https://m.noodlefinger.io) (music video).

**Version**: v1.2 (2026-04-11)

## Features

- **Video generation** -- LTX-2.3 (22B transformer) with text-to-video, image-to-video, audio-to-video, and temporal retake
- **Image generation** -- Flux 2 Dev and Klein KV for text-to-image, image-to-image, and multi-reference editing
- **Image editing** -- JoyAI instruction-based single-image editing via sidecar on cuda:1
- **Music generation** -- ACE Step xl-base+LM for text-to-music, covers, repainting, and stem extraction
- **Dual-GPU architecture** -- 2-tenant auto-swap on cuda:0 (LTX and Flux), concurrent ACE + JoyAI on cuda:1
- **Batch scheduler** -- submit up to 50 generation jobs in a single request, auto-sorted to minimize GPU swaps
- **Turbo mode** -- claims both GPUs for LTX, 2 concurrent denoiser workers, 2x video throughput
- **Dashboard** -- real-time GPU telemetry and management at `/dashboard`

## Quick start

```bash
# Start the server
systemctl --user start taco-backend

# Health check
curl http://localhost:8090/health

# Generate an image (sync)
curl -X POST http://localhost:8090/v1/text-to-image \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat in space", "model": "flux2-klein", "width": 1024, "height": 1024, "num_inference_steps": 4}' \
  --output cat.webp

# Run tests
uv run --no-sync pytest tests/ -q -p no:cacheprovider
```

## Architecture

```
cuda:0 (RTX PRO 6000, 96 GB)          cuda:1 (RTX PRO 6000, 96 GB)
┌──────────────────────────┐           ┌──────────────────────────┐
│  LTX  ↔  Flux            │           │  ACE (~18 GB)            │
│  (2-tenant swap,         │           │  + JoyAI (~50 GB)        │
│   auto-swapped on        │           │  (coexisting, no swap)   │
│   dispatch)              │           │                          │
└──────────────────────────┘           └──────────────────────────┘
 ~79 GB active (LTX)                    Combined ~68 GB
 ~81 GB active (Flux Dev)               Fits within 96 GB budget
 Mutually exclusive — cannot
 coexist during forward pass
```

LTX and Flux share cuda:0 and are mutually exclusive (combined ~160 GB > 96 GB physical). The dispatcher auto-swaps inside the inference lock -- clients never orchestrate it. Turbo mode claims both GPUs for LTX with 2 concurrent workers. See [GPU Architecture](docs/gpu-architecture.md) for swap latency and details.

## Endpoints overview

58 endpoints total. Key routes:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + model status |
| GET | `/dashboard` | GPU management dashboard |
| POST | `/v2/text-to-video` | Async video from text |
| POST | `/v2/image-to-video` | Async video from image(s) / keyframes |
| POST | `/v2/audio-to-video` | Async video synced to audio |
| POST | `/v2/retake` | Async re-render video segment |
| POST | `/v2/text-to-image` | Async image from text |
| POST | `/v2/image-to-image` | Async single-ref image edit |
| POST | `/v2/image-edit` | Async multi-ref edit (Flux) or instruction edit (JoyAI) |
| POST | `/v2/music` | Async music generation |
| POST | `/v2/batch` | Submit batch of 1-50 generation jobs |
| GET | `/v2/jobs/{id}/stream` | SSE live progress (preferred over polling) |
| GET | `/v2/jobs/{id}/result` | Download result (MP4/WEBP/audio) |
| POST | `/v1/system/turbo` | Toggle dual-GPU turbo mode |
| POST | `/v1/system/pause` | Evict all models, free VRAM |
| GET | `/v1/chat/completions` | Chat/vision proxy (OpenAI-compatible) |

All endpoints except `/health` and `/v1/approved-images/events` require `Authorization: Bearer <api-key>`. Every generation endpoint has both sync (`/v1/...`) and async (`/v2/...`) variants. See [API Reference](docs/API.md) for the full spec.

## Documentation

| Document | Contents |
|----------|----------|
| [API Reference](docs/API.md) | Full 58-endpoint spec with request/response schemas |
| [Quick Start Guide](docs/QUICKSTART.md) | 5-minute integration guide for frontend devs |
| [GPU Architecture](docs/gpu-architecture.md) | Dual-GPU layout, swap mechanics, turbo mode, latency tables |
| [Models & Latency](docs/models.md) | All model specs, step counts, VRAM, speed benchmarks |
| [Configuration](docs/configuration.md) | Environment variables, .env, systemd units |

## Tech stack

Python 3.13 -- FastAPI -- PyTorch 2.11 (cu130, sm_120) -- diffusers 0.38.0.dev0 -- LTX-2.3 -- Flux 2 Dev/Klein KV -- ACE Step -- peft -- comfy-kitchen -- uv
