# taco-backend

Dual-GPU inference server for AI video, image, music generation, and image editing. Powers [noodle-i](https://i.noodlefinger.io) (image), [noodle-v](https://v.noodlefinger.io) (video), and [m.noodlefinger.io](https://m.noodlefinger.io) (music video).

**Version**: v1.13.0 (2026-04-23) — Live Workers dashboard panel + `GET /v1/system/workers` introspection; Modal pool max raised 4 → 10.
**Public API base URL**: `https://api.noodlefinger.io` *(Cloudflare-proxied — `taco.noodlefinger.io` was retired 2026-04-18, DNS no longer resolves)*

## Features

- **Video generation** -- LTX-2.3 (22B transformer, v1.1 distilled models) with text-to-video, image-to-video, audio-to-video, and temporal retake
  - **Chain conditioning** *(v1.12.0, Experimental)* -- multi-frame video-segment conditioning for seamless MusicVideo composition. `POST /v2/video/extract-segment` returns an MP4 of a contiguous 9-frame tail; pass as `segment_uri` on the next i2v/a2v clip and backend hard-pins 9 consecutive target pixel frames via a single VAE-encoded multi-frame latent. Eliminates subject drift across seams. Gated behind FE `flags.v112_seamless_segment`; legacy v1.11.5 keyframes path fully supported.
- **Video outpaint** *(v1.7.0)* -- IC-LoRA expands a source video's canvas to a larger target resolution; LoRA fills the black padding with temporally-consistent content. 9 placement positions, optional stage-2 skip for fast previews
- **CFG++ sampler** -- ported from ComfyUI's `euler_ancestral_cfg_pp`, default ON, togglable via `/v1/system/sampler` and dashboard
- **Image generation** -- Flux 2 Dev and Klein KV for text-to-image, image-to-image, and multi-reference editing
- **Identity preservation** *(v1.8.0)* -- optional `preserve_identity` flag on Klein image-edits; pulls denoised latents + attention features toward the first reference for subject/facial consistency under heavy prompt deviation. 3 presets (`balanced`, `faithful`, `loose`) + strength dial
- **Image editing** -- JoyAI instruction-based single-image editing via sidecar on cuda:1
- **ERNIE-Image** -- baidu/ERNIE-Image 8B DiT text-to-image via sidecar on cuda:1, ~11 s at turbo steps
- **Music generation** -- ACE Step xl-base+LM for text-to-music, covers, repainting, and stem extraction
- **Dual-GPU architecture** -- 2-tenant auto-swap on cuda:0 (LTX and Flux), ACE + JoyAI/ERNIE swap on cuda:1
- **Turbo mode + multi-provider pool** -- runtime toggle claims both GPUs for LTX (2 concurrent local workers); optional Modal-backed pool *(v1.6)* adds up to 10 remote workers *(max raised 4 → 10 in v1.13.0)*; v1.9.0 adds RunPod as a second provider alongside Modal (up to 2 more) for **up to 14 concurrent video workers** (2 local + 10 Modal + 2 RunPod), tunable live from the dashboard
- **Batch scheduler** -- submit up to 50 generation jobs in a single request, auto-sorted to minimize GPU swaps, per-item result download
- **Generation history + reproducibility** *(v1.4)* -- every completed v2 job persists to SQLite with raw request body, gen-config snapshot, seed, and enhanced prompt; thumbnails auto-generated
- **Dashboard** -- real-time GPU telemetry, sampler toggle, advanced generation controls (14 tunable params), remote-pool slider, management at `/dashboard`
  - **Live Workers panel** *(v1.13.0)* -- per-worker busy/idle status (local + Modal + RunPod) with in-flight job summary, backed by `GET /v1/system/workers`

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
│  (2-tenant swap,         │           │  + JoyAI (~50 GB) OR     │
│   auto-swapped on        │           │    ERNIE (~33 GB)        │
│   dispatch)              │           │  (JoyAI/ERNIE swap)      │
└──────────────────────────┘           └──────────────────────────┘
 ~79 GB active (LTX)                    ACE + JoyAI ~68 GB
 ~81 GB active (Flux Dev)               ACE + ERNIE ~51 GB
 Mutually exclusive — cannot             JoyAI ↔ ERNIE mutually
 coexist during forward pass             exclusive on cuda:1
```

LTX and Flux share cuda:0 and are mutually exclusive (combined ~160 GB > 96 GB physical). The dispatcher auto-swaps inside the inference lock -- clients never orchestrate it. cuda:1 hosts ACE (always resident) plus JoyAI or ERNIE-Image (mutually exclusive swap). Turbo mode (runtime) or DUAL_GPU_LTX (boot-time) claims both GPUs for LTX with 2 concurrent workers. See [GPU Architecture](docs/gpu-architecture.md) for swap latency and details.

## Endpoints overview

74 endpoints total. Key routes:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + model status |
| GET | `/dashboard` | GPU management dashboard |
| POST | `/v2/text-to-video` | Async video from text |
| POST | `/v2/image-to-video` | Async video from image(s) / keyframes |
| POST | `/v2/audio-to-video` | Async video synced to audio |
| POST | `/v2/retake` | Async re-render video segment |
| POST | `/v2/video-outpaint` *(v1.7.0)* | Async canvas expand via IC-LoRA |
| POST | `/v2/text-to-image` | Async image from text |
| POST | `/v2/image-to-image` | Async single-ref image edit |
| POST | `/v2/image-edit` | Async multi-ref edit (Flux) or instruction edit (JoyAI) |
| POST | `/v2/music` | Async music generation |
| POST | `/v2/batch` | Submit batch of 1-50 generation jobs |
| GET | `/v2/jobs/{id}/stream` | SSE live progress (preferred over polling) |
| GET | `/v2/jobs/{id}/result` | Download result (MP4/WEBP/audio) |
| POST | `/v1/system/turbo` | Toggle dual-GPU turbo mode |
| GET/POST | `/v1/system/pool` | Get state + scale multi-provider remote pool *(v1.6 modal, v1.9 runpod)* |
| POST | `/v1/system/pool/remote-workers/{provider}` | Scale one provider (modal\|runpod) *(v1.9.0)* |
| GET | `/v1/system/workers` *(v1.13.0)* | Live per-worker status (local + modal + runpod) |
| GET | `/uploads/get/{upload_id}` | Read back an upload *(v1.9.1)* |
| POST | `/v2/video/extract-frames` | Extract N frames from a video as storage:// PNGs (v1.10.0) |
| POST | `/v2/video/extract-segment` *(v1.12.0)* | Extract a contiguous 9/17/25/33-frame MP4 segment for chain conditioning |
| GET/POST | `/v1/system/sampler` | Get/toggle CFG++ vs Euler sampler |
| GET/POST | `/v1/system/config` | Get/update all generation parameters |
| POST | `/v1/system/pause` | Evict all models, free VRAM |
| GET | `/v2/batch/{id}/result/{index}` | Download individual batch item result |
| GET | `/v1/chat/completions` | Chat/vision proxy (OpenAI-compatible) |

All endpoints except `/health` require `Authorization: Bearer <api-key>`. Every generation endpoint has both sync (`/v1/...`) and async (`/v2/...`) variants, except `/v2/video-outpaint` which is async-only. See [API Reference](docs/API.md) for the full spec.

## Documentation

| Document | Contents |
|----------|----------|
| [API Reference](docs/API.md) | Full 74-endpoint spec with request/response schemas, error taxonomy, common types |
| [Quick Start Guide](docs/QUICKSTART.md) | 5-minute integration guide for frontend devs |
| [Outpaint Frontend Guide](docs/outpaint-frontend-guide.md) | `/v2/video-outpaint` integration (v1.7.0) |
| [Retake Frontend Guide](docs/retake-frontend-guide.md) | `/v2/retake` integration for noodle-v |
| [GPU Architecture](docs/gpu-architecture.md) | Dual-GPU layout, swap mechanics, turbo mode, remote pool, latency tables |
| [Models & Latency](docs/models.md) | All model specs (incl. LoRA registry), step counts, VRAM, speed benchmarks |
| [Configuration](docs/configuration.md) | Environment variables, .env, systemd units |
| [CLAUDE.md](CLAUDE.md) | Codebase onboarding guide for AI assistants |
| [CHANGELOG](CHANGELOG.md) | Per-version deltas back to v1.1 |

## Tech stack

Python 3.13 -- FastAPI -- PyTorch 2.11 (cu130, sm_120) -- diffusers 0.38.0.dev0 -- ltx-core 1.1.1 -- ltx-pipelines 1.1.1 -- Flux 2 Dev/Klein KV -- ACE Step -- peft -- comfy-kitchen -- uv
