# Configuration

[Back to README](../README.md)

**Current version:** v1.13.0 (2026-04-23). See [CHANGELOG](../CHANGELOG.md).

## Environment Variables

All environment variables are read from the `.env` file in the project root (`/mnt/nvme-1/servers/taco-backend/.env`). Changes require a server restart.

### Feature Flags

| Variable | Default | Values | Description |
|----------|---------|--------|-------------|
| `LOAD_FLUX` | `false` | `1`, `true`, `yes` | Enable Flux image generation on cuda:0 |
| `LOAD_ACE` | `false` | `1`, `true`, `yes` | Enable ACE music generation sidecar on cuda:1 |
| `LOAD_JOYAI` | `false` | `1`, `true`, `yes` | Enable JoyAI image-edit sidecar on cuda:1 |
| `LOAD_ERNIE` | `false` | `1`, `true`, `yes` | Enable ERNIE-Image text-to-image sidecar on cuda:1. Mutually exclusive with JoyAI |
| `DUAL_GPU_LTX` | `false` | `1`, `true`, `yes` | Dedicate both GPUs to LTX video generation. 2 concurrent workers via sidecar on cuda:1. Disables Flux, ACE, JoyAI, ERNIE |
| `TORCH_COMPILE` | `false` | `1`, `true`, `yes` | Compile transformer blocks for Inductor-optimized inference. First request after load takes ~60-120s warmup. Default OFF (no benefit on Blackwell with cuDNN FA4) |
| `GEMMA_VARIANT` | `default` | `default`, `sikaworld` | Text encoder variant. `default` = Google Gemma 3 12B PT (BF16). `sikaworld` = abliterated FP4 (uncensored, NVFP4 quantized) |

### Device Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LTX_DEVICE` | `cuda:0` | Device for LTX video generation |
| `FLUX_DEVICE` | `cuda:0` | Device for Flux image generation (shares cuda:0 with LTX in swap mode) |
| `TURBO_GPU_DEVICES` | `["cuda:0", "cuda:1"]` | GPUs assigned to LTX denoiser workers in turbo mode |

### Sidecar URLs

| Variable | Default | Description |
|----------|---------|-------------|
| `ACE_SIDECAR_URL` | `http://127.0.0.1:8001` | URL of the ACE Step music generation sidecar |
| `JOYAI_SIDECAR_URL` | `http://127.0.0.1:8092` | URL of the JoyAI image-edit sidecar |
| `ERNIE_SIDECAR_URL` | `http://127.0.0.1:8094` | URL of the ERNIE-Image text-to-image sidecar |
| `LTX_SIDECAR_URL` | `http://127.0.0.1:8093` | URL of the local LTX video sidecar on cuda:1 (used in turbo / `DUAL_GPU_LTX` modes) |

### Remote LTX Sidecar Pool (v1.5 → v1.9.0 multi-provider)

Optional HTTP sidecar(s) on remote hardware. v1.9.0 supports **multiple providers alongside each other** (Modal + RunPod), each scaled independently. Every provider adds 1..N extra concurrent video workers on top of the 2 local workers during turbo mode. Pool is turbo-scoped — workers only run while turbo is active.

**Legacy v1.6-v1.8**: `LTX_REMOTE_SIDECAR_URL` / `LTX_REMOTE_SIDECAR_TOKEN` / `LTX_REMOTE_SIDECAR_MAX_WORKERS` are still honored and aliased to the `modal` provider. No migration needed — old deployments keep working.

| Variable | Default | Description |
|----------|---------|-------------|
| `LTX_REMOTE_SIDECAR_URL` | `""` | **Legacy alias** — points at the `modal` provider when set. Prefer `LTX_MODAL_SIDECAR_URL` in new deployments. |
| `LTX_REMOTE_SIDECAR_TOKEN` | `""` | Legacy alias. Bearer token value (not env-var name). |
| `LTX_REMOTE_SIDECAR_MAX_WORKERS` | `10` | Legacy alias. Upper bound on the modal pool. *(default raised from 4 → 10 in v1.13.0)* |
| `LTX_MODAL_SIDECAR_URL` | legacy | Modal-specific base URL. Falls back to `LTX_REMOTE_SIDECAR_URL` if unset. |
| `LTX_MODAL_SIDECAR_TOKEN` | legacy | Modal Bearer token. |
| `LTX_MODAL_MAX_WORKERS` | `LTX_REMOTE_SIDECAR_MAX_WORKERS` (default `10` in v1.13.0) | Modal pool cap. Must NOT exceed the Modal app's `max_containers` in `modal_app.py`. |
| `LTX_RUNPOD_SIDECAR_URL` | `""` (disabled) | RunPod Load-Balancing Serverless base URL — typically `https://api.runpod.ai/v2/<endpoint_id>/lb`. |
| `LTX_RUNPOD_SIDECAR_TOKEN` | `""` | Bearer token matching the endpoint's `SIDECAR_AUTH_TOKEN` secret. |
| `LTX_RUNPOD_MAX_WORKERS` | `2` | RunPod pool cap. Must match or be ≤ the endpoint's `workers.max`. |

Scale at runtime via `POST /v1/system/pool/remote-workers` (per-provider dict or legacy `{count}`) or `POST /v1/system/pool/remote-workers/{provider}` (RESTful per-provider). See [API.md](API.md#post-v1systempoolremote-workers).

### Queue Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_MUSIC_PENDING` | `5` | Max concurrent music generation jobs. Returns `429 music_queue_full` when exceeded |
| `MAX_QUEUE_DEPTH` | `200` | Global pending-job ceiling across all bearers. v1.16.4 raised from 30. |
| `MAX_BATCH_QUEUE_DEPTH` | `30` | Max concurrent batch submissions. v1.16.4 raised from 5. |
| `MAX_BATCH_ITEMS` | `50` | Max items per single batch request (1-50) |
| `PER_KEY_QUEUE_CAP` | `100` | Max in-flight jobs per bearer. v1.16.4 raised from 15. |
| `PER_KEY_MUSIC_CAP` | `20` | Max in-flight music jobs per bearer. v1.16.4 raised from 5. |
| `PER_KEY_BATCH_CAP` | `20` | Max in-flight batch submissions per bearer. v1.16.4 raised from 5. |
| `AUTO_TURBO_IDLE_MINUTES` | `15` | Minutes of cuda:1 inactivity before an opportunistic turbo-mode elevation may occur |

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `MP4_TMPDIR` | `/dev/shm` (falls back to `/tmp`) | Temp dir for intermediate MP4 encode buffer. PyAV needs a path, and `/dev/shm` is tmpfs (pure RAM). Saves 50-200 ms/job vs ext-backed `/tmp`. v1.4+ |
| `TACO_API_KEY` | `""` | Additional API key appended to the set loaded from `.api_keys` (convenient for env-based deployments) |

### Example `.env`

```env
LOAD_FLUX=1
LOAD_ACE=1
LOAD_JOYAI=1
# LOAD_ERNIE=1           # Uncomment for ERNIE-Image (mutually exclusive with JoyAI on cuda:1)
GEMMA_VARIANT=default
MAX_MUSIC_PENDING=5
MAX_QUEUE_DEPTH=200
MAX_BATCH_QUEUE_DEPTH=30
MAX_BATCH_ITEMS=50
PER_KEY_QUEUE_CAP=100
PER_KEY_MUSIC_CAP=20
PER_KEY_BATCH_CAP=20
# DUAL_GPU_LTX=1    # Uncomment for dedicated dual-GPU LTX mode
# TORCH_COMPILE=1   # Uncomment to enable torch.compile (adds 60-120s warmup)
# Remote-sidecar pool (v1.5 → v1.9.0 multi-provider) — leave URLs empty to disable
# Legacy single-provider (still works, aliases to modal):
# LTX_REMOTE_SIDECAR_URL=https://tacos8me--taco-ltx-sidecar-ltxsidecar-fastapi-app.modal.run
# LTX_REMOTE_SIDECAR_TOKEN=your-modal-bearer-token
# LTX_REMOTE_SIDECAR_MAX_WORKERS=10
# v1.9.0 per-provider form:
# LTX_MODAL_SIDECAR_URL=https://tacos8me--taco-ltx-sidecar-ltxsidecar-fastapi-app.modal.run
# LTX_MODAL_SIDECAR_TOKEN=your-modal-bearer-token
# LTX_MODAL_MAX_WORKERS=10
# LTX_RUNPOD_SIDECAR_URL=https://api.runpod.ai/v2/<endpoint_id>/lb
# LTX_RUNPOD_SIDECAR_TOKEN=your-runpod-bearer-token
# LTX_RUNPOD_MAX_WORKERS=2
```

## Generation Config (`.gen_config.json`)

Runtime generation parameters are persisted to `.gen_config.json` in the project root. This file is auto-managed by the server -- edit via `GET/POST /v1/system/config` or the dashboard advanced controls panel.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampler` | `"cfg_pp"` | `"cfg_pp"` or `"euler"` |
| `fast_stage1_steps` | `8` | Distilled model denoising steps (4-20) |
| `pro_stage1_steps` | `30` | Pro/dev model denoising steps (10-50) |
| `scheduler_max_shift` | `2.05` | LTX2Scheduler max shift |
| `scheduler_base_shift` | `0.95` | LTX2Scheduler base shift |
| `cfg_scale` | `3.0` | Classifier-free guidance scale |
| `stg_scale` | `1.0` | Spatiotemporal guidance scale |
| `rescale_scale` | `0.7` | CFG rescale factor |
| `modality_scale` | `3.0` | Modality guidance scale |
| `stg_blocks` | `[28]` | STG block indices |
| `stage2_sigmas` | `[0.85, 0.725, 0.4219, 0.0]` | Stage 2 sigma schedule |
| `eta_stage1` | `1.0` | Ancestral noise for distilled stage 1 |
| `eta_default` | `0.0` | Ancestral noise for guided/stage 2 |

Changes take effect on the next generation request -- no restart needed. Use `POST /v1/system/config/reset` or the dashboard reset button to restore defaults.

## Authentication

API keys are stored in `/mnt/nvme-1/servers/taco-backend/.api_keys`, one key per line. When the file is empty or missing, authentication is disabled.

All endpoints except `/health` and `/dashboard` require:
```
Authorization: Bearer <api-key>
```

API keys are hashed with SHA-256 for storage in the history database -- raw keys are never persisted.

## systemd Services

taco-backend uses three systemd user services:

| Service | Unit name | Port | Description |
|---------|-----------|------|-------------|
| taco-backend | `taco-backend.service` | 8090 | Main FastAPI server (LTX + Flux + job queue + batch scheduler) |
| JoyAI sidecar | `joyai-sidecar.service` | 8092 | JoyAI image-edit model on cuda:1 |
| ACE Step | `ace-step.service` | 8001 | ACE music generation on cuda:1 |
| ERNIE-Image | `ernie-image-sidecar.service` | 8094 | ERNIE-Image text-to-image on cuda:1 (mutually exclusive with JoyAI) |
| LTX sidecar | `ltx-sidecar.service` | 8093 | LTX video worker on cuda:1 (DUAL_GPU_LTX mode only) |

### Service Management

```bash
# Main server
systemctl --user start taco-backend
systemctl --user stop taco-backend
systemctl --user restart taco-backend
systemctl --user status taco-backend

# Logs
journalctl --user -u taco-backend -f

# JoyAI sidecar
systemctl --user start joyai-sidecar
systemctl --user stop joyai-sidecar
systemctl --user status joyai-sidecar

# ACE Step sidecar
systemctl --user start ace-step
systemctl --user stop ace-step
systemctl --user status ace-step
```

All three services are independent. The sidecars can be started/stopped without affecting taco-backend. taco-backend proxies to them via httpx and gracefully returns `503` if a sidecar is unreachable.

## Port Assignments

| Port | Service | Binding |
|------|---------|---------|
| 8090 | taco-backend (main) | `0.0.0.0:8090` |
| 8092 | JoyAI image-edit sidecar | `127.0.0.1:8092` |
| 8001 | ACE Step music sidecar | `127.0.0.1:8001` |
| 8093 | LTX video sidecar | `127.0.0.1:8093` |
| 8094 | ERNIE-Image sidecar | `127.0.0.1:8094` |

Sidecars bind to localhost only -- they are not directly accessible from the network. All external traffic goes through taco-backend on port 8090.

## Launcher: run.sh

The `run.sh` script sets up the runtime environment and starts uvicorn:

- Sets `LD_LIBRARY_PATH` to include cuDNN and cuBLAS libraries
- Configures `PYTORCH_CUDA_ALLOC_CONF` for optimal GPU memory allocation
- Launches uvicorn on port 8090

```bash
bash run.sh
```

For production, use the systemd service instead of running `run.sh` directly.

## Precision Settings (config.py)

| Setting | Value | Reason |
|---------|-------|--------|
| `torch.backends.cuda.matmul.allow_tf32` | `False` | Full float32 precision for VAE decode |
| `torch.backends.cudnn.allow_tf32` | `False` | Full float32 for VAE convolutions |
| `torch.backends.cudnn.deterministic` | `True` | Stable algorithm selection |

TF32 was previously enabled but degraded VAE output quality on Blackwell GPUs. The VAE uses `force_upcast=True` expecting real float32 math.

bf16 reduced precision accumulation is left at PyTorch default (`True`). LTX-2 was trained with default bf16 accumulation -- forcing float32 accumulation creates a training/inference mismatch.

## Testing

```bash
# Run all tests (IMPORTANT: --no-sync prevents nvidia package downgrades)
uv run --no-sync pytest tests/ -q -p no:cacheprovider

# Run specific test
uv run --no-sync pytest tests/test_flux_lora_registry.py -v
```

## Health Check

```bash
curl http://localhost:8090/health
```

Returns server status including loaded models, GPU state, and queue depth.
