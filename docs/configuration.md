# Configuration

[Back to README](../README.md)

## Environment Variables

All environment variables are read from the `.env` file in the project root (`/mnt/nvme-1/servers/taco-backend/.env`). Changes require a server restart.

### Feature Flags

| Variable | Default | Values | Description |
|----------|---------|--------|-------------|
| `LOAD_FLUX` | `false` | `1`, `true`, `yes` | Enable Flux image generation on cuda:0 |
| `LOAD_ACE` | `false` | `1`, `true`, `yes` | Enable ACE music generation sidecar on cuda:1 |
| `LOAD_JOYAI` | `false` | `1`, `true`, `yes` | Enable JoyAI image-edit sidecar on cuda:1 |
| `DUAL_GPU_LTX` | `false` | `1`, `true`, `yes` | Dedicate both GPUs to LTX video generation. 2 concurrent workers via sidecar on cuda:1. Disables Flux, ACE, JoyAI |
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
| `LTX_SIDECAR_URL` | `http://127.0.0.1:8093` | URL of the LTX video sidecar (used in DUAL_GPU_LTX mode) |

### Queue Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_MUSIC_PENDING` | `5` | Max concurrent music generation jobs. Returns `429 music_queue_full` when exceeded |
| `MAX_BATCH_QUEUE_DEPTH` | `5` | Max concurrent batch submissions |
| `MAX_BATCH_ITEMS` | `50` | Max items per single batch request (1-50) |

### Example `.env`

```env
LOAD_FLUX=1
LOAD_ACE=1
LOAD_JOYAI=1
GEMMA_VARIANT=default
MAX_MUSIC_PENDING=5
MAX_BATCH_QUEUE_DEPTH=5
MAX_BATCH_ITEMS=50
# DUAL_GPU_LTX=1    # Uncomment for dedicated dual-GPU LTX mode
# TORCH_COMPILE=1   # Uncomment to enable torch.compile (adds 60-120s warmup)
```

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
