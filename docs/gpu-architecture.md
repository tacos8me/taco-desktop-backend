# GPU Architecture

[Back to README](../README.md)

## Dual-GPU Layout

taco-backend runs on two RTX PRO 6000 Blackwell GPUs (96 GB each). Each GPU has a distinct role:

```
cuda:0 (96 GB)                          cuda:1 (96 GB)
+---------------------------------+     +---------------------------------+
|  LTX  <-->  Flux                |     |  ACE xl-base+LM    (~18 GB)    |
|  (2-tenant swap, auto-managed)  |     |  JoyAI Image Edit  (~50 GB)    |
|                                 |     |  --------------------------------|
|  LTX active:  ~79 GB           |     |  Combined:          ~68 GB     |
|  Flux active: ~81 GB (Dev)     |     |  Headroom:          ~28 GB     |
|               ~32 GB (Klein)   |     |  No swap needed                |
+---------------------------------+     +---------------------------------+
```

**cuda:0** hosts LTX (video) and Flux (image) as mutually exclusive tenants. Their combined active memory (~160 GB) exceeds the 96 GB physical limit, so the server auto-swaps between them on every inference request.

**cuda:1** hosts ACE (music, ~18 GB) and JoyAI (image edit, ~50 GB) concurrently. Combined footprint of ~68 GB fits comfortably within 96 GB with no swapping required.

No third GPU exists on this box. Earlier references to `cuda:2` / RTX 4000 are stale.

### DUAL_GPU_LTX Mode (v1.3)

Boot-time alternative to turbo mode. Set `DUAL_GPU_LTX=1` in `.env` and restart.

- cuda:1 runs an LTX sidecar (`ltx_sidecar_client.py` → `127.0.0.1:8093`) as a second video worker
- Flux, ACE, and JoyAI are disabled at startup (`LOAD_FLUX`, `LOAD_ACE`, `LOAD_JOYAI` forced false)
- 2 concurrent video jobs without runtime toggling
- Unlike turbo mode, this is a persistent configuration — no entry/exit latency per batch
- Sidecar generate timeout: 600 s

## Auto-Swap Protocol (cuda:0)

Two helpers in `server.py` manage the LTX/Flux swap. Both **must** be called while holding `_inference_lock`:

| Helper | Trigger | Action | Cost |
|--------|---------|--------|------|
| `_ensure_ltx_resident()` | Any video request | No-op if LTX loaded; else calls `ltx_manager.load_all()` | 7-30 s cold load |
| `_ensure_flux_ready()` | Any Flux image request | No-op if LTX not loaded; else calls `ltx_manager.evict_all()` | ~3 s eviction |

**Key behaviors:**
- Wired into `_dispatch_job()` (v2 async queue) and all v1 sync handlers
- LTX is **not** auto-reloaded after a Flux request -- it stays evicted until the next video request
- Single-tenant workloads (all-video or all-image) incur zero swap overhead
- Mixed workloads pay a per-direction-change cost (see latency table below)

### VRAM Math

| Tenant | Components | Active VRAM |
|--------|-----------|-------------|
| LTX | 60 GB transformer + 19 GB encoder hub + decoder activations | ~79 GB |
| Flux Dev | 60 GB transformer + ~14 GB CPU-offload forward-pass peak | ~81 GB |
| Flux Klein | Full bf16 resident (transformer + text encoder + VAE) | ~32 GB |
| **Combined** | LTX + Flux Dev | **~160 GB > 96 GB** |

Flux Dev uses `enable_model_cpu_offload`, so its idle GPU footprint is near-zero (weights live on pinned CPU memory between requests). But during a forward pass, the components page onto the GPU and exceed what would fit alongside LTX.

### evict_all Leak Fix (v1.1.4)

Prior to v1.1.4, `DenoiserWorker` held strong refs to source model builders that kept ~22 GB of encoder hub pinned after eviction. Fix: explicitly null reference paths before dropping workers. Verified: cuda:0 drops from 66.9 GB to **683 MiB** after unload. (v1.3 refactored from `ModelLedger` → `SingleGPUModelBuilder` / `CachingModelFactory` but the eviction pattern is the same.)

## Turbo Mode (v1.2)

Turbo mode temporarily claims **both** GPUs for LTX, enabling 2 concurrent denoiser workers and 2x video throughput.

Toggle via `POST /v1/system/turbo` with body `{"enable": true/false}`.

### Entry (~20 s)
1. Evicts ACE, JoyAI, and Flux from both GPUs
2. Loads dual-GPU LTX with encoder hub on cuda:0 and 2 denoiser workers
3. Starts a second `worker_loop` pulling from `_job_queue`

### Active
- **2 video jobs process concurrently** via `asyncio.gather` in the batch worker
- Flux, ACE, JoyAI, and music endpoints all return `503 turbo_mode_active`
- Only video generation endpoints are functional

### Exit (~15 s)
1. Cancels the second worker
2. Evicts dual-GPU LTX
3. Restores single-GPU LTX on cuda:0
4. Restarts ACE + JoyAI on cuda:1

## cuda:1 Coexistence

ACE and JoyAI share cuda:1 without any swap mechanism:

| Service | VRAM | Port | systemd unit |
|---------|------|------|--------------|
| ACE Step xl-base+LM | ~18 GB | `127.0.0.1:8001` | `ace-step.service` |
| JoyAI Image Edit | ~50 GB | `127.0.0.1:8092` | `joyai-sidecar.service` |
| **Total** | **~68 GB** | | |
| **Available** | **~28 GB headroom** | | |

Both are gated by env vars (`LOAD_ACE=1`, `LOAD_JOYAI=1`) and run as separate systemd services. taco-backend proxies to them via httpx.

## Swap Latency Table

| Transition | Added Cost | Notes |
|------------|-----------|-------|
| Video to video | 0 s | LTX stays resident |
| Image to image (same model) | 0 s | Flux stays resident |
| Image to image (Dev to Klein) | ~20-30 s | Normal Flux model swap |
| **Flux to LTX** (image then video) | **+7-30 s** | Cold LTX load (~7 s warm page cache, ~30 s cold) |
| **LTX to Flux** (video then image) | **+3 s** | LTX eviction, then normal Flux forward pass |
| LoRA strength change | ~0 ms | Runtime `set_adapters` call, no reload |
| LoRA file change | ~30-60 s | Full pipeline unload + reload (Dev) |
| **Turbo entry** | **~20 s** | Evict ACE + JoyAI + Flux, load dual-GPU LTX |
| **Turbo exit** | **~15 s** | Evict dual-GPU LTX, restore single-GPU + sidecars |

## Manual Control Endpoints

For operators who want explicit control (all require Bearer auth):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/ltx/unload` | POST | Unload LTX from cuda:0 |
| `/v1/ltx/reload` | POST | Reload LTX to cuda:0 |
| `/v1/flux/unload` | POST | Unload Flux from cuda:0 |
| `/v1/flux/reload` | POST | Reload Flux to cuda:0 |
| `/v1/system/pause` | POST | Free all GPU VRAM, cancel queued jobs |
| `/v1/system/resume` | POST | Reload all models |
| `/v1/system/turbo` | POST | Toggle turbo mode |
| `/v1/system/sampler` | GET/POST | Get/toggle CFG++ vs Euler sampler |
| `/v1/system/gpu` | GET | nvidia-smi telemetry (2 s cache) |
| `/dashboard` | GET | GPU management dashboard (no auth) |
