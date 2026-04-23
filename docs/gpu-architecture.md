# GPU Architecture

[Back to README](../README.md)

**Current version:** v1.13.0 (2026-04-23). Turbo-mode entry/exit hardened in v1.5; remote-sidecar pool added in v1.6; expanded to multi-provider (Modal + RunPod) in v1.9.0. v1.10.0 added seamless MusicVideo export via multi-frame chain conditioning; v1.12.0 replaced the 3-keyframes chain with a single video-segment conditioning. v1.13.0 raises the Modal pool cap from 4 → 10 and adds `GET /v1/system/workers` for live per-worker introspection (no GPU-topology changes). See [CHANGELOG](../CHANGELOG.md).

## Dual-GPU Layout

taco-backend runs on two RTX PRO 6000 Blackwell GPUs (96 GB each). Each GPU has a distinct role:

```
cuda:0 (96 GB)                          cuda:1 (96 GB)
+---------------------------------+     +---------------------------------+
|  LTX  <-->  Flux                |     |  ACE xl-base+LM    (~18 GB)    |
|  (2-tenant swap, auto-managed)  |     |  + JoyAI (~50 GB)              |
|                                 |     |    OR ERNIE-Image (~33 GB)     |
|  LTX active:  ~79 GB           |     |  --------------------------------|
|  Flux active: ~81 GB (Dev)     |     |  ACE+JoyAI:  ~68 GB            |
|               ~32 GB (Klein)   |     |  ACE+ERNIE:  ~51 GB            |
|                                 |     |  JoyAI <-> ERNIE swap          |
+---------------------------------+     +---------------------------------+
```

**cuda:0** hosts LTX (video) and Flux (image) as mutually exclusive tenants. Their combined active memory (~160 GB) exceeds the 96 GB physical limit, so the server auto-swaps between them on every inference request.

**cuda:1** hosts ACE (music, ~18 GB) alongside either JoyAI (image edit, ~50 GB) or ERNIE-Image (text-to-image, ~33 GB). JoyAI and ERNIE are mutually exclusive (combined ~83 GB with ACE exceeds 96 GB). ACE coexists with either tenant.

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

## Turbo Mode (v1.2, hardened v1.5)

Turbo mode temporarily claims **both** GPUs for LTX, enabling 2 concurrent local denoiser workers. Combined with the optional multi-provider remote-sidecar pool (v1.6 Modal, v1.9 RunPod; v1.13 Modal max raised to 10) it scales up to **14 concurrent video workers** (2 local + 10 Modal + 2 RunPod).

Toggle via `POST /v1/system/turbo` with body `{"enable": true/false}`.

### Entry (~20 s, v1.5 hardening)
1. Flux unload on cuda:0
2. **`systemctl --user stop`** on ALL cuda:1 tenants (`ace-step`, `joyai-sidecar`, `ernie-image-sidecar`, plus any stale `ltx-sidecar`) — HTTP `/unload` is no longer trusted because sidecar Python processes could hold tensors resident after a successful /unload response, causing OOM during the subsequent LTX sidecar load.
3. Wait for cuda:1 to drain below 2 GB (20 s deadline) via `nvidia-smi` polling. Abort and restore tenants if not drained.
4. `systemctl --user start ltx-sidecar`, poll `/health`, call `/load`
5. Spawn second in-process `worker_loop` pulling from `_job_queue`
6. For every configured provider in `ltx_remote_sidecars` (keyed by `"modal"`, `"runpod"`, …) with `_remote_worker_targets[p] > 0`, `_scale_remote_pool()` warms the provider's endpoint and spawns N additional worker loops bound to that provider via `functools.partial` (up to each provider's `MAX_WORKERS`).

### Active
- **2+ video jobs process concurrently** via dedicated worker loops (one per local GPU + one per remote container)
- `asyncio.gather` in the batch worker dispatches multiple items in parallel
- Flux, ACE, JoyAI, ERNIE, and music endpoints all return `503 turbo_mode_active`
- Only video generation endpoints are functional

### Exit (~15 s)
1. Cancel the second (and remote) worker tasks
2. HTTP `/unload` on local LTX sidecar; remote sidecars are left for Modal's native `scaledown_window` (5 min) to reclaim
3. `systemctl --user stop ltx-sidecar` on cuda:1
4. `_restore_cuda1_tenants()` — `systemctl --user start` each configured tenant (`LOAD_ACE` / `LOAD_JOYAI` / `LOAD_ERNIE`)

### Remote-Sidecar Pool (v1.5 single-provider → v1.9 multi-provider)

Optional extra workers dispatch to HTTP sidecars on remote hardware. v1.9.0 supports multiple providers side-by-side: Modal (RTX Pro 6000, $3.03/hr) and RunPod Load-Balancing Serverless (RTX PRO 6000 Blackwell, ~$2.66/hr). Each provider has its own URL, token, and max-worker cap; operators scale them independently via the dashboard or API.

- **Config per provider** in `.env` (all optional):
  - Modal: `LTX_MODAL_SIDECAR_URL`, `LTX_MODAL_SIDECAR_TOKEN`, `LTX_MODAL_MAX_WORKERS` (default 10 since v1.13.0; was 4 in v1.6–v1.12). Falls back to the legacy `LTX_REMOTE_SIDECAR_*` env vars if unset.
  - RunPod: `LTX_RUNPOD_SIDECAR_URL`, `LTX_RUNPOD_SIDECAR_TOKEN`, `LTX_RUNPOD_MAX_WORKERS` (default 2).
- **Scaling**:
  - Legacy `POST /v1/system/pool/remote-workers {"count": N}` — scales **modal only** (backwards compat).
  - Per-provider body `{"modal": N, "runpod": M}`.
  - RESTful `POST /v1/system/pool/remote-workers/{provider} {"count": N}` (v1.9.0).
  - Dashboard: two rows — Modal and RunPod — each with 0..MAX buttons.
- **Turbo-scoped**: workers only run while turbo is active, so non-video jobs queued outside turbo aren't stolen by remote workers that can only serve video.
- **Failure isolation**: remote transport failures do NOT auto-exit turbo — any remote is optional extra capacity; jobs on that provider fail individually but main + local-sidecar + other-provider workers keep serving.
- **State query**: `GET /v1/system/pool` returns `{turbo_active, providers: {modal: {configured, url, target, active, max}, runpod: {...}}, remote_*: <legacy aliases to modal>}`.
- **Media inlining**: `_dispatch_job_turbo_remote(job, *, provider)` base64-encodes local media (`image_path`, `audio_path`, `video_path`, keyframes) into the request body — neither Modal nor RunPod can see the host's `uploads/` filesystem (`v1.6.1`). v1.7.0 outpaint follows the same pattern for `video_path`. v1.12.0 adds `segment_b64` inlining for the new chain-conditioning segment MP4 (same mechanism as `video_b64`).
- **Per-provider LoRA paths**: `config.LTX_PROVIDER_LORAS_MOUNT` maps `"modal" → /mnt/nvme-1/huggingface/loras/`, `"runpod" → /runpod-volume/loras/`. The outpaint dispatch path rewrites the local `LORAS_DIR` prefix to the provider's mount before sending.
- **RunPod sidecar repo**: `/mnt/nvme-1/servers/ltx-sidecar-runpod/` — Dockerfile, `runpod_app.py` (FastAPI + `/ping` health probe), `download_weights.py`, `endpoint.yaml`. Mirrors `ltx-sidecar-modal/` shape.

## cuda:1 Tenants

ACE is always resident on cuda:1. JoyAI and ERNIE-Image are mutually exclusive — only one can be loaded at a time (combined ~83 GB with ACE would exceed 96 GB).

| Service | VRAM | Port | systemd unit | Env var |
|---------|------|------|--------------|---------|
| ACE Step xl-base+LM | ~18 GB | `127.0.0.1:8001` | `ace-step.service` | `LOAD_ACE=1` |
| JoyAI Image Edit | ~50 GB | `127.0.0.1:8092` | `joyai-sidecar.service` | `LOAD_JOYAI=1` |
| ERNIE-Image (8B DiT) | ~33 GB | `127.0.0.1:8094` | `ernie-image-sidecar.service` | `LOAD_ERNIE=1` |

| Configuration | VRAM | Headroom |
|---------------|------|----------|
| ACE + JoyAI | ~68 GB | ~28 GB |
| ACE + ERNIE | ~51 GB | ~45 GB |
| ACE only | ~18 GB | ~78 GB |

All are gated by env vars and run as separate systemd services. taco-backend proxies to them via httpx.

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
| ERNIE-Image t2i (turbo) | ~11 s | 8 steps, 1024x1024, sidecar on cuda:1 |
| **Turbo entry** | **~20 s** | Evict ACE + JoyAI/ERNIE + Flux, load dual-GPU LTX |
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
| `/v1/system/config` | GET/POST | Get/update generation config (14 LTX params, `.gen_config.json`) |
| `/v1/system/config/reset` | POST | Reset generation config to defaults |
| `/v1/system/pool` | GET | Remote-sidecar pool state (v1.6) |
| `/v1/system/pool/remote-workers` | POST | Scale remote worker count 0..MAX (v1.6) |
| `/v1/system/workers` | GET | Live per-worker state (local + modal + runpod) (v1.13.0) |
| `/v1/system/gpu` | GET | nvidia-smi telemetry (2 s cache) |
| `/dashboard` | GET | GPU management dashboard (no auth) |
