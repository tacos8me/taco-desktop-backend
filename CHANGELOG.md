# Changelog

All notable changes to taco-backend. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## v1.5 — 2026-04-17

### Turbo-mode hardening: systemctl-stop replaces HTTP /unload for cuda:1 tenants

Root cause (observed in production today): `_enter_turbo_mode` called `await joyai.unload()` and `await ernie.unload()` via HTTP. Those requests can succeed on the wire while the sidecar's Python process keeps tensors resident. The subsequent `ltx_sidecar.load()` then tries to allocate ~46 GB of LTX transformer on a cuda:1 that still has 44 GB of JoyAI resident → CUDA OOM, turbo-enter fails mid-sequence, ACE already stopped, leaving the system in a broken state.

Fix (`server.py`):
- New `_systemctl_unit(unit, action)` helper — runs `systemctl --user <action> <unit>` in a thread, raises `RuntimeError` with stderr on non-zero exit. Replaces per-service `_ace_systemctl` (kept as back-compat alias).
- New `_stop_cuda1_tenants()` — stops `ace-step`, `joyai-sidecar`, `ernie-image-sidecar`, and any stale `ltx-sidecar` via systemctl. Best-effort; "already stopped" is not an error.
- New `_restore_cuda1_tenants()` — inverse: systemctl-start each configured tenant (`LOAD_*=1`). Called at turbo exit AND on turbo-entry abort rollback.
- New `_wait_cuda1_free(threshold_mib=2000, timeout_s=20.0)` — polls `nvidia-smi` until cuda:1 drops below the threshold. Returns False on timeout.
- New `_list_cuda1_processes()` — enumerates compute-app PIDs on the cuda:1 bus for diagnostics.
- `_enter_turbo_mode` rewritten: Flux unload → systemctl-stop all cuda:1 tenants → **wait for cuda:1 to drain (20 s deadline)** → abort with detailed error + tenant restore if not drained → systemctl-start ltx-sidecar → poll /health → /load → spawn workers. No more OOM on turbo entry.
- `_exit_turbo_mode` rewritten: HTTP /unload both sidecars (graceful) → systemctl-stop ltx-sidecar → `_restore_cuda1_tenants()`.

### LTX remote-sidecar pool (3-worker turbo)

Turbo mode previously topped out at 2 concurrent video workers (main cuda:0 in-process + local cuda:1 sidecar). v1.5 adds an OPTIONAL third worker that dispatches to a remote HTTP sidecar — e.g., Modal RTX Pro 6000 for overflow capacity.

- `config.py` — new env vars `LTX_REMOTE_SIDECAR_URL` + `LTX_REMOTE_SIDECAR_TOKEN`. When URL is empty (default), behavior is unchanged from v1.4. When set, turbo enter warms the remote via `/health` then spawns a third `worker_loop` dispatching via `_dispatch_job_turbo_remote`.
- `ltx_sidecar_client.py` — `LtxSidecarClient` gained `auth_token` + `label` kwargs. `_headers()` injects `Authorization: Bearer <token>` when configured. Module now exposes two instances: `ltx_sidecar` (local, label="local") + `ltx_remote_sidecar` (label="remote", or None if not configured).
- `server.py::_dispatch_job_turbo_remote` — routes to the remote client. **Unlike the local path, remote transport failures do NOT auto-exit turbo** — the remote is treated as optional extra capacity; jobs on that worker fail individually but main + local-sidecar workers keep serving.
- `_exit_turbo_mode` also /unloads the remote (saves Modal credit burn if the backing host is scale-to-pay like Modal).

Verified live: 3 concurrent fast t2v submissions → `processing: 3` in queue → 3 parallel workers → all completed in ~40 s wall-clock (vs ~90 s sequential).

### Companion: Modal RTX Pro 6000 LTX sidecar deployment

Scaffolded at `/mnt/nvme-1/servers/ltx-sidecar-modal/modal_app.py` (NOT in the taco-backend repo — ops tree). Includes:
- Custom image: debian_slim + torch cu130 + transformers 5.3.0 (pinned — Gemma3TextConfig attr mismatch in older transformers) + local `/mnt/nvme-1/repos/LTX-2` editable install (user has uncommitted `getattr(rope_local_base_freq)` fallback that upstream master lacks).
- `modal.Volume` at `/mnt/nvme-1/huggingface` pre-populated with 125 GB of LTX-2.3 checkpoints + Gemma 3 12B PT from HF.
- `@app.cls` + `@modal.enter` eager-loads the model per container boot. Cold start: ~60–80 s. Warm: instant.
- FastAPI app with Bearer-token middleware (secret `taco-sidecar-auth`).
- Public URL: `https://tacos8me--taco-ltx-sidecar-ltxsidecar-fastapi-app.modal.run`

Free Modal credit: $30/mo → ~10 hrs of RTX Pro 6000 ($3.03/hr). Scales to zero after 10 min idle (no burn when unused).

### Files changed

| File | Change |
|------|--------|
| `server.py` | `_enter_turbo_mode` + `_exit_turbo_mode` rewritten; new systemctl + cuda:1 drain helpers; `_dispatch_job_turbo_remote`; remote-worker spawn logic |
| `ltx_sidecar_client.py` | `auth_token` + `label` kwargs; module-level `ltx_remote_sidecar` instance |
| `config.py` | `LTX_REMOTE_SIDECAR_URL` + `LTX_REMOTE_SIDECAR_TOKEN` env vars |
| `CLAUDE.md` / `CHANGELOG.md` | version bump + docs |

---

## v1.4.1 — 2026-04-16

### Hot-fix

- **`import subprocess` missing from `server.py` top-level** — `_enter_turbo_mode` (line 1436) invokes `subprocess.run(["systemctl", "--user", "start", "ltx-sidecar"], ...)` inside a lambda, and `_warmup_page_cache` (line 529) invokes `subprocess.run` in an `asyncio.to_thread(...)` call. Both resolve `subprocess` via module scope, where it wasn't imported. Any `POST /v1/system/turbo {enable:true}` hit `NameError: name 'subprocess' is not defined` and left the system in a half-transitioned state (ACE stopped by `_ace_systemctl("stop")` before the lambda executed; LTX sidecar never started; no turbo dual-worker). Latent bug present since turbo mode landed in v1.2 — the page-cache warmup was also silently failing (`asyncio.create_task` swallowed the unhandled exception). The two functions that imported `subprocess` locally (`_ace_systemctl` at ~1078, `_query_gpu_info` at ~1324) worked fine, which masked the broader issue.

Symptom in production today: `POST /v1/system/turbo {"enable":true}` → 500 `turbo_toggle_failed`; queued gens stuck because main worker held `_inference_lock` while `_enter_turbo_mode` crashed mid-handshake.

Fix: one-line `import subprocess` at module top, covering all four call sites. No behavior change for previously-working paths.

**File**: `server.py`.

---

## v1.4 — 2026-04-16

Five semi-independent changesets landed together.

### Full-fidelity history capture (schema v2)

History DB now stores everything needed to re-run a generation exactly.

- **Schema v2 migration** — four new columns on `generations`: `params_json` (raw Pydantic request body with `storage://` URIs preserved), `gen_config_json` (LTX `_gen_config` snapshot at dispatch time, or `{turbo_steps, turbo_guidance}` for Flux-turbo), `seed` (resolved integer — auto-generated if client omits), `enhanced_prompt` (LTX prompt rewrite when `enhance_prompt=true`). Online `ALTER TABLE ADD COLUMN` gated on `PRAGMA user_version`; idempotent; old rows left intact with NULL new columns.
- **New endpoint** `GET /v2/history/{generation_id}` — returns the full record including parsed `params` + `gen_config`. Bearer auth, 404 for not-yours-or-not-found. `/v2/history` list endpoint shape unchanged (backward compat preserved).
- **Path → URI sanitizer** (`_sanitize_params_for_history`) rewrites `image_path` / `audio_path` / `video_path` / `source_audio_path` / `reference_audio_path` / `image_paths` list / `keyframes[].image_path` back to stable `storage://<uuid>` form before persistence.
- **Enhanced prompt plumbing** — `on_prompt_enhanced` callback threaded through `split_model_manager._encode_prompts` → 5 `_run_*` methods → 4 public async wrappers. Dispatcher captures the rewritten text onto `Job.enhanced_prompt`; worker_loop ships it to history.
- **Files**: `history_store.py`, `job_queue.py`, `server.py`, `split_model_manager.py`, `docs/API.md`, `CLAUDE.md`.

### Flux dashboard controls

Dashboard now has a collapsible "Flux" section exposing the turbo sub-parameters that previously required server restart to change.

- `_flux_config` dict (2 tunables: `turbo_steps` default 8, `turbo_guidance` default 2.5), persisted to `.flux_config.json`, survives restart.
- Endpoints: `GET /v1/system/flux-config`, `POST /v1/system/flux-config` (merge-update), `POST /v1/system/flux-config/reset`.
- `flux_manager._generate` / `_img2img` / `_edit` gained `turbo_steps` + `turbo_guidance` kwargs; dispatcher injects from `_flux_config`.
- `gen_config_snapshot` captures the turbo subset when `turbo=true` so history reflects what actually ran.
- **Files**: `server.py`, `flux_manager.py`, `dashboard.html`, `config.py`.

### PR 1 — perf quick-wins

Small, uncontroversial latency savings across every CFG-enabled video gen.

- **P1 negprompt cache** — `DEFAULT_NEGATIVE_PROMPT` encoded once per encoder lifecycle (nulled in `evict_all`); subsequent CFG-path gens skip the 0.4–0.8 s Gemma encode. Lives on encoder device (cuda:0); survives CPU↔GPU paging because the cached tensor is independent of the encoder's parameter tensors.
- **P4 redundant synchronize drop** — removed `torch.cuda.synchronize()` after `text_encoder.encode` in `_encode_prompts`. Same default stream already serializes; the subsequent `.to(target)` syncs implicitly.
- **P5 MP4 tmpfile on tmpfs** — `_video_to_bytes` now writes the PyAV intermediate to `/dev/shm` (verified tmpfs) via new `config.MP4_TMPDIR`. Saves 50–200 ms/job on ext-backed `/tmp` (confirmed by `stat -f /tmp` → ext2/ext3). Fallback to `/tmp` if shm missing.
- **P7 tqdm TTY auto-detect** — `tqdm(range(...), disable=None)` silences the per-step progress bar under systemd (no TTY). Cleans up journalctl, saves ~1 ms × steps.
- **O3 timed encoder** — wrapped `_encode_prompts` in `_timed("encode_prompts")`; acts as proof-of-P1 post-deploy (once root logger is configured to reach journal).
- **Files**: `split_model_manager.py`, `config.py`.

### PR 2 — ops resilience

Four independent defensive changes; O-A is live-verified, the other three activate only in failure modes.

- **O-A cancellation propagation** — `DELETE /v2/jobs/{id}` now actually stops the LTX denoiser. New `GenerationCancelledError` raised from `ProgressDenoiser.__call__` when `job.status == CANCELLED`; the sigma loop unwinds naturally. `worker_loop` distinguishes cancellation from failure (status → `cancelled`, not `failed`; no error recorded). Verified: GPU util 100 % → 0 % within 3 s of DELETE at 11 % progress. Previously the denoiser would have kept burning for ~25 more steps.
- **O-B LTX OOM recovery** — new `_oom_recovery(worker)` context manager + `@_with_oom_recovery` decorator applied to all five `_run_*` methods. On CUDA OOM: evict transformer + `cleanup_memory()`, then re-raise. Mirrors the `flux_manager.py` pattern; prevents the classic failure where a mid-VAE-decode OOM leaks ~22 GB into the allocator cache and OOMs every subsequent request.
- **O-C half-load recovery** — new `SplitModelManager.reset()` nulls workers + encoder_ledger + the neg-prompt cache, then per-GPU sync + `empty_cache()`. `_load_all_impl` wrapped so `_last_load_failed` is set on exception; `_ensure_ltx_resident` calls `reset()` before retry when the flag is up. Prevents blind `load_all()` retries against partially-populated GPU memory.
- **O-D sidecar crash → auto-exit turbo** — `_dispatch_job_turbo` catches `LtxSidecarError` with status 502/503/504 (transport failures) and schedules `_auto_exit_turbo_on_sidecar_failure` via `asyncio.create_task` so the next queued job doesn't fail the same way. Job-level errors (4xx) don't trigger.
- **Files**: `split_model_manager.py`, `server.py`, `job_queue.py`.

### PR 3 — cleanup audit (documentation + 2 safe drops)

Audited the 10 open-coded `gc.collect() + torch.cuda.synchronize() + torch.cuda.empty_cache()` triples in `split_model_manager.py`. Finding:

- **Do not dedup to `cleanup_memory()`**: the helper uses current-device sync, but our multi-GPU paths (`evict_transformer`, `evict_all`, `reset`) require explicit per-device sync (`torch.cuda.synchronize(self.device)` / `torch.cuda.synchronize(torch.device(device_name))`). Blind dedup would silently regress DUAL_GPU_LTX correctness.
- **Two sites are truly redundant**: back-to-back `gc.collect()` after `worker.evict_transformer()` in `_run_retake` (the encode-prep path and the pre-VAE-decode path). `evict_transformer` already calls gc+sync+empty internally; a second gc immediately after picks up nothing. Dropped both.
- **Eight sites are load-bearing**: added multi-line comments at `evict_transformer`, `ensure_transformer`, `_page_encoder_to_cpu`, `evict_all`, and the four identical inter-stage cleanup blocks in `_run_t2v` / `_run_t2v_hq` / `_run_i2v` / `_run_a2v`. Comments explain why the cleanup can't be safely dedup-ed (device-specific sync, multi-GPU loop, flush of just-cleared auxiliary model refs before VAE decode's ~15 GB peak). Future optimizers: don't strip these.

Measured savings: ~40 ms on retake only. The original "100–300 ms/job" estimate over-promised — most sites exist for memory-safety, not ceremony.

### Infra

- `.gitignore`: `.flux_config.json`, `*.bak-pr*`, `cufile.log`.
- Backup files `*.bak-pr1-*` / `*.bak-pr2-*` / `*.bak-pr3-*` on disk for rollback; ignored by git.

### Deferred (not shipped)

Tracked as task IDs in the working-set task list for future PRs:

- **#95** — Gemma `tokenizer.chat_template` missing: blocks `enhance_prompt=true` on all LTX modes. Discovered during v1.4 smoke. Enhance_prompt history column is wired and will populate correctly once the tokenizer is fixed.
- **#96** — Smoke test for remaining gen types (i2v/a2v/retake/i2i/image-edit/ernie) with uploaded source media. Text-to-image + text-to-video + music were smoke-tested; the source-media-requiring types need a manual test pass.
- **#100** PR 4 (feature trivials): G1 client seed on video, G7 retake boundary validation, G8 retake `enhance_prompt` field, G9 batch priority honoring, G10 stage-2 latent preservation on OOM.
- **#101** PR 5 (feature smalls): G2 custom `negative_prompt` on video, G3 HQ guider params aligned to upstream (`stg_scale=0.0, rescale_scale=0.45`), G5 per-request `gen_config` override.
- **#102** PR 6: P2 queue-aware encoder residency — skip `_page_encoder_to_cpu` when the next queued LTX job fits alongside the encoder (fast-mode batches save ~10 s / 5 jobs).
- **P8** (after #95 lands): cache `generate_enhanced_prompt` output keyed by `(prompt_hash, image_hash, seed)`; noodle-i Char loop repays the 1–3 s Gemma generation every iteration otherwise.
- **G4** (after #95 lands): route `enhance_i2v` for image-to-video mode; currently we always call `enhance_t2v` and ignore the reference image.
- **Structural**: pinned-host-memory base-weight cache for `ensure_transformer` — HQ jobs cycling `dev_lora_025 → dev_lora_050` mid-generation reload from disk every swap. Real refactor, own epic.

---

## v1.3 — 2026-04-13

- ERNIE-Image sidecar (8B DiT text-to-image on cuda:1 port 8094, Apache 2.0). Swaps with JoyAI on cuda:1; coexists with ACE.
- Dashboard advanced controls: 14 tunable LTX generation parameters (sampler, steps, scheduler shifts, CFG/STG/rescale/modality scales, stage-2 sigmas, eta). Persisted to `.gen_config.json` via `GET/POST /v1/system/config`.
- CFG++ sampler (euler_ancestral_cfg_pp ported from ComfyUI, adapted to LTX-2 CONST flow-matching) — now default.
- LTX-2.3 v1.1 distilled models.
- SingleGPUModelBuilder + CachingModelFactory replacing ModelLedger.
- BatchSplitAdapter on every transformer call.
- bf16 reduced-precision accumulation restored to PyTorch default (was previously False — caused character movement artifacts).

## v1.2

- Turbo mode: dual-GPU LTX via cuda:1 sidecar for 2 concurrent video workers.
- ACE music sidecar on cuda:1 (18 GB, xl-base + 4B LM via vLLM).
- JoyAI image-edit sidecar migrated from cuda:0 to cuda:1.
- Dashboard + `GET /v1/system/gpu` telemetry endpoint.
- Batch scheduler (1–50 items, priority stored).
- v2 job observability: phases, SSE stream, `/v2/jobs/{id}/preview`.

## v1.1

- Single-GPU swap mode (LTX ↔ Flux auto-swap on cuda:0).
- Keyframe symbolic indices (`"first" | "middle" | "last"` + negative ints).
- Flux LoRA folder-drop discovery (adapter mode, strength changes free).
- Fast-mode audio-to-video.
- History store (SQLite, WAL, thumbnails, 30-day retention).
- Approved images pipeline (noodle-i → noodle-v).

## v1.0

Initial release.
