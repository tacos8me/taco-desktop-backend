# System Overview

**Audience**: anyone landing on this repo cold who needs to understand
*what the thing is* before diving into how the parts fit. One altitude
above [`ARCHITECTURE.md`](ARCHITECTURE.md): that doc answers "how do
the parts fit"; this one answers "what is this thing at all".

**Date**: 2026-04-29 — taco-backend `v1.18.0-rc3`.

If you want endpoint shapes, see [`API.md`](API.md). If you want code
anchors, see [`../CLAUDE.md`](../CLAUDE.md). If you want the
implementation diagram, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 1. Service map

Every running process, what it speaks, where it listens, which GPU it
holds. Arrows = "calls". Dashed = "tees / fire-and-forget".

```
                            ┌──────────────────────────────────────────────┐
                            │   Browser (portal + noodle-i + noodle-v)     │
                            │   https://*.noodlefinger.io                  │
                            └───────────┬──────────────────┬───────────────┘
                                        │                  │
                                  Caddy │ forward_auth     │ direct (bearer)
                                        ▼                  │
                            ┌────────────────────────┐     │
                            │   noodlefinger-bff     │     │
                            │   FastAPI :8002        │     │
                            │   user_signals SQLite  │     │
                            └───────────┬────────────┘     │
                                        │                  │
                                        │  proxy           │
                                        ▼                  ▼
                            ┌──────────────────────────────────────────────┐
                            │           taco-backend (FastAPI :8090)       │
                            │           cuda:0  ◀── LTX  swap  Flux ──▶    │
                            │           jobs · history.db (WAL) · uploads/ │
                            └───┬─────────┬────────┬────────┬──────┬───────┘
                                │         │        │        │      │
            HTTP (sidecar)      │         │        │        │      │
                                │         │        │        │      │
              ┌─────────────────┘         │        │        │      └────────────┐
              │                  ┌────────┘        │        └─────┐             │
              ▼                  ▼                 ▼              ▼             ▼
   ┌────────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────┐
   │ ace-step    :8001  │ │ joyai  :8092 │ │ ltx    :8093 │ │ ernie    │ │ madmom     │
   │ cuda:1  music gen  │ │ cuda:1 edit  │ │ cuda:1 video │ │ :8094    │ │ :8095 CPU  │
   │  (always-on, LOAD_ │ │ (LOAD_JOYAI) │ │ (turbo or    │ │ cuda:1   │ │ (LOAD_     │
   │   ACE=1)           │ │              │ │  DUAL_GPU)   │ │ (swaps   │ │  MADMOM=1) │
   │                    │ │              │ │              │ │  w/joyai)│ │            │
   └────────────────────┘ └──────────────┘ └──────────────┘ └──────────┘ └────────────┘
                                                  │
                                ┌─────────────────┴────────────────┐
                                ▼                                  ▼
                  ┌───────────────────────────┐       ┌───────────────────────────┐
                  │ sapiens-sidecar    :8096  │       │ ltx-remote pools          │
                  │ cuda:1 (LOAD_SAPIENS=0,   │       │   Modal up to 10 workers  │
                  │  stub-mode default)       │       │   RunPod up to 2 workers  │
                  │ Validator tier-2          │       │ HTTPS bearer + base64-mp4 │
                  └───────────────────────────┘       └───────────────────────────┘

         taco-backend ─── HTTP ──▶ llama-swap @ 192.168.1.80:8080
                                   gemma-3-12b-nvfp4 (chat)
                                   gemma-4-31b-it     (vision/judge)
                                   gemma-3-12b-embed  (3584-d /v1/embeddings)

         noodlefinger-mcp (uvx subprocess inside Claude Code)
              ├── HTTP ─▶ taco-backend (tier-1 actions)
              └── HTTP ─▶ bff /api/mcp/events (fire-and-forget tee)
```

**GPU placement** (only two physical GPUs):

- **cuda:0** — RTX PRO 6000 96 GB. LTX video transformer (~79 GB) ↔ Flux
  image transformer (~81 GB Dev / ~32 GB Klein). Mutually exclusive on
  forward pass; auto-swapped per request (~3-30 s per direction change).
- **cuda:1** — RTX PRO 6000 96 GB. Always: ACE music (~18 GB). One of:
  JoyAI (~50 GB) ↔ ERNIE (~33 GB) (mutually exclusive). Optional: sapiens
  pose sidecar. In **turbo** or **DUAL_GPU_LTX** mode, cuda:1 is wholly
  reassigned to a second LTX worker.

**Process count** at rest on a healthy box:
`taco-backend.service` + `ace-step.service` + (`joyai-sidecar` OR
`ernie-image-sidecar`) + `madmom-sidecar` + (`sapiens-sidecar` if
opted-in) + `bff.service` + `caddy`. ~7 systemd user units typical.

---

## 2. Data plane diagram — request lifecycle

A single video generation traced from POST through `generations.id`.

```
        Client                              taco-backend                                 Storage
        ──────                              ────────────                                 ───────
   1.  POST /v2/audio-to-video ─────▶  server.py: route + Pydantic model validation
                                                 │
   2.                                  check_api_key middleware (bearer hash, opt CORS)
                                                 │
   3.                                  per-key rate-limit gate (PER_KEY_QUEUE_CAP)
                                                 │
   4.                                  job_queue.submit() → Job(status=QUEUED, id=uuid)
                                                 │
   5.  202 Accepted ◀──────────────  return {"job_id": ..., "status": "queued"}
                                                 │
                                                 ▼
                                       worker_loop dequeues
                                                 │
   6.                                  _ensure_ltx_resident()  ◀── evicts Flux if loaded
                                                 │
   7.                                  set Job.worker_id = "local-0"
                                                 │
   8.                                  _dispatch_job() → _run_a2v()  (split_model_manager)
                                                 │      phase = "denoising" (0-90%)
   9.                                  ProgressDenoiser sigma loop
                                                 │      phase → "decoding" (90-95%)
  10.                                  VAE decode, evict transformer
                                                 │      phase → "encoding" (95-99%)
  11.                                  PyAV H.264 encode → MP4 bytes
                                                 │      phase → "saving" (99-100%)
  12.                                  upload_store.put(mp4_bytes) ──────▶ uploads/<uuid>.mp4
                                                 │
  13.                                  history.save() ──────────────────▶ generations row
                                                 │      (api_key_hash, prompt, model,
                                                 │       result_uri, validator_score=NULL,
                                                 │       parent_clip_id, shot_uuid,
                                                 │       shot_config_key, lora_applied_id)
                                                 │
  14.                                  thumbnail extraction (PyAV) ─────▶ thumbnails/<uuid>.jpg
                                                 │
  15.                                  Job.status = COMPLETED
                                                 │
  16.                                  _decr_queue_on_complete(job)
                                                 │
                                                 ├── _on_job_complete(job)
                                                 │       │
                                                 │       └── _is_training_opted_in?
                                                 │             │
                                                 │             └── asyncio.create_task(
                                                 │                   _dispatch_validator(job)) ───┐
                                                 │                                                │
  17.  GET /v2/jobs/{id} (poll)    ◀──  Job(status=COMPLETED, result_uri=...)                     │
                                                                                                  │
                                       (fire-and-forget, ~5-15 s later)                           │
                                                                                                  ▼
                                       validator.run_all_tiers(video_path)
                                          │
                                          ├── stream-sha256 video → cache key
                                          ├── INSERT OR IGNORE validator_runs ◀── (sha, version) UNIQUE
                                          │
                                          ├── tier 1: RAFT (cuda:0, ~150 ms cold/hot) ──▶ flow stats
                                          ├── tier 2: sapiens sidecar (or stub) ────────▶ pose stats
                                          ├── tier 3: Gemma judge via llama-swap ───────▶ verdict
                                          │
                                          └── composite() → score + recommendation
                                                  │
                                                  └── UPDATE generations
                                                        SET validator_score = ?,
                                                            validator_payload_json = ?,
                                                            validator_version = ?
                                                        WHERE id = ?
```

Citations: `server.py:_on_job_complete` (~3724), `validator.py:run_all_tiers`,
`history_store.save` (history_store.py:829), `worker_loop` in `job_queue.py`.
The phase callback caps at 0.90 because steps 9-15 reserve the top 10%
of progress for post-denoise work (see CLAUDE.md §"v2 job observability").

---

## 3. Capture plane — every column on `generations` and what writes it

The `generations` table is the spine of the capture machine. Every
row is one completed job; every column is a different observation.

| Column | What it means | Writer | Schema since |
|---|---|---|---|
| `id` | UUID; primary key; matches `result_uri` filename stem | `worker_loop.history.save` | v1 |
| `api_key_hash` | sha256 of bearer; the privacy-gate join key | `_hash_key(api_key)` in `history_store` | v1 |
| `prompt`, `model`, `width`, `height`, `frames`, `fps` | denorm of the request | `worker_loop.history.save` from `body.model_dump()` | v1 |
| `result_uri` | `storage://<uuid>` for the result file | `upload_store.put()` then `history.save` | v1 |
| `params_json`, `gen_config_json`, `seed`, `enhanced_prompt` | full reproducibility | `_sanitize_params_for_history` then `save` | v2 |
| `validator_score` | composite 0..1 | `_dispatch_validator` (server.py) UPDATE post-tier-orchestration | v3 |
| `validator_payload_json` | full tier1+tier2+tier3 JSON | same | v3 |
| `validator_version` | `config.VALIDATOR_VERSION` at scoring time | same | v3 |
| `parent_clip_id` | `generations.id` of the clip being retaken; the strongest pair signal (`signal_strength=0.9`) | `POST /v2/retake` handler resolves via `find_id_by_result_uri()` and threads it into `_HISTORY_ONLY_PARAMS` | v3 |
| `shot_uuid` | opaque MCP shot identity; survives session resumes | MCP v0.7.0 `cut_music_video` orchestrator forwards via `_HISTORY_ONLY_PARAMS` | v3 |
| `shot_config_key` | sha of (prompt, image, audio window, model, lora id+strength); the cohort join key for pair construction | MCP v0.7.0 forwards (computed orchestrator-side) | v3 |
| `composition_id` | denorm convenience; canonical inverted-index lives in `composition_clips` (DEAD-LETTER per CLAUDE.md) | originally MCP-set; canonical writer is `composition_clips` insert in `POST /v2/compositions/{id}/export` | v3 |
| `lora_applied_id`, `lora_applied_strength` | which LoRA was actually fused into the run | `_lora_applied_pair(body)` on every video v2 endpoint, threaded through `job.params` since v1.18.0-rc2 | v3 (write-fixed in rc2) |
| `prompt_embedding` (BLOB) | DEAD-LETTER. Originally the per-row embedding cache; superseded by `clip_embeddings` virtual table (sqlite-vec). Never written from rc1 forward. Removal candidate for v1.19+. | none post-rc1 | v3 |
| `motion_intent` | per-shot operator hint ("static", "dynamic", "slow tracking") forwarded to validator tier-3 | MCP v0.7.0 → server.py `analyze-motion` route forwards to `validator._run_tier3_judge` | v4 (rc1) |
| `embedding_model_version` | tags rows with which embedder model produced their `clip_embeddings` row | `chat_manager.embed` ingestion path | v4 |
| `ab_arm` (post-rc5, NOT YET WRITTEN) | Phase C A/B routing tag (e.g. `"baseline"` / `"candidate-T+1"`); NULL on every row today | planned: dispatch shim that stamps before `_dispatch_job` | not yet shipped — see ARCHITECTURE §11 |

Adjacent tables (one row each per concept): `composition_clips`
(per-export inverted index), `validator_runs` (per-(sha, version)
cache, UNIQUE-keyed), `preference_pairs` (Phase C corpus,
INSERT-OR-IGNORE keyed on `(chosen, rejected, signal_source)`),
`training_runs` (Phase C ledger), `api_key_metadata` (per-bearer
opt-in flag — the privacy-gate spine), `clip_embeddings` (vec0
virtual table, 3584-d float32).

---

## 4. Loop diagram — the flywheel

Four loops at three different cadences. Continuous = every request.
Weekly = systemd timers. Ad-hoc = operator-driven.

```
                                   ┌─────────────────────────────────────────┐
                                   │ DEPLOYED LoRA  (MCP_PRODUCTION_LORA)    │
                                   │ better outputs feed back into next gen  │
                                   └────────────────┬────────────────────────┘
                                                    │
       continuous                                   │  (per-request, transparent)
       ──────────                                   ▼
   ┌─────────────────────────┐         ┌──────────────────────────┐
   │ 1. CAPTURE              │ ──────▶ │ 2. VALIDATE              │
   │    (per generation,     │         │    (fire-and-forget       │
   │     write generations   │         │     after job complete,   │
   │     row + lineage cols) │         │     ~5-15 s downstream)   │
   └────────────┬────────────┘         └─────────────┬─────────────┘
                │                                    │
                │                                    ▼
                │                          generations.validator_score
                │                          validator_runs cache (sha, version)
                │
       weekly cron (Mon 04:00 UTC)
       ───────────────────────────
                │
                ▼
   ┌──────────────────────────────────────────────────────────┐
   │ scripts/construct_preference_pairs.py                    │
   │   4 sources, idempotent, version-scoped, opt-in-filtered │
   │     user_retake     0.9   (parent_clip_id lineage)       │
   │     composition_kept 0.5  (composition_clips ⋈ key)      │
   │     validator_pass   0.7  (shot-cohort threshold)        │
   │     validator_fail   0.3  (synthetic negatives)          │
   │   → INSERT OR IGNORE preference_pairs                    │
   │   → bumps .preference_pairs_watermark                    │
   └─────────────────────────────┬────────────────────────────┘
                                 │
                                 │  ad-hoc, operator-driven
                                 │  (waits for ~1000 pairs, ~6-8 weeks)
                                 ▼
   ┌──────────────────────────────────────────────────────────┐
   │ scripts/train_dpo_sft.py --execute                       │
   │   defense-in-depth dry-run by default                    │
   │   selects chosen_clip_ids, signal_strength ≥ cfg         │
   │   90/10 train/eval split, snapshot dataset.jsonl         │
   │   PEFT LoRA (rank=64, alpha=64), bf16,                   │
   │     paged_adamw_32bit, gradient_checkpointing            │
   │   ~50-60 GPU-hours per cycle                             │
   │   → write training_runs row (full repro metadata)        │
   │   → mark consumed pairs used_in_training_run_id          │
   │   → register lora_registry as candidate (NOT auto-deploy)│
   └─────────────────────────────┬────────────────────────────┘
                                 │
                                 │  weekly cron
                                 ▼
   ┌──────────────────────────────────────────────────────────┐
   │ scripts/ab_decision.py                                   │
   │   paired t-test on per-MV mean validator_score across    │
   │     ab_arm cohorts (MV grouping = composition_id)        │
   │   promote   ≥ +10% AND p < 0.05                          │
   │   deprecate ≤ -5%  AND p < 0.05                          │
   │   else: insufficient_samples or no_action                │
   │   AB_AUTO_PROMOTE=1 (default) writes training_runs;      │
   │   =0 reports without writing                             │
   └─────────────────────────────┬────────────────────────────┘
                                 │
                                 │  on promote: operator updates
                                 │  MCP_PRODUCTION_LORA in .env, restarts
                                 │
                                 │  on regression: POST /v1/system/lora/rollback
                                 │  → atomic .env rewrite, restart-required
                                 │
                                 ▼
                              (back to top)

       ad-hoc / version-bump
       ──────────────────────
                                 │
                                 ▼
   ┌──────────────────────────────────────────────────────────┐
   │ POST /v2/system/bulk-revalidate (admin-gated)            │
   │   re-run validator on rows where validator_version !=    │
   │   target; default dry_run=true                           │
   │   used after Gemma upgrade or JUDGE_PROMPT bump          │
   └──────────────────────────────────────────────────────────┘

       continuous / authoring
       ──────────────────────
                                 ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 3. RETRIEVE  (Phase B, manual today)                     │
   │   POST /v2/embeddings/search       — find_similar_shots  │
   │   POST /v2/embeddings/recommend-loras — recommend_loras  │
   │   ranking: 0.50 sim + 0.35 v_norm + 0.10 recency         │
   │            + 0.05 comp_kept                              │
   │   privacy-gated: WHERE api_key_hash = sha256(caller)     │
   └──────────────────────────────────────────────────────────┘
```

---

## 5. Control plane — how to drive the system

All operator surfaces in one place. Cross-link to `API.md` for shapes.

### Admin endpoints (bearer in `.admin_keys`)

| Endpoint | Purpose | Doc |
|---|---|---|
| `POST /v1/system/turbo` | Toggle turbo mode (cuda:1 → 2nd LTX worker) | [`gpu-architecture.md`](gpu-architecture.md) |
| `GET/POST /v1/system/pool` + `/pool/remote-workers[/{provider}]` | Modal + RunPod remote-worker scaling 0..MAX | [`gpu-architecture.md`](gpu-architecture.md) |
| `POST /v1/system/pause` / `/resume` | Acquire `_inference_lock`, evict, restore | [`API.md`](API.md) |
| `GET/POST /v1/system/config` + `/config/reset` | LTX generation parameters (sampler, sigmas, scales) | [`API.md`](API.md) |
| `GET/POST /v1/system/flux-config` + `/reset` | Flux turbo_steps/turbo_guidance | [`API.md`](API.md) |
| `GET/POST /v1/system/sampler` | Sampler/eta/stage2_sigmas subset alias | [`API.md`](API.md) |
| `POST /v1/ltx/unload` / `/v1/ltx/reload` | Manual LTX eviction | [`API.md`](API.md) |
| `POST /v1/flux/unload` / `/v1/flux/reload` | Manual Flux eviction | [`API.md`](API.md) |
| `POST /v2/system/bulk-revalidate` | Re-validator on rows where version != target; dry-run by default | [`RETRIEVAL_WORKFLOW.md`](RETRIEVAL_WORKFLOW.md) |
| `POST /v1/system/lora/rollback` | Atomic `.env` rewrite to previous deployed LoRA; restart-required | [`PHASE_C_TRAINING_RUNBOOK.md`](PHASE_C_TRAINING_RUNBOOK.md) |

### Read-only telemetry

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness; sidecar readiness (`validator_version`, `sqlite_vec_available`) |
| `GET /v1/system/gpu` | nvidia-smi cache (2 s); per-GPU memory/temp/util/turbo state |
| `GET /v1/system/workers` | Live per-worker state (id `local-0`/`modal-3`/`runpod-1`, busy/idle, current job) |
| `GET /v1/system/metrics` | Validator dispatch counters + embeddings block (search totals, p50/p95) |

### Dashboard panels (`GET /dashboard`)

Static SPA at `dashboard.html`, served by taco-backend. Panels:

- **GPU telemetry** (cuda:0 + cuda:1 memory bars, temp, util, tenant)
- **Live Workers** (v1.13.0) — table polled every 2 s from `/v1/system/workers`
- **LTX Advanced Controls** — 14 tunables (sampler, sigmas, scales, etas)
- **Flux Config** — 2 tunables (turbo_steps, turbo_guidance)
- **Remote Pool** — N+1 button grids 0..MAX for Modal and RunPod
- **Turbo toggle** — single button, status-line shows ENTER/EXIT progress

### MCP tools (`noodlefinger-mcp`)

Tier-0 (read-only catalog): `get_endpoint`, `list_loras`,
`search_endpoints`, `get_flow`, `get_changelog`. Tier-1 (action,
bearer-auth'd): `submit_job`, `wait_for_job`, `upload_file`,
`download_storage_uri`, `extract_segment`, `get_beat_grid`,
`cut_music_video`, `resume_music_video`, `find_similar_shots`,
`recommend_loras`, `bulk_revalidate`. See [`MCP.md`](MCP.md).

### BFF surfaces (`noodlefinger-bff @ :8002`)

`/api/dashboard/summary`, `/api/signals/by-actor`, `/api/signals/by-bearer`,
`/api/signals/aggregate`, `/api/mcp/events` (write-side from MCP tee).
Caddy `forward_auth` injects `X-User-Email` so the BFF never sees raw
WorkOS tokens.

---

## 6. A day in the life of a clip

One a2v request traced from POST through `validator_runs` cache and
into `clip_embeddings`. Timestamps are typical-warm-state, `ltx-2-3-fast`
at 1080×1920×97 frames.

```
   t = +0.0 s   POST /v2/audio-to-video                   server.py
                  body: {prompt, audio_uri, lora?, ...}
                  → bearer auth, rate-limit gate

   t = +0.05 s  job_queue.submit()                        job_queue.py
                  Job(status=QUEUED, id=abcd-...)
                  202 Accepted returned

   t = +0.5 s   worker_loop dequeues                      job_queue.py
                _ensure_ltx_resident() (no-op if hot)
                Job.worker_id = "local-0"

   t = +0.8 s   _run_a2v() begins                         split_model_manager.py
                  encode_prompts() (~300 ms hot)
                  encode audio latent
                  phase = "denoising"

   t = +18 s    sigma loop completes (8 distilled steps)
                  evict transformer (~22 GB freed)
                  phase = "decoding"

   t = +22 s    VAE decode complete                       split_model_manager.py
                  phase = "encoding"

   t = +24 s    PyAV H.264 encode → MP4 bytes             split_model_manager.py
                  phase = "saving"
                upload_store.put() → uploads/abcd-...mp4

   t = +24.5 s  history.save() async                      history_store.py
                  generations row inserted
                  validator_score = NULL
                  thumbnail extracted async

   t = +25 s    Job.status = COMPLETED                    job_queue.py
                  GET /v2/jobs/{id} polls flip green
                _on_job_complete(job) callback fires      server.py:3724
                  _is_training_opted_in? yes
                  asyncio.create_task(_dispatch_validator)

   t = +27 s    validator.run_all_tiers begins            validator.py
                  sha256 stream (1 MiB chunks)
                  cache miss
                  tier 1 RAFT (cold-load + flow): ~3 s
                  tier 2 sapiens (stub today): ~50 ms
                  tier 3 Gemma judge: 5 keyframes → llama-swap

   t = +33 s    composite + recommendation                validator.py
                  UPDATE generations SET validator_score = 0.74,
                    validator_payload_json = {...},
                    validator_version = '1.17.0-rc5'
                    WHERE id = 'abcd-...'
                  INSERT OR IGNORE validator_runs (sha, ver, payload)

   t = +34 s    [if backfill running or rc2-onward write path]
                chat_manager.embed(prompt) → llama-swap
                  3584-d float32-LE bytes
                INSERT clip_embeddings (id='abcd-...', embedding, model_ver)

   ─── from here the row is searchable ─────────────────────────────────

   later        find_similar_shots("smoky red drummer") embeds query,
                vec0_distance scan filters api_key_hash = caller,
                this row appears in results with similarity ~0.82
                if its prompt matches semantically.
```

Citations: `server.py:_on_job_complete` (~3724), `validator.py:run_all_tiers`,
`history_store.save` (history_store.py:829),
`scripts/backfill_prompt_embeddings.py` for offline ingest into
`clip_embeddings`.

---

## 7. First 10 minutes after a restart

Walking checklist for an operator who just ran `systemctl --user restart taco-backend`.
Order matters — each step assumes the prior is green.

1. **`/health` returns 200 with all sidecars ready.**
   ```bash
   curl -s http://localhost:8090/health | jq .
   ```
   Look for: `validator_version` present, `sqlite_vec_available: true`,
   no `503` lurking. If sapiens is opted-in, watch for it in the
   sidecars dict.

2. **Live Workers panel — both local slots idle.**
   Open `http://localhost:8090/dashboard`. The Live Workers table
   should show `local-0` and (under turbo) `local-1` as `idle`. If a
   worker shows busy with no current_job, you have a zombie job —
   `cleanup_loop` will sweep it within 30 minutes (CLAUDE.md
   v1.15.2), but it's a smell.

3. **GPU panel — expected residency.**
   - Normal mode: cuda:0 either ~80 GB (LTX hot) or ~0.7 GB (cold or
     just-evicted); cuda:1 holds ACE (~18 GB) + (JoyAI ~50 GB OR
     ERNIE ~33 GB) + sapiens if opted-in.
   - Turbo mode: cuda:0 holds LTX (~79 GB), cuda:1 holds ltx-sidecar
     (~79 GB), nothing else.
   - DUAL_GPU_LTX: same as turbo but Flux/ACE/JoyAI permanently off.

4. **(v1.19.0+, planned) Quality panel — mean validator_score > 0.65.**
   Not in dashboard yet (Stream B-dash). For now, raw query:
   ```bash
   sqlite3 history.db "SELECT AVG(validator_score) FROM generations
      WHERE validator_score IS NOT NULL
        AND validator_version = (SELECT validator_version FROM generations
                                 ORDER BY created_at DESC LIMIT 1)
        AND created_at > strftime('%s','now','-7 days');"
   ```
   < 0.55 means the validator pipeline is degraded somewhere
   upstream (Gemma unhealthy, RAFT crashing, all-tiers-failed
   fallback firing). See §8.

5. **Pair counter growing (post-corpus-warmup only).**
   ```bash
   sqlite3 history.db "SELECT signal_source, COUNT(*)
      FROM preference_pairs GROUP BY signal_source;"
   ```
   In the first ~6 weeks after rc5+ validator ships, all four
   sources return 0 — that is expected (CLAUDE.md notes it
   verbatim). After that, `user_retake` and `validator_pass`
   should grow ~weekly.

6. **Validator failures — 0 in last hour.**
   ```bash
   journalctl --user -u taco-backend --since "1 hour ago" \
     | grep -E "validator.*(failed|error|ConnectError|timeout)" | wc -l
   ```
   Expected: 0. Non-zero — go to §8 row 5.

---

## 8. Failure modes cheat-sheet

Top 5 errors and where to look first. Symptom → most-likely cause →
fix path.

### 1. Turbo not engaging

**Symptom**: `POST /v1/system/turbo {"enable": true}` returns 503 or
hangs past 30 s.

**Most-likely cause**: cuda:1 didn't drain to <2 GB inside the 20 s
`_wait_cuda1_free` window. Some cuda:1 tenant didn't respond to
`systemctl stop`.

**Where to look**:
```bash
nvidia-smi
systemctl --user status ace-step joyai-sidecar ernie-image-sidecar \
  sapiens-sidecar ltx-sidecar
```

A still-resident process means it ignored stop or wasn't gated by the
`LOAD_*` flag. `_restore_cuda1_tenants()` has already rolled back, so
the system is in normal mode — try again after manually killing the
stuck process. Code path: `server.py:_enter_turbo_mode` (~1712).

### 2. Embeddings 502 / 503

**Symptom**: `POST /v2/embeddings/search` returns 503 with
`"embedding service unavailable"` or `"embedding search not available
— install sqlite-vec extension"`.

**Most-likely cause**: llama-swap upstream isn't serving
`/v1/embeddings`, or the sqlite-vec `.so` failed to load on backend
boot.

**Where to look**:
- `curl -s http://192.168.1.80:8080/v1/models | jq '.data[].id'` —
  must include the embedding-mode entry (e.g. `gemma-3-12b-embed`).
  If your llama-swap config only has chat-mode entries, vLLM is
  refusing the `/embeddings` route.
- Backend logs: `journalctl --user -u taco-backend | grep -i sqlite_vec`
  — the import error string surfaces here (missing libstdc++,
  wrong glibc, etc.). `SQLITE_VEC_LOAD_ERROR` propagates via
  `/health`.

Code paths: `chat_manager.embed` (proxies llama-swap),
`history_store.SQLITE_VEC_AVAILABLE` (module-level flag).

### 3. Sapiens 503

**Symptom**: validator tier-2 logs `tier2=None` warnings every job.

**Most-likely cause**: `LOAD_SAPIENS=0` (rc2 default, by design — the
real model isn't shipped yet) OR the sidecar service isn't running
even though `LOAD_SAPIENS=1`.

**Where to look**:
```bash
grep LOAD_SAPIENS .env
systemctl --user status sapiens-sidecar
```

The validator tolerates tier-2 failure gracefully — composite drops
the tier2 weight and returns. This is **expected behavior in stub
mode**; the warnings are noise, not a bug. Real fix: wait for
Stream C (Sapiens-2 real implementation, task #24).

### 4. Analyze-motion timing out

**Symptom**: `POST /v2/video/analyze-motion` first call takes ~5 s,
subsequent ~150 ms. Or: every call takes ~5 s.

**Most-likely cause** (warm path): RAFT weights cold-loading — they
download lazily from pytorch.org on first call (~22 MB), cached to
`~/.cache/torch/hub/`. Hot path is ~150 ms.

**Where to look**:
- If only the first call after restart is slow: expected.
- If every call is slow: the cuda:0 LTX transformer is contending
  with RAFT for VRAM. RAFT is supposed to lazy-load and evict
  per-call (`validator.py:_run_tier1_raft`). Check `nvidia-smi`
  during a stuck call — if cuda:0 is at >90 GB throughout, the
  evict half of the lazy-load is leaking.

### 5. Validator scores all 0.5 (or all "warn")

**Symptom**: `SELECT validator_payload_json FROM generations LIMIT 5`
shows tier3 always returning the fallback `{verdict: "warn", score:
0.5}`.

**Most-likely cause**: tier-3 fallback path firing — Gemma is
unhealthy, schema validation rejecting every response, or
llama-swap is wedged.

**Where to look**:
```bash
journalctl --user -u taco-backend | grep -E "tier3|JudgeResponseV1"
journalctl -u llama-swap --since "1 hour ago" | tail -100
curl -s http://192.168.1.80:8080/v1/models | jq '.data[].id'
```

The vision model (`gemma-4-31b-it`) must be in the swap config; if
only the chat model (`gemma-3-12b-nvfp4`) is loaded, every multimodal
request lands on the wrong model and the schema-validator drops the
response. Code path: `validator.py:_run_tier3_judge` (fallback returns
`{verdict: "warn", score: 0.5}` on any failure).

---

For per-symptom triage (HTTP 4xx/5xx codes, NULL fields,
queue-saturation FAQ), see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
For the strategic frame and forward roadmap, see
[`/home/ian/.claude/plans/melodic-sniffing-beacon.md`](/home/ian/.claude/plans/melodic-sniffing-beacon.md).
