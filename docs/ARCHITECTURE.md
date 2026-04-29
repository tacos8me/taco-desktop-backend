# Architecture Overview

**Audience**: an engineer onboarding the project who wants the topology in 10 minutes, not the code.
**Status as of 2026-04-29**: taco-backend `v1.18.0-rc3` shipped. All four loops (capture, validate, retrieve, train) are now live. Phase D (auto-promote LoRA on green AB) is the only remaining deferred work — see [§11](#11-whats-deferred).

For implementation detail, see `CLAUDE.md` (gateway), `docs/CAPTURE_VALIDATOR.md` (validator), `docs/gpu-architecture.md` (GPU swap), `docs/API.md` (HTTP contract), `docs/MCP.md` (LLM tools).

---

## 1. The capture+validator+retrieval+training flywheel

taco-backend is an LTX-compatible inference server. But the strategic frame is bigger: **every clip the operator ships is a labeled positive; every retake is a labeled (chosen, rejected) preference pair**. The system is designed so that day-to-day music-video production passively produces training data for the next LoRA we fine-tune. The 32-clip MVs the operator is generating right now ARE the dataset.

The four loops:

```
                  ┌─────────────────────────────────────────────────┐
                  │                                                 │
                  ▼                                                 │
        ┌─────────────────┐         ┌──────────────────┐            │
        │   1. CAPTURE    │ ──────▶ │   2. VALIDATE    │            │
        │  user generates │         │  RAFT + Sapiens  │            │
        │  a clip via API │         │  + Gemma judge   │            │
        └─────────────────┘         └──────────────────┘            │
                                            │                       │
                                            ▼                       │
                                    score + verdict                 │
                                    saved to history.db             │
                                            │                       │
                  ┌─────────────────────────┤                       │
                  │                         │                       │
                  ▼                         ▼                       │
        ┌─────────────────┐         ┌──────────────────┐            │
        │  3. RETRIEVE    │         │   4. TRAIN       │            │
        │  similar shots  │         │  DPO LoRA from   │            │
        │  via embeddings │         │  preference_pairs│            │
        │  (Phase B)      │         │  (Phase C)       │            │
        └─────────────────┘         └──────────────────┘            │
                  │                         │                       │
                  │                         ▼                       │
                  │                 deployed LoRA ─────────────────▶│
                  │                                          better outputs
                  ▼
        operator authors better
        shot lists, faster
```

**Today (v1.18.0-rc3)**: all four loops are wired. Every completed video job from an opted-in bearer flows through the validator and gets a composite score saved to `generations.validator_score` (loop 2). Phase B retrieval shipped in v1.18.0-rc2 (loop 3) — `/v2/embeddings/search` and `/v2/embeddings/recommend-loras` are live. Phase C training infrastructure shipped in v1.18.0-rc3 (loop 4) — `scripts/construct_preference_pairs.py`, `scripts/train_dpo_sft.py`, `scripts/ab_decision.py`, and `POST /v1/system/lora/rollback` are operator-driven; first training run waits until corpus crosses ~1000 pairs.

**Why this matters**: a single human operator running 4 MVs/week of 32 clips each generates ~5000 labeled clips/year. With validator scores + retake provenance + composition-survival signals, that's ~1000 high-strength preference pairs/year — enough to fine-tune a quality-improving DPO LoRA per quarter. The validator pipeline is the labeler.

---

## 2. Component map

```
                                ┌──────────────────────────────┐
                                │       External Clients       │
                                │  (noodle-i, noodle-v,        │
                                │   noodlefinger-mcp, dashboard)│
                                └──────────────┬───────────────┘
                                               │  HTTPS / Bearer
                                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                         taco-backend (FastAPI :8090)               │
│                                                                    │
│   server.py — routes, auth, dispatch, turbo coordination           │
│       │                                                            │
│       ├──▶ job_queue.py — async queue, per-key rate limits         │
│       │       │                                                    │
│       │       ▼                                                    │
│       │   worker_loop ── on_complete ──▶ _on_job_complete          │
│       │       │                              │                     │
│       │       │                              └──▶ validator        │
│       │       ▼                                   (fire-and-forget)│
│       │   dispatcher                                               │
│       │       ├─ local cuda:0 (LTX/Flux swap)                      │
│       │       ├─ local cuda:1 sidecars (HTTP)                      │
│       │       ├─ Modal pool (HTTP, base64 media)                   │
│       │       └─ RunPod pool (HTTP, base64 media)                  │
│       │                                                            │
│       ├──▶ split_model_manager.py — LTX video                      │
│       ├──▶ flux_manager.py        — Flux 2 image                   │
│       ├──▶ chat_manager.py        — proxy to llama-swap            │
│       ├──▶ validator.py           — 3-tier scoring                 │
│       ├──▶ history_store.py       — SQLite WAL                     │
│       ├──▶ upload_store.py        — UUID file storage              │
│       ├──▶ composition_store.py   — comp metadata                  │
│       └──▶ export_handler.py      — ffmpeg concat + transcode      │
│                                                                    │
└────────────────────────┬─────────────────────────┬─────────────────┘
                         │                         │
                         ▼                         ▼
              ┌────────────────────┐    ┌──────────────────────┐
              │  history.db (WAL)  │    │  validator_artifacts │
              │  schema v3         │    │  /uploads/           │
              │  + thumbnails/     │    │  (UUID-keyed)        │
              └────────────────────┘    └──────────────────────┘

         ── future (Phase B+) ──────────────────────────────────────
              clip_embeddings (sqlite-vec virtual table) ──▶ retrieval
              preference_pairs ──▶ DPO LoRA training ──▶ lora_registry
         ──────────────────────────────────────────────────────────
```

**Sidecars** (separate processes, HTTP-proxied):

```
  cuda:0       cuda:1                 CPU            llama-swap
  ──────       ──────                 ───            ──────────
  LTX  ◀──┐    ace-step       :8001   madmom :8095   chat / vision
  Flux ◀──┘    joyai-sidecar  :8092   ──────────     (Gemma 3/4 12B)
               ernie-sidecar  :8094                  used by:
               ltx-sidecar    :8093                    /v1/chat/completions
               sapiens-sidecar:8096                    /v2/char/rank
                                                       validator tier-3
```

Sidecars are systemd user units, gated by `LOAD_*` env vars. Failures surface as explicit `503` (no silent fallback) except where opt-in (validator tier-2 stub mode is graceful).

---

## 3. GPU topology

Two RTX PRO 6000 Blackwell GPUs (96 GB each). No third GPU.

```
        cuda:0 (96 GB)                       cuda:1 (96 GB)
   ┌────────────────────────┐           ┌────────────────────────┐
   │   LTX  ◀── swap ──▶  Flux │           │  ACE (always, ~18 GB)  │
   │                        │           │  ────────────────────  │
   │   LTX active: ~79 GB   │           │  JoyAI (~50 GB)        │
   │   Flux Dev:   ~81 GB   │           │     OR                 │
   │   Flux Klein: ~32 GB   │           │  ERNIE (~33 GB)        │
   │                        │           │     (mutually exclusive│
   │   mutually exclusive   │           │      with each other)  │
   │   (combined > 96 GB)   │           │                        │
   │                        │           │  + sapiens (~?, opt-in)│
   │                        │           │  + ltx-sidecar (turbo) │
   └────────────────────────┘           └────────────────────────┘
```

**Three operating modes** govern how those two GPUs are allocated:

```
   normal              turbo              dual-gpu-ltx
   ──────              ─────              ────────────
   cuda:0: LTX↔Flux    cuda:0: LTX        cuda:0: LTX
   cuda:1: ACE+JoyAI   cuda:1: LTX(side)  cuda:1: LTX(side)
                       (image+music       (no image/music
                        return 503)        ever; boot flag)

   1 video at a time   2 local + up to    2 local always
   image/music OK      12 remote = 14
                       concurrent video
```

**Normal**: cuda:0 auto-swaps between LTX (~79 GB) and Flux (~81 GB Dev / ~32 GB Klein) on every request. The swap is enforced by `_ensure_ltx_resident()` / `_ensure_flux_ready()` helpers wired into every dispatcher (server.py). LTX→Flux costs ~3 s eviction; Flux→LTX costs ~7-30 s cold load. Single-tenant workloads pay zero swap overhead.

**Turbo** (`POST /v1/system/turbo`): cuda:1 tenants are `systemctl stop`'d, cuda:1 drains to <2 GB (verified via `nvidia-smi`), `ltx-sidecar` starts on cuda:1, a second worker_loop spawns. Optionally scales remote sidecars on Modal (max 10) and RunPod (max 2). **2 local + 12 remote = 14 concurrent video workers**. Image/music endpoints return 503 while turbo is active. Entry ~20 s, exit ~15 s.

**DUAL_GPU_LTX** (`DUAL_GPU_LTX=1` boot flag): both GPUs dedicated to LTX permanently. Flux/ACE/JoyAI disabled. No runtime toggle, no swap latency. Use when video is the only workload.

**Why mutually exclusive on cuda:0**: LTX active (~79 GB) + Flux active (~81 GB) = ~160 GB > 96 GB physical. Even though Flux Dev uses CPU offload (~zero idle footprint), forward-pass peak claims the GPU. They cannot coexist.

---

## 4. Data layer

`history.db` is a single SQLite file in WAL mode. Single writer, many readers, no external DB. The schema is currently at **v3** (rc1 ship); v4 is the next planned migration (Phase B keystone).

### Tables and who writes to them

| Table | Purpose | Primary writer |
|---|---|---|
| `generations` | One row per completed job. Holds prompt, model, dims, result_uri, thumbnail, validator_score, shot lineage. | `worker_loop` after each job + `_dispatch_validator` for validator fields |
| `composition_clips` | Inverted-index: which clips ended up in which composition. NULL `clip_history_id` for synthetic flash-inserts. | `POST /v2/compositions/{id}/export` (best-effort) |
| `validator_runs` | Cache of validator results keyed by `(video_sha256, validator_version)`. UNIQUE index ensures idempotent writes across concurrent calls. | `validator.run_all_tiers` |
| `preference_pairs` | Forward-looking: (chosen, rejected) tuples for DPO training. Sources: user_retake, composition_kept, validator_pass, validator_fail. | **No writer yet — Phase C** |
| `training_runs` | Forward-looking: ledger of fine-tuning runs (LoRA path, base model SHA, eval metrics, deploy/deprecate timestamps). | **No writer yet — Phase C** |
| `api_key_metadata` | Per-bearer training opt-in flag + tier. Spine for the privacy gate. | One-time seed from `.api_keys` on v2→v3 migration; manual INSERTs thereafter |

The fields that distinguish v3 from v2: 11 new nullable columns on `generations` (validator scoring + shot lineage + retake provenance + LoRA-applied placeholders + `prompt_embedding` BLOB), 3 indexes, 5 new tables. Migration is single-startup, idempotent, additive — pre-v3 rows get NULL in new columns; rollback is safe.

`api_key_hash` is sha256 of the bearer; raw keys are never stored. The `api_key_metadata.training_opt_in` column is the **privacy gate** — every passive validator dispatch checks it before scoring.

### Off-DB storage

- **`uploads/`** — UUID-keyed user-supplied images, audio, source videos, and job results. `upload_store.py` mediates. `storage://<uuid>` URIs in API responses resolve here.
- **`thumbnails/`** — 256-wide JPEGs auto-extracted from result MP4s via PyAV.
- **`validator_artifacts/`** — directory configured (`VALIDATOR_ARTIFACTS_DIR`) but no writer yet (Phase B).
- **`loras/`** — LTX LoRA files + `registry.json`. **`flux_loras/`** — Flux LoRA folder-drop discovery (filesystem is source of truth, no registry).

---

## 5. Sidecars

Sidecars run as separate systemd user units. taco-backend proxies to them via httpx. Each gated by a `LOAD_*` env var; failures surface as explicit 503 (no silent fallback) except where stub-tolerance is the design (sapiens).

| Sidecar | Port | GPU | Role | Default |
|---|---|---|---|---|
| `ace-step` | :8001 | cuda:1 | ACE music gen (xl-base + LM, ~18 GB) | `LOAD_ACE=1` |
| `joyai-sidecar` | :8092 | cuda:1 | JoyAI image edit (~50 GB) | `LOAD_JOYAI` |
| `ernie-image-sidecar` | :8094 | cuda:1 | ERNIE-Image t2i (~33 GB) | `LOAD_ERNIE` |
| `ltx-sidecar` | :8093 | cuda:1 | LTX video (turbo or dual-GPU mode) | turbo-only |
| `madmom-sidecar` | :8095 | CPU | Higher-accuracy beat/downbeat detection | `LOAD_MADMOM=1` |
| `sapiens-sidecar` | :8096 | cuda:1 | Validator tier-2 pose stability (currently stub) | `LOAD_SAPIENS=0` |
| Modal LTX pool | remote | remote | Up to 10 concurrent remote video workers | turbo-only, opt-in |
| RunPod LTX pool | remote | remote | Up to 2 concurrent remote video workers | turbo-only, opt-in |

**llama-swap** (external service at `192.168.1.80:8080`) is the LLM gateway. Proxied by `chat_manager.py`. Used by `/v1/chat/completions`, `/v2/char/rank`, and the validator's tier-3 Gemma judge. Different model IDs are exposed as named slots (`gemma-3-12b-nvfp4` for chat, `gemma-4-31b-it` for vision/judge ranking).

JoyAI ↔ ERNIE swap on cuda:1 (combined ~83 GB with ACE would exceed 96 GB; only one loads). All cuda:1 tenants are `systemctl stop`'d on turbo entry and restarted on exit.

---

## 6. The 3-tier validator

The validator is the labeler. It runs on every completed video job from an opted-in bearer (passive) and on demand via `POST /v2/video/analyze-motion` (active, used by the MCP `quality_validation` hook).

```
                ┌─────────────────────────────┐
                │   validator.run_all_tiers   │
                │  cache check → tiers → comp │
                └──────────┬──────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  TIER 1    │  │  TIER 2    │  │  TIER 3    │
     │  RAFT      │  │  Sapiens   │  │  Gemma     │
     │  optical   │  │  pose      │  │  vision    │
     │  flow      │  │  stability │  │  judge     │
     │  (in-proc, │  │  (sidecar  │  │  (llama-   │
     │   cuda:0)  │  │   :8096)   │  │   swap)    │
     │  ~150ms    │  │  stub-now  │  │  ~3 s      │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
           ▼               ▼               ▼
       motion          pose            verdict + score
       magnitude       variance        + retake hint
       per window
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                   composite() →
                   0.4·t1 + 0.2·t2 + 0.4·t3
                           │
                           ▼
                   recommendation:
                   pass / warn / retake
```

**Tier 1 (RAFT)**: torchvision raft_small, lazy-loaded on cuda:0, evicted after each call so LTX reclaims the GPU cleanly. Outputs motion magnitude per pair, time-windowed flow buckets, and motion smoothness. Catches the "static-in-first-half then motion" defect class (the v0.4.6 MCP defect) automatically.

**Tier 2 (Sapiens)**: pose-temporal stability via the sapiens sidecar. **Currently stub** (`LOAD_SAPIENS=0` default); the sidecar returns `{"stub": true}` and composite treats that as `0.2·1.0` — neutral, no penalty. Real Sapiens-2 inference lands in a future rc.

**Tier 3 (Gemma judge)**: 5 keyframes (first/quartile/middle/3-quartile/last) sent multimodal to llama-swap with strict-JSON schema (`JudgeResponseV1`). System prompt is `JUDGE_PROMPT_V1`, tuned for the same defect classes RAFT catches plus identity drift and prompt-output mismatch. Schema-validated; on failure returns `{verdict: "warn", score: 0.5}` so composite still computes.

**Composite**: `0.4·tier1_norm + 0.2·tier2 + 0.4·tier3`, with a verdict-level override (tier3 says "retake" → recommendation is "retake" regardless of numeric composite). `pass` ≥ 0.65, `warn` 0.45-0.65, `retake` < 0.45.

**Caching** via `validator_runs(video_sha256, validator_version)` UNIQUE index. SHA-256 streamed from disk in 1 MiB chunks. Bumping `VALIDATOR_VERSION` forces re-runs.

**Two dispatch paths cooperate**: passive (`_on_job_complete` → fire-and-forget) populates the cache after each job; active (`POST /v2/video/analyze-motion` from MCP) hits the cache instantly ~50 ms later from the orchestrator's quality-validation hook.

---

## 7. The MCP layer

`noodlefinger-mcp` (separate repo) is the LLM-facing tool layer. Two tiers:

- **Tier 0 — docs/lookup tools** (`get_endpoint`, `list_loras`, `get_changelog`, `search_endpoints`, `get_flow`, etc.): zero-side-effect, read-only catalog access. Lets a coding assistant find the right endpoint and shape without RAGing the whole repo.
- **Tier 1 — action tools** (`submit_job`, `wait_for_job`, `upload_file`, `download_storage_uri`, `extract_segment`, `get_beat_grid`, `cut_music_video`, `resume_music_video`, ...): wraps actual taco-backend HTTP calls with bearer auth.

The headline tool is **`cut_music_video`**: an orchestrator that takes a music prompt + shot list + audio, plans the shot DAG, dispatches clips in parallel (auto-scaled to available worker count), handles tail-extraction for chain conditioning, optionally validates each clip via `quality_validation` (the active validator hook), and returns a fully-assembled MV ready for `export_composition`.

**`quality_validation`** in `cut_music_video` is the active validator hook. When `enabled=true`, after each clip is generated the orchestrator calls `/v2/video/analyze-motion`, caches the score in session state, and (if `on_failure="retake"`) re-submits the clip with budget tracking. Final result includes a `quality_telemetry` block with per-clip scores + retake counts + total validation overhead.

The MCP also tees every state transition (`clip_submitted`, `clip_completed`, `validator_result`, `composition_exported`, ...) to the BFF (next section), giving us a per-user signal stream for product analytics and training data without coupling MCP latency to BFF availability.

---

## 8. The BFF layer

`noodlefinger-bff` is a FastAPI service that captures **user signals** distinct from the operator's `audit` log. Two writers:

1. **MCP→BFF tee** (`POST /api/mcp/events`): the MCP fires every session event (clip lifecycle, validator results, retakes, exports) at the BFF as fire-and-forget HTTP. Anonymous-tolerant (falls back to `actor_email="unknown"`); failures swallowed so MCP correctness never depends on BFF reachability.
2. **Frontend events** (gallery downloads, explicit ratings, share clicks): direct from noodle-i/noodle-v/noodle-portal.

All land in a single `user_signals` table with `(ts, actor_email, signal_type, target_id, metadata_json)`. Distinct from operator `audit` entries — `user_signals` are training-data inputs and product-engagement metrics.

**Read-side query endpoints** (Phase B+): `GET /api/signals/by-actor`, `by-bearer`, `aggregate`. Used by the dashboard to show per-user signal distributions.

`api_key_hash` scoping is added in Phase B (BFF v0.3.0) for multi-tenant readiness — today the system is single-tenant, but the spine is being put in place.

---

## 9. Tenancy model

Today: **single-operator deployment**. One bearer in `.api_keys` does all the work; no isolation between requests is enforced beyond the per-key rate limits. Validator opt-in is ON by default for the seeded bearer.

Multi-tenant ready, not multi-tenant operational:

- Every `generations` row has `api_key_hash` (sha256 of bearer). Raw keys are never stored.
- Every validator dispatch checks `api_key_metadata.training_opt_in` before scoring. Default opt-out for unknown bearers added post-seed.
- Phase B retrieval endpoints (`/v2/embeddings/search`) MUST filter results by `api_key_hash` (privacy gate, test-enforced).
- BFF `user_signals` gains `api_key_hash` in v0.3.0 for the same reason.

When the first external bearer arrives, the path is: add to `.api_keys`, INSERT explicit `api_key_metadata` row with `training_opt_in=0` (opt-out by default), and rely on the privacy gate everywhere downstream. No code change.

---

## 10. The data flywheel

Putting it all together — how a single MV production session generates training data:

```
   Operator runs cut_music_video on a 32-clip shot list
             │
             ▼
   orchestrator dispatches clips in parallel (DAG-aware)
             │
             ▼
   each clip: LTX video gen → result MP4 + thumbnail
             │
             ▼  (worker_loop on_complete callback)
   passive validator dispatch (fire-and-forget)
       ├─ tier 1 RAFT          ──┐
       ├─ tier 2 Sapiens (stub) ─┤── composite + recommendation
       └─ tier 3 Gemma judge   ──┘
             │
             ▼
   generations.validator_score populated
             │
             ▼
   active validator hook (MCP synchronous)  ──▶  cache hit, ~50 ms
       │
       ▼
   if on_failure="retake" + recommendation="retake":
       ├─ budget check (max_retakes_per_clip)
       └─ re-submit clip with optional retake_hint applied to prompt
             │
             ▼
   operator hand-curates final composition (drop bad clips)
             │
             ▼
   POST /v2/compositions/{id}/export
       └─ writes composition_clips rows (chosen-set anchor)
             │
             ▼
   MCP tee fires events at BFF
       └─ user_signals capture: gallery downloads, exports, ratings

   ── Phase C (deferred) ────────────────────────────────────────
   weekly cron: construct_preference_pairs.py
       ├─ user_retake     (chosen=child, rejected=parent_clip_id)
       ├─ composition_kept (chosen=in comp, rejected=not in comp w/ same shot_config_key)
       ├─ validator_pass  (chosen=passed, rejected=warned w/ same shot_config_key)
       └─ validator_fail  (synthetic negatives)

   training cron: train_dpo_sft.py
       ├─ filter signal_strength ≥ 0.5
       ├─ scope by validator_version
       ├─ SFT-on-chosen LoRA training (~50-60 GPU-hours)
       └─ writes training_runs row, registers LoRA as candidate

   A/B framework
       ├─ MV-level routing (50/50 candidate vs baseline)
       ├─ paired t-test on per-arm validator scores
       └─ auto-promote on p<0.05 + ≥10% delta
   ──────────────────────────────────────────────────────────────
             │
             ▼
   deployed LoRA → operator authors next MV with better outputs
       (back to top of flywheel)
```

The **shot_config_key** is the join key that makes pair construction work. Two clips with the same prompt, image, audio window, model, LoRA id, and LoRA strength share a `shot_config_key` and are pair-eligible regardless of session. The `parent_clip_id` (set by `/v2/retake`) directly captures retake provenance for the strongest signal source (`signal_strength=0.9`).

---

## 11. What's live (and what's still deferred)

The full flywheel is now wired end-to-end. Phase A schema landed in v1.17.0-rc1, validator pipeline in rc2, retrieval in v1.18.0-rc2, training infrastructure in v1.18.0-rc3.

**Phase B — Retrieval (taco-backend v1.18.0-rc2)** — LIVE:
- `chat_manager.embed` / `embed_batch` proxy llama-swap `/v1/embeddings` (3584-dim Gemma → float32-LE bytes)
- `clip_embeddings` virtual table via sqlite-vec extension (the deprecated `prompt_embedding` BLOB column on `generations` is no longer written — see ADR-003)
- `POST /v2/embeddings/search` (privacy-gated semantic search; ranking `0.50·sim + 0.35·v_norm + 0.10·recency + 0.05·comp_kept`)
- `POST /v2/embeddings/recommend-loras` (similarity-then-group LoRA aggregation, `0.7·mean + 0.3·boost`)
- `POST /v2/system/bulk-revalidate` (admin-gated re-validator, defaults to dry-run)
- Per-key token-bucket rate limit (10 req/sec/key, burst 10) on `/v2/embeddings/*` and `/v2/system/bulk-revalidate`
- `lora_applied_id` write fix end-to-end (every video v2 endpoint now persists `body.lora.id`)
- `/v1/system/metrics` extended with `embeddings` block (search totals, p50/p95 latency, recommend/bulk-revalidate counters)
- `scripts/backfill_prompt_embeddings.py` (idempotent, resumable backfill)

**Phase C — SFT LoRA Training (taco-backend v1.18.0-rc3)** — INFRASTRUCTURE LIVE, FIRST RUN PENDING CORPUS:
- `scripts/construct_preference_pairs.py` weekly cron (4 sources — `user_retake` 0.9, `composition_kept` 0.5, `validator_pass` 0.7, `validator_fail` 0.3; version-scoped; idempotent via `idx_pp_unique_pair_source` UNIQUE)
- `scripts/train_dpo_sft.py` (LoRA-only via PEFT, paged_adamw_32bit, gradient_checkpointing, bf16; defaults to dry-run, `--execute` required for GPU consumption; ~50-60 GPU-hours per cycle)
- `scripts/ab_decision.py` weekly cron (paired t-test on per-MV mean validator_score; promote ≥+10% AND p<0.05; deprecate ≤-5% AND p<0.05; <30 MVs/arm = insufficient_samples; see ADR-016)
- `configs/sft_quality_lora.yaml` (rank=64, alpha=64, q/k/v/out_proj target modules, 3 epochs, lr=5e-4)
- `POST /v1/system/lora/rollback` admin endpoint (atomic `.env` rewrite, restart-required, audit shape with `note` field)
- Privacy-gate spine: every source query filters by `api_key_metadata.training_opt_in = 1`
- Validator-version scoping spine: cross-version pairs cannot enter training

**Phase C.1 — Diffusion-DPO**: deferred until SFT v1 proves out via A/B. SFT-on-chosen is the v1 baseline (user-locked).

**Phase D — Per-genre LoRAs, active learning loop, frame-level CLIP embeddings, auto-promote on green AB**: still deferred. Genre tagging would require a `genre` column on `generations` (not yet added). Active learning surfaces borderline-composite clips (0.45-0.55) to the user for explicit thumbs-up/down. Auto-promote on green AB (currently operator-gated) is the natural follow-on once a few cycles of A/B data accumulate.

**Single-operator volume note**: at 4 MVs/week × 32 clips × ~5% retake rate, the corpus crosses the 1000 high-strength preference pair threshold ~6-8 weeks after rc5+ validator ships against the new corpus. Phase C scripts wait until the corpus is large enough; first invocation is operator-driven, not auto.

---

For the strategic frame and forward roadmap with timing and dependency graph, see `/home/ian/.claude/plans/melodic-sniffing-beacon.md`.
