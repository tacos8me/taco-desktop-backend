# Capture + Validator Machine — Roadmap

**Last updated**: 2026-04-29
**Owner**: ian@whoufm.com (single-tenant deploy)
**Architecture doc**: `docs/CAPTURE_VALIDATOR.md` (sibling)
**Origin plan**: `/home/ian/.claude/plans/melodic-sniffing-beacon.md`

This document lists concrete next steps for the capture+validator machine.
Each item carries a **Trigger** (the precondition that flips it from "later" to
"now"), an **Effort** estimate, **Blocker**(s) it depends on, **Priority**
(P0/P1/P2), and an **Owner-hint** (who or how many agents). Do not treat any
section as ordered work-list — read the trigger before picking up.

## Status as of 2026-04-29

| Layer | Status | Tag |
|---|---|---|
| Schema v3 + lineage + retake | LIVE | v1.17.0-rc1 |
| Validator pipeline (RAFT + Sapiens stub + Gemma judge) | LIVE | v1.17.0-rc2 |
| Polish (docstring + turbo-stop gating) | MERGED, NOT RESTARTED | v1.17.0-rc3 |
| Sapiens-sidecar (stub) | active | n/a |
| MCP v0.7.0 quality_validation hook | LIVE | v0.7.0 |
| BFF v0.2.0 user_signals + event tee | LIVE | v0.2.0 |

**Capture phase**: ON (validator dispatches fire-and-forget on every completed
video job for opted-in keys). **Active validation hook**: opt-in, off by
default in `cut_music_video.quality_validation.enabled`. **Phase B
(retrieval)**: not started. **Phase C (training)**: not started.

The capture loop logs every clip + every retake + every composition export
and accumulates the labeled dataset that downstream phases consume. The most
load-bearing operational reality right now is "is rc3 actually running?" —
the polish fixes are dormant under `LOAD_SAPIENS=0` and the rc2 binary still
serves traffic. Validator dispatch IS live; sapiens-stub IS contributing the
documented 0.2·1.0 to every composite.

## Roadmap structure

Each item has:

- **Trigger**: the empirical signal that says "ready"
- **Effort**: hours / days / weeks
- **Blocker**: anything upstream
- **Priority**: P0 (must-do), P1 (should-do), P2 (nice-to-have)
- **Owner-hint**: one engineer, multiple agents, etc.

---

## Section 1 — IMMEDIATE (next 7 days)

### 1.1 Restart taco-backend to pick up rc3

- **Trigger**: convenient maintenance window (queue empty, no MV in progress)
- **Effort**: 1-2 min (restart + ~45 s model reload)
- **Blocker**: none
- **Priority**: P1 (rc3 fixes are dormant under `LOAD_SAPIENS=0`; not urgent)
- **Owner-hint**: operator
- **What changes**: `_is_training_opted_in` docstring corrected;
  `_stop_cuda1_tenants` gated on `LOAD_SAPIENS` so turbo entry no longer
  emits a spurious `systemctl stop sapiens-sidecar` when the operator hasn't
  enabled the sidecar.
- **Verification**: `journalctl --user -u taco-backend -n 50 | grep -i 'rc3\|started'`
  should show v1.17.0-rc3 in the boot banner. `curl localhost:8090/health`
  returns 200 within 60 s.

### 1.2 Verify validator dispatch is firing on real jobs

- **Trigger**: 24-48 h post-restart (rc3 or otherwise — rc2 also dispatches)
- **Effort**: 30 min query session
- **Blocker**: none
- **Priority**: P0 — if validator isn't dispatching, the whole capture
  machine is broken and Phase B + C inherit empty corpora
- **Owner-hint**: operator
- **Action**:
  ```sql
  SELECT
    COUNT(*) AS total_videos,
    COUNT(validator_score) AS scored,
    AVG(validator_score) AS mean_score,
    MIN(created_at), MAX(created_at)
  FROM generations
  WHERE type IN ('text-to-video', 'image-to-video', 'audio-to-video',
                 'retake', 'video-outpaint', 'video-hdr')
    AND created_at > strftime('%s', 'now', '-2 days') * 1.0;
  ```
  Expect `scored / total_videos > 0.9` for opted-in keys (allow ~10 % for
  in-flight + dispatch-failures).
- **Followup if dispatch ratio is low**: grep `journalctl --user -u taco-backend
  -n 5000 | grep -i 'validator'` — failure modes likely fall into (a)
  result_uri resolution failures (best-effort path missing), (b) RAFT model
  download failures (first-call), (c) Gemma chat 503. None propagate; all
  log WARN.

### 1.3 Calibrate validator score distribution

- **Trigger**: ~1 week of capture (≥100 `validator_runs` rows)
- **Effort**: half-day analysis
- **Blocker**: 1.2 (must verify dispatch first)
- **Priority**: P0 — wrong threshold = retake budget burned on borderline
  clips OR borderline-bad clips ship without retake
- **Owner-hint**: operator + one analysis agent
- **Action**: histogram of `composite_score` and per-tier scores from
  `validator_runs.payload_json`. Identify modal distribution. Recalibrate
  the retake threshold (currently 0.45) if the real-data center is offset.
  Look specifically for:
  - Composite floor (< 0.3 means tier-1 dynamic_degree calibration is wrong
    — most likely fix: bump the `min(dyn / 5.0, 1.0)` divisor)
  - Composite ceiling (everyone clusters at 0.95 — recommendation never
    fires `retake`; data is uncalibrated)
  - Tier-3 verdict-vs-numeric agreement rate
- **Output**: a one-time PR adjusting `validator.composite()` thresholds OR
  bumping `VALIDATOR_VERSION` to force re-run (cache invalidated by version
  bump per the rc1 schema's UNIQUE index).

### 1.4 LOAD_SAPIENS=1 readiness gate

- **Trigger**: rc3 restart done **AND** 1 week of stable capture (no
  validator errors in journal) **AND** sapiens real inference is wired
  (rc-final ship, see 1.6)
- **Effort**: 1 hour to flip + smoke test (env edit + restart)
- **Blocker**: 1.6 (rc-final), 1.1 (restart), 1.2 (dispatch verified)
- **Priority**: P1
- **Owner-hint**: operator
- **Risk**: cuda:1 contention with ACE; mitigated by `_stop_cuda1_tenants`
  coordination already in place (rc3 fix verified).
- **Verification**: after flip, run a single clip end-to-end and confirm
  `validator_runs.payload_json -> '$.tier2.stub'` is `false`.

### 1.5 MCP refresh in operator's Codex session to pick up v0.7.0

- **Trigger**: operator wants synchronous validation + retake decisions on
  next MV
- **Effort**: 5 min (`uvx --refresh`, restart Codex)
- **Blocker**: none
- **Priority**: P0 if operator wants to test the active retake loop
  (otherwise P2 — capture still happens passively via on-complete dispatch)
- **Owner-hint**: operator
- **Action**: ensure `quality_validation: {enabled: true,
  max_retakes_per_clip: 1, on_failure: "retake"}` on a 4-clip test MV; watch
  logs for retake fires; check `quality_telemetry` block in result payload.

### 1.6 Sapiens-sidecar real inference (rc-final)

- **Trigger**: ready when **either** (a) capture data shows tier-1 + tier-3
  alone are insufficient signal (composite-vs-user disagreement >25 %) OR
  (b) human-clip identity drift is observed in MVs and tier-3 isn't
  catching it
- **Effort**: 1-2 days (model load in lifespan hook, real PyAV decode +
  Sapiens forward pass, populate `keypoints`/`confidence` arrays)
- **Blocker**: ~4 GB weight download from HF (gated to 15 min in `setup.sh`;
  188 GB free under `/mnt/nvme-1/huggingface` confirmed). Also, the SETUP.md
  notes a known limitation in the systemd unit: `Type=exec` doesn't gate on
  `/health`, so a weight-load failure won't trigger `Restart=on-failure`.
  Should be flipped to `Type=notify` + `sd_notify('READY=1')` from the
  FastAPI lifespan hook before flipping LOAD_SAPIENS=1 in production.
- **Priority**: P2 (nice — but tier-2 is only ~20 % of composite weight; if
  RAFT + Gemma alone are working, defer)
- **Owner-hint**: one engineer (1-2 days focused)
- **Where**: `/mnt/nvme-1/servers/sapiens-sidecar/service.py` — the stub
  `/v1/analyze-pose` route. Schema is forward-stable; client-side change is
  one removed `if payload.get("stub")` guard in `validator._run_tier2_sapiens`.

---

## Section 2 — THIS MONTH (next 30 days)

### 2.1 Validator score dashboard panel

- **Trigger**: ~500 `validator_runs` rows accumulated (~5 days at current
  cadence given a 32-clip MV/day pace)
- **Effort**: 2-3 days (`taco-backend/dashboard.html` — add panel querying
  `validator_runs` joined to `generations` via `validator_artifact_uri` /
  `video_sha256`)
- **Blocker**: data accumulation
- **Priority**: P1 — operators need visibility before flipping
  `LOAD_SAPIENS` (1.4) or extending capture surface (2.2, 2.3)
- **Owner-hint**: one engineer
- **What it shows**:
  - Distribution histograms per-tier (composite, tier1_dynamic_degree,
    tier2_pose_temporal_variance, tier3_score)
  - Retake rate (% of clips with recommendation==retake)
  - Validator-vs-user disagreement (validator==pass but user retook;
    validator==retake but composition_clips includes the row)
  - p50 / p95 latency per tier
  - Per-validator_version cohort (so a version bump's effect is legible)
- **Backend support**: new `GET /v1/system/metrics/validator?since=<ts>`
  endpoint or extend existing `/v1/system/metrics` (per the original plan
  §A3). Should aggregate; do NOT page raw rows.

### 2.2 Gallery hooks (the orphaned `user_signals_log` helper)

- **Trigger**: when the operator's untracked `gallery.py` + `gallery.jsx`
  PR lands (currently `??` in BFF working tree per Wave 3 plan deviation)
- **Effort**: 30 min (wire `user_signals_log()` into 3 routes:
  `download_media`, `export_composition`, `resubmit_job`)
- **Blocker**: gallery PR landing first
- **Priority**: P1 — without this, only `mcp_event` signals flow;
  download / export / resubmit signals are missing from the corpus
- **Owner-hint**: one engineer (15-min code, 15-min smoke)
- **Note**: per the Wave 3 plan deviation, the `user_signals_log` helper
  ships in `bff/src/noodlefinger_bff/audit.py` already; just needs the
  three gallery routes to call it. This is the cheapest unblock-Phase-C
  task on the board.
- **Risk**: signal_strength weighting in 4.1 assumes these signals exist
  — without them, only 1 of 4 pair-source channels is live (mcp_event /
  retake), reducing the corpus dramatically.

### 2.3 prompt_embedding lazy-fill via llama-swap

- **Trigger**: data layer can support `find_similar_shots` query (Phase B
  prerequisite, see 3.2)
- **Effort**: 2-3 days
- **Blocker**: none — Gemma is already loaded; just config + helper +
  backfill
- **Priority**: P1 — Phase B blocked on this
- **Owner-hint**: one engineer + one llama-swap config touch
- **Action**:
  1. Extend llama-swap config with `/v1/embeddings` endpoint pointing at
     the already-resident Gemma 3 12B (llama.cpp ships the embeddings API
     natively — no new model)
  2. Add `chat_manager.embed(text: str) -> bytes` returning float32-
     packed 3584-dim embeddings (`np.array(...).astype('float32').tobytes()`)
  3. Wire into `_dispatch_validator` so newly-completed jobs get embed +
     UPDATE in the same task (zero extra round-trip — already in a fire-
     and-forget background task)
  4. One-time backfill job: scan `generations WHERE prompt_embedding IS
     NULL AND created_at > <cutoff>` and embed in batches of 64
- **Output verification**: `SELECT COUNT(*) FROM generations WHERE
  prompt_embedding IS NOT NULL` should grow monotonically; embeddings are
  ~14.3 KB / row → 14 MB / 1000 rows; trivially fits in WAL.

### 2.4 Validator audit log → user-visible

- **Trigger**: any validator score becomes user-actionable (i.e., the
  retake hook is being used in production by operators other than the
  primary)
- **Effort**: 1-2 days (BFF endpoint + dashboard widget showing recent
  validator decisions per bearer)
- **Blocker**: 2.1 (dashboard panel) — same query infra
- **Priority**: P2 (single-tenant doesn't need cross-bearer audit yet)
- **Owner-hint**: one engineer

### 2.5 On-complete dispatch failure observability

- **Trigger**: any sign that validator dispatch is silently dropping (e.g.,
  1.2 reveals a low scored/total ratio)
- **Effort**: half-day
- **Blocker**: none
- **Priority**: P1 — silent failures here corrupt the corpus invisibly
- **Owner-hint**: one engineer
- **Why this is here**: the original plan explicitly designs `_dispatch_
  validator` as fire-and-forget. That's the right call for queue throughput
  but means a hung sapiens client / RAFT OOM / chat-503 burst can fail to
  populate the score WITHOUT any user-visible signal beyond a `WARN` log
  line per failure. The operator gets no red dot anywhere.
- **Action**: add a counter to `/v1/system/metrics`:
  - `validator_dispatch_total{outcome="success|failure|skipped"}`
  - `validator_dispatch_latency_seconds`
  - threshold-alert in dashboard panel 2.1 if `failure / total > 5%` over
    a sliding 1 h window
- **Note**: this is the most surprising omission from the original plan
  — see Section 8 cross-references.

---

## Section 3 — PHASE B: RETRIEVAL (4-8 weeks out)

Phase B's keystone deliverable is `find_similar_shots(prompt, k)` — an
MCP tool that turns the LLM caller into someone who can say "for prompts
like yours, X strategy worked, Y didn't." Everything in this section is
in service of that.

### 3.1 sqlite-vec extension load

- **Trigger**: ≥1000 generations with `prompt_embedding` populated
- **Effort**: 2-3 days
- **Blocker**: 2.3 (prompt_embedding backfill)
- **Priority**: P1
- **Owner-hint**: one engineer
- **Action**: load `sqlite-vec` `.so` extension in `history_store.py`'s
  connection setup; create `clip_embeddings` virtual table; HNSW index on
  the embedding column. Wire `enable_load_extension(True)` in the conn
  config; load extension once on startup; verify with a synthetic
  `vec_distance_l2` query.
- **Risk**: extension must load before WAL writers — wire in `_open_db`
  before any user query touches the connection. Don't ship without testing
  on a clone of production `history.db` (extension load + HNSW build is
  one-time work but indices are sized to the corpus).

### 3.2 `find_similar_shots(prompt, k)` MCP tool

- **Trigger**: 3.1 done **AND** ≥500 history rows with **both**
  `prompt_embedding` AND `validator_score` populated
- **Effort**: 2-3 days (MCP tier-1 tool + backend
  `/v2/embeddings/search` endpoint)
- **Blocker**: 3.1
- **Priority**: P0 (Phase B's keystone)
- **Owner-hint**: one engineer
- **Action**: vector similarity → returns top-K shots + their
  `validator_score` + retake outcome (was it kept? was it parented?). MCP
  tool signature: `find_similar_shots(prompt: str, k: int = 5,
  min_validator_score: float | None = None) -> list[ShotRecord]`. Backend
  endpoint takes the prompt, embeds it via `chat_manager.embed`, runs
  `vec_distance_l2` against `clip_embeddings`, joins to `generations` for
  outcome metadata.

### 3.3 `recommend_loras(prompt, motion_intent)` MCP tool

- **Trigger**: 3.2 done **AND** per-LoRA validator-score aggregates
  available (i.e., `lora_applied_id` populated on enough rows)
- **Effort**: 2 days
- **Blocker**: 3.2
- **Priority**: P1
- **Owner-hint**: one engineer
- **Action**: rank LoRAs by historical mean `validator_score` on prompts
  similar to caller's. Statistically: aggregate `validator_score` per
  `lora_applied_id` per prompt-cluster. Returns top-K LoRA IDs + expected
  composite-score delta. Wire as MCP tier-1 tool.

### 3.4 Frame-level CLIP embeddings

- **Trigger**: visual-similarity retrieval is wanted (e.g., "find clips
  that look like this still")
- **Effort**: 1 week (CLIP-ViT model load, thumbnail batch process,
  sqlite-vec table)
- **Blocker**: ~340 MB CLIP model + backfill compute (~1 week one-time
  on existing thumbnails — 10k+ rows × 30 ms = 5 min if batched on
  cuda:0 idle). Cuda:0 contention with LTX is a real concern; gate this
  to off-hours via systemd timer or slot into the same lazy-load /
  evict pattern as RAFT.
- **Priority**: P2
- **Owner-hint**: one engineer

### 3.5 Phrase-aware audio slicing (allin1 sidecar)

- **Trigger**: beat-aligned slicing (MCP v0.5.0) shows phrase-misalignment
  artifacts (a measured signal — e.g., validator_score systematically
  drops on beat-aligned MVs vs uniform on certain genres)
- **Effort**: 1-2 weeks (allin1 sidecar standup + MCP integration; allin1
  is BSD; needs separate venv, similar to madmom)
- **Blocker**: empirical evidence that beat alignment is insufficient —
  don't build this on intuition
- **Priority**: P2
- **Owner-hint**: one engineer

---

## Section 4 — PHASE C: TRAINING (8-16 weeks out)

### 4.1 DPO pair construction script

- **Trigger**: ≥1000 high-strength preference pairs derivable from history
- **Effort**: 1 week
- **Blocker**: corpus accumulation (the capture machine running for 30+
  days at ~50-100 video jobs/day)
- **Priority**: P0 (Phase C blocker)
- **Owner-hint**: one engineer + one analysis agent
- **Action**: ETL script grouping `generations` rows by
  `shot_config_key`, joining `composition_clips` (kept) +
  `parent_clip_id` (retaken) → emit pairs into `preference_pairs` table
  with `signal_source` and `signal_strength` per the original plan §4.
- **Pair sources** (from melodic-sniffing-beacon.md §4):
  - **User retake** (`signal_strength=0.9` — explicit; chosen=parent,
    rejected=retake_target). Highest-confidence label.
  - **Validator + user accept** (0.7 — calibration data; chosen=
    user_kept, rejected=hypothetical-retake-not-taken; only useful as
    calibration, not training)
  - **Composition export** (0.5 — implicit kept; chosen=`composition_
    clips` member, rejected=sibling clips at same shot_uuid never used)
  - **Validator hard fail** (0.3 — synthetic; chosen=any-kept, rejected=
    `composite_score < 0.3` clips that were never re-tried). Useful for
    aversion training but lowest signal-strength.
- **Schema reminder**: `preference_pairs(pair_id, chosen_clip_id,
  rejected_clip_id, signal_source, signal_strength, used_in_training_run_
  id, created_at)` lives in v3 schema. `used_in_training_run_id` foreign-
  keys back to `training_runs.run_id`.

### 4.2 First DPO LoRA training run

- **Trigger**: 4.1 produces ≥1000 high-strength pairs
- **Effort**: 6-12 GPU-hours overnight (Blackwell sm_120, LTX-22B base,
  DPO loss, single-tier LoRA output)
- **Blocker**: 4.1
- **Priority**: P0
- **Owner-hint**: one engineer + DPO-experienced reviewer
- **Action**:
  - Run training off-hours (cuda:0 contention with LTX is total — no
    overlap possible). Pause `taco-backend` during the run; coordinate
    via `POST /v1/system/pause`.
  - A/B blind preference test against baseline (10 prompts × 2
    generations × A-then-B blind survey).
  - Auto-deploy via `lora_registry` if ≥60 % win rate (per plan §C);
    `training_runs.deployed_at` populated.
  - Track `validator_score` of LoRA-applied clips for regression
    detection.

### 4.3 Per-genre LoRAs

- **Trigger**: corpus stratifies (≥100 pairs per genre) — likely 6+
  months out at single-operator pace
- **Effort**: 2-3 weeks per genre LoRA
- **Blocker**: data volume + reliable genre tagging (this surfaces
  another sub-task: how is genre derived? candidate: from MV input
  metadata or from llama-swap classification of audio uploads)
- **Priority**: P2
- **Owner-hint**: one engineer per genre

### 4.4 Continuous training cron

- **Trigger**: 4.2 has shipped a winner once
- **Effort**: 1 week (weekly retrain cron, A/B framework, auto-promote /
  deprecate)
- **Blocker**: 4.2
- **Priority**: P1
- **Owner-hint**: one engineer
- **Action**: systemd timer firing `weekly_retrain.py`. Eligibility gate:
  ≥200 new preference_pairs since last training_run; otherwise skip.
  Auto-deprecate prior LoRA if new one wins blind A/B.

---

## Section 5 — PHASE D+: VALIDATOR SELF-IMPROVEMENT (year+ out)

### 5.1 Validator threshold auto-tuning per user

- **Trigger**: enough validator-vs-user disagreement data per bearer
  (≥100 disagreements per bearer)
- **Effort**: 1 week
- **Blocker**: external bearers exist + their gallery hooks fire signals
  consistently
- **Priority**: P2

### 5.2 Active learning loop

- **Trigger**: borderline-score clips need explicit labeling for high-
  value training data (composite scores in 0.45-0.55 range)
- **Effort**: 2-3 weeks (UI surface + signal collection)
- **Blocker**: 2.4 (validator audit log surface), 2.2 (gallery hooks)
- **Priority**: P2

### 5.3 Judge prompt versioning + recalibration

- **Trigger**: Gemma model upgrade or systematic disagreement with user
  signals
- **Effort**: 1 week + ~10 GPU-hours re-validation
- **Priority**: P2
- **Note**: `VALIDATOR_VERSION` should bump when `JUDGE_PROMPT_V1` text
  changes; cache invalidation handled by the rc1 UNIQUE index. Operator
  decision: re-validate full corpus (10 GPU-hours) vs lazy invalidation.

---

## Section 6 — RISK REGISTER

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Reward hacking (LoRA over-fits to validator) | Medium | High | Multi-source `signal_strength` weighting; user signals always weighted higher than synthetic; A/B blind preference test gates auto-deploy |
| Distribution shift on LTX base bump | Medium | High | Schema captures `base_model_sha` in `training_runs`; partial retain via re-validation; `VALIDATOR_VERSION` bump on base bump |
| Sampling bias (operator's tastes overrepresented) | High | Medium | Per-user LoRAs; shared LoRAs only when corpus crosses N users; document in CLAUDE.md |
| Validator drift (Gemma judge upgrade) | Low | Medium | Validator versioning; re-run on training corpus before next train (5.3) |
| Cold-start gap (no data yet for Phase B) | Inherent | None | Capture is logging-only; B retrieval is useful with even 10 examples (no minimum gate) |
| Privacy gap on future external bearer | Low | High | `api_key_metadata.training_opt_in` is the spine; default-OUT for unknown bearers (verified rc3 docstring fix); flip to opt-in is one explicit INSERT |
| GPU contention (sapiens + LTX cuda:1) | Medium | Low | `_stop_cuda1_tenants` coordination in place (rc3 verified) |
| Validator artifact disk growth | Low | Low | Cleanup loop extends to 30-day retention (per plan A3 hook) |
| MCP→BFF tee outage cascades | Low | Low | Fire-and-forget 2 s timeout; logs WARN; doesn't propagate |
| Validator dispatch silently fails | Medium | High | Item 2.5 — add counter + dashboard alert; this is the **most under-specified risk** in the original plan (see §8) |
| sqlite-vec extension breaks on PRAGMA upgrade | Low | High | Pin extension version; test on clone before any HNSW rebuild |
| DPO over-fits to single-operator preferences | High | Medium | 4.3 per-genre stratification + 4.4 continuous retrain so model never freezes on a stale snapshot of taste |
| `Type=exec` masks sapiens weight-load failure | Medium | Medium | Switch to `Type=notify` before flipping LOAD_SAPIENS=1 (1.6 includes this) |

---

## Section 7 — Decision points the operator should track

These are the empirical signals that should trigger a roadmap re-prioritization
when they appear. Watch for:

- **"Validator scores cluster at 0.7"** → distribution is healthy; threshold
  OK; proceed with Phase B prep
- **"Validator scores cluster at 0.95"** → either retake threshold is too
  low, OR content is uniformly excellent; check tier-3 verdict distribution
  first (if `pass` rate is also 99 %, judge prompt is too lenient — bump
  `JUDGE_PROMPT_V1` and `VALIDATOR_VERSION`)
- **"Validator scores cluster at 0.3-0.5"** → composite formula is mis-
  calibrated; tier-1 normalization (`min(dyn / 5.0, 1.0)`) is the most
  likely culprit on Blackwell-generated content
- **"Disagreement rate >30 %"** → recalibrate or update judge prompt (5.3)
- **"Retake budget consistently exhausted"** (`max_retakes_per_clip=1` and
  most clips hit the cap without succeeding) → validator over-aggressive,
  or content is genuinely failure-prone on a specific shot type
- **"Composition export latency > 30 s"** → `composition_clips` writes
  blocking; consider async batching (the v3 schema makes this trivially
  rewritable as an async background task)
- **"`user_signals` growth disproportionate to MV count"** → tee is double-
  firing somewhere; investigate MCP→BFF event tee + gallery hooks
- **"Validator score AVG drops after deploying a new LoRA"** → trained LoRA
  is regressing; auto-deprecate via 4.4
- **"Cohort of clips has `validator_score IS NULL`"** → on-complete dispatch
  is failing for a class of jobs; trigger 2.5 investigation immediately
- **"sapiens-sidecar `/health` returns ready=false for >5 min"** → 1.6
  `Type=notify` switch was overdue; audit the lifespan hook

---

## Section 8 — Cross-references and surprising omissions

**Cross-references**:

- `docs/CAPTURE_VALIDATOR.md` — architecture (sibling doc)
- `docs/MCP.md` — MCP integration (tier-0 + tier-1 surface)
- `docs/operator-tuning.md` — env vars (`LOAD_SAPIENS`, `VALIDATOR_VERSION`,
  rate-limit caps)
- `docs/MV_EDITING.md` — composition / `cut_music_video` grammar
- `CHANGELOG.md` — ship history (v1.17.0-rc1 / rc2 / rc3 entries)
- `/home/ian/.claude/plans/melodic-sniffing-beacon.md` — original plan

**Most surprising omission noticed in the original plan** (folded into
this roadmap as item 2.5):

The original plan correctly designs `_dispatch_validator` as fire-and-
forget so the queue worker never blocks. That's the right call for
throughput. But the plan has **no observability for dispatch failures** —
if RAFT OOMs, sapiens hangs, or Gemma chat 503s in a burst, the dispatch
silently drops the score for those jobs. The operator gets WARN log lines
per failure but no aggregate signal: no counter, no dashboard widget, no
threshold alert. A multi-day outage of (say) Gemma chat would corrupt the
corpus invisibly — `validator_score` would be NULL on a cohort of rows
and downstream Phase C pair construction (4.1) would silently filter
those rows out, biasing the training set toward "validator dispatch was
healthy that day."

The fix is item 2.5: add a `validator_dispatch_total{outcome=...}`
counter to `/v1/system/metrics` and surface a dashboard alert in 2.1 if
`failure / total > 5%` over a sliding 1 h window. Effort is half-day;
priority should be P1 because the corpus quality depends on it.

A secondary smaller omission: the plan describes `signal_strength`
weighting for pair sources (§4) but doesn't specify what happens to a
pair when **both** signals point the same way (e.g., user retook AND
validator scored it as `retake`). The expected behavior is signal_
strength=max(0.9, 0.3) = 0.9 with a `signal_source=user_retake` (most-
explicit wins), but this should be documented in 4.1's ETL spec before
the script is written.
