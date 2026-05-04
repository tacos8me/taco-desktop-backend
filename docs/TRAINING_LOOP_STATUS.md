# TRAINING_LOOP_STATUS.md — Where the training loop stands

**Snapshot date**: 2026-05-04 (post-v1.20.0)

This is the operator-facing progress snapshot for the
`capture → validate → rate → preference-pairs → train → A/B → rollback`
loop and the parallel L2.5 exemplar fine-tune path. It names what is
wired today, what is explicitly not yet shipped (with task-list refs so
nothing falls between the cracks), and where to read each contract.

If you want the third-party client surface, jump to
[RATING_LORA_INTEGRATION.md](./RATING_LORA_INTEGRATION.md). If you want
the operator runbook for the first SFT cycle, jump to
[PHASE_C_TRAINING_RUNBOOK.md](./PHASE_C_TRAINING_RUNBOOK.md). This doc
sits one level up: it is the map of which doors are open.

## Where we are

- **Phase 1 (rating + exemplar backend)**: SHIPPED in v1.20.0.
- **Phase 2 (dashboard rating UIs)**: SHIPPED in leatherjacket-dashboard
  (separate repo). `SHIP` verdict from the Phase 2 review.
- **Phase 3 prereqs**: SHIPPED — schema v8 (`system_flags` kill-switch +
  `pending_construction_until` quarantine), Path C′ ETL writer, async
  LoRA build path returning 202 in <500 ms, cross-cohort diversity in
  the rating queue.
- **Phase 3 build (first L2.5 LoRA from real exemplars)**: PENDING —
  operator-driven, ~10-15 GPU-hours in an off-hours window.
- **Phase 4 (calibration soak metrics + ship-readiness sign-off)**: NOT
  STARTED.

## What is wired end-to-end

Every link in this chain is in the live codebase as of v1.20.0:

1. **Capture** → every video v2 request body persists `parent_clip_id`,
   `shot_uuid`, `shot_config_key`, `composition_id`, `lora_applied_*`,
   `motion_intent`, `ab_arm` into `generations` (rc1..rc6 lineage +
   rc5 cohort).
2. **Validate** → `_on_job_complete` fires `_dispatch_validator(job)` on
   every opted-in completed video; tier-1 RAFT + tier-2 real Sapiens-2
   pose + tier-3 Gemma judge produce `validator_score` +
   `validator_payload_json` + `validator_version`.
3. **Surface borderline clips** → `GET /v1/ratings/queue` returns the
   active-learning slice (validator_score in
   `[RATING_QUEUE_RETAKE, RATING_QUEUE_PASS]` ± per-bearer threshold
   override clamped global ± 0.15) with cross-cohort diversity merging
   so a rater never sees three of the same cohort in a row.
4. **Operator rates** → `POST/PATCH/DELETE /v1/clips/{id}/rating[/{rid}]`.
   Anti-fatigue gate (2-second min + 200/90min session cap) returns
   `RATE_LIMIT_TOO_FAST` / `SESSION_CAP_EXCEEDED`. Validator-score
   blur-until-click on the dashboard mitigates anchoring.
5. **Path C′ ETL** → `scripts/construct_human_rating_pairs.py` runs
   hourly. Human-rating preference_pairs land with
   `signal_strength=0.75` and `pending_construction_until=now+86400`
   (24h quarantine). The hourly pass re-validates 5 invariants (bearer
   opt-in, validator composite NOT NULL, cross-bearer consent,
   `validator_visible_at_rating` filter, source-pause flag) and either
   commits the pair (clears the timestamp) or deletes it.
6. **SFT-on-chosen training** → `scripts/train_dpo_sft.py` SELECT now
   filters `AND pending_construction_until IS NULL`, so quarantined
   pairs cannot enter training. Defense-in-depth dry-run by default;
   `--execute` is required to consume GPU. Snapshots dataset to
   `training_runs/<run_id>/dataset.jsonl`, persists full reproducibility
   metadata (`training_seed` + `hyperparams_json` +
   `dataset_snapshot_path` + `code_sha` + `validator_version_at_train`),
   marks consumed pairs with `used_in_training_run_id`, registers in
   `lora_registry` as a candidate (NOT auto-deployed).
7. **A/B harness** → `scripts/ab_decision.py` weekly cron, paired
   t-test on per-MV mean validator_score across `_ab_arm` cohorts.
   Promote ≥+10% AND p<0.05; deprecate ≤-5% AND p<0.05; insufficient
   <30 MVs/arm; manual Welch's-t fallback when scipy unavailable.
8. **Rollback** → `POST /v1/system/lora/rollback` (admin) verifies the
   `lora_id` is the current `MCP_PRODUCTION_LORA`, sets
   `training_runs.deprecated_at`, finds the previous deployed +
   not-deprecated run, atomically rewrites `MCP_PRODUCTION_LORA=<prev>`
   in `.env`, returns the audit shape. Applies on next process
   restart that re-reads `.env`.
9. **L2.5 exemplar fine-tune (parallel to the SFT loop)** → operator
   stars clips into a set via `POST /v1/exemplar-sets/{id}/members`,
   triggers a build via `POST /v1/exemplar-sets/{id}/build-lora`
   (admin, async, returns 202 in <500 ms),
   `scripts/build_lora_from_exemplars.py` runs the trainer subprocess.
   `GET .../builds` polls the status state machine; `POST
   .../build/{run_id}/cancel` SIGTERMs the registered process.

## What is NOT yet shipped

Linked to the live task list so nothing falls through:

- **#15 — Phase 3: L2 session priors** (dashboard-side). Per-session
  preference smoothing on top of L1's per-LoRA `session_boosts` map.
  L1 is wired and persisted in localStorage; L2 priors layer next.
- **#16 — Phase 3: active-learning queue UI** (dashboard-side). Backend
  `/v1/ratings/queue` is live. Dashboard surface for borderline-only
  thumbs-up/down (composite ∈ [0.45, 0.55]) is the next UI ship; design
  open.
- **#17 — Phase 3: first real LoRA build** (operator-driven, ~10-15
  GPU-hours). Needs ≥20 curated exemplars + 80 GB cuda:0 in an
  off-hours window. See "Open operator action items" below.
- **#18 — Phase 3 review: e2e LoRA loop**. Builds on #17 — review the
  first end-to-end exemplar build with the deployed candidate behind
  an A/B arm.
- **#19 — Phase 4: calibration soak metrics**. Time-frozen 30-clip
  gold-standard set re-rated quarterly to detect operator drift (per
  the Phase 1 safety review §6.3).
- **#20 — Phase 4 review: ship-readiness sign-off**.
- **#26 — Phase 2 polish: parent_job_id + session_boost badge**. Two
  small dashboard-side polish items deferred from the Phase 2 ship.

## Open operator action items

These are blocking or near-blocking from prior reviews. Resolve in the
order listed.

1. **Tier-3 Gemma judge upstream is 502'ing**. The
   `gemma-4-31b-it` model on llama-swap is returning 502 — every video
   completed since the outage has tier-3 absent, which weakens
   composite scoring (silent false-passes accumulating in the corpus).
   Restart the llama-swap upstream, then run `POST
   /v2/system/bulk-revalidate` (admin-gated, default `dry_run=true`)
   with `{"target_validator_version": <current>}` to recompute. This
   is the single highest-leverage fix; do this first.
2. **Sidecar redeploys (Modal + RunPod)**. v1.20.0 plumbs
   `gen_config_overrides` through the local sidecar. The Modal +
   RunPod sidecars need a redeploy so the per-request operator knobs
   actually apply on remote workers. Pre-deploy sidecars silently
   ignore the new field — no break.
3. **Bulk-revalidate clips since the judge went down** (after #1 is
   resolved). Same `POST /v2/system/bulk-revalidate` recipe.
4. **`BACKEND_BEARER` on the leatherjacket-dashboard process**
   (operator-machine env). The rating proxy needs this set to
   authenticate to taco-backend; otherwise the rating UI shows 401
   loops.
5. **First L2.5 build prerequisites**: ≥20 curated exemplars in a set
   + 80 GB cuda:0 free in an off-hours window. Tue 02:00–10:00 UTC is
   the suggested window per the Phase C runbook so the build does not
   contend with daily MV authoring traffic.

## Decision points for further work

- **Diffusion-DPO upgrade path** (Phase C.1). Locked deferred per
  Phase 0 review. When SFT-on-chosen v0.0.1 ships and proves out via
  A/B, revisit. The schema-v4 reproducibility columns
  (`training_seed` + `hyperparams_json`) cover the diff cleanly.
- **Multi-tenant ergonomics**. The `cross_bearer_rating_consent_at`
  column is wired in v1.20.0 (`api_key_metadata`). UI for a 2nd
  bearer to grant / revoke cross-bearer rating consent lives in the
  BFF, not taco-backend; design open until a 2nd bearer arrives.
- **Active-learning loop expansion**. Today the queue surfaces
  validator borderline (composite ∈ [retake, pass]). #16 expands the
  surface to explicit thumbs-up/down on a tighter band [0.45, 0.55].
  UI design open.
- **Per-MV vs per-genre vs cross-MV LoRA strategy**. Today L2.5
  produces per-MV LoRAs. Phase D candidates explore per-genre and
  cross-MV averaged. Defer until at least 3 per-MV LoRAs have shipped
  and we have A/B data to compare.
- **Held-out gold-standard set**. Time-frozen 30-clip set re-rated
  quarterly to detect operator drift (per safety review §6.3).
  Implementation tracked under #19.

## Cron status

- **`preference-pairs.timer`** is ARMED (Mon 04:00 UTC). Verify with
  `systemctl --user list-timers | grep preference`. Runs
  `scripts/construct_preference_pairs.py` (and, in v1.20.0+, the
  hourly Path C′ writer `scripts/construct_human_rating_pairs.py`).
- **`ab-decision.timer`** is ARMED (weekly). Reads
  `generations.ab_arm` and writes promote/deprecate decisions to
  `training_runs`.
- **LoRA build trigger** is OPERATOR-DRIVEN, not crontab. Per the plan,
  every build is admin-initiated via `POST
  /v1/exemplar-sets/{id}/build-lora` so the cost (~10-15 GPU hours
  per L2.5 build, ~50-60 GPU hours per SFT run) is always opted into.

## Test coverage snapshot

370 green / 0 fail. Of those, 22 are explicitly safety-marked across:

- `tests/test_human_ratings.py` (28 tests) — rating endpoints,
  thresholds, recommend-loras boosts, safety invariants
- `tests/test_exemplar_lora_build.py` (14 tests) — exemplar set CRUD,
  privacy, dry-run, cancel, status state machine
- `tests/test_v1_20_schema.py` (9 tests) — v6 → v7 → v8 migrations,
  delete cascade across `delete_rater_corpus`
- `tests/test_phase3_etl.py` (10 tests) — Path C′ re-validation +
  pause flag
- `tests/test_sidecar_pass_through.py` (7 tests) — turbo dispatch +
  Modal/RunPod save-apply-restore
- `tests/test_gen_config_overrides.py` (14 tests) — stage 1 / VAE
  validation, atomic write, audit log

Key safety invariants under test:

- **Length-coupling**: caller cannot smuggle a longer-than-9 sigma list
  to extend stage 1 step count.
- **Value-range**: sigma values clamped to `[0, 1]`; VAE tile sizes
  bounded.
- **Privacy gate**: a bearer cannot see another bearer's clips, even
  via the rating queue or recommend-loras.
- **Opt-in re-check**: `PATCH /v1/clips/{id}/rating/{rid}` re-reads
  `training_opt_in` so a mid-session opt-in flip is honored.
- **Atomic write**: `.gen_config.json` write goes through a tempfile +
  `os.replace`; the audit log appends only after the write succeeds.

## Where to read the contracts

- [RATING_LORA_INTEGRATION.md](./RATING_LORA_INTEGRATION.md) (643
  lines, dashboard-agnostic) — copy-pasteable curl per endpoint;
  privacy gate cheat-sheet; error envelope; integration testing
  recipe; common gotchas. This is the third-party client surface.
- [PHASE_C_TRAINING_RUNBOOK.md](./PHASE_C_TRAINING_RUNBOOK.md) — the
  operator runbook for the first SFT cycle.
- [CAPTURE_VALIDATOR.md](./CAPTURE_VALIDATOR.md) — the validator
  pipeline reference (RAFT + Sapiens-2 + Gemma judge).
- [PRIVACY_GOVERNANCE.md](./PRIVACY_GOVERNANCE.md) — what is captured,
  who can see it, what enters training, how opt-out works.
- [DECISIONS.md](./DECISIONS.md) — the append-only ADR log; every
  load-bearing decision in the loop has a record there.
- [../CHANGELOG.md](../CHANGELOG.md) — per-version delta log.
- [../CLAUDE.md](../CLAUDE.md) — the codebase cheat-sheet, including
  per-version highlights.
