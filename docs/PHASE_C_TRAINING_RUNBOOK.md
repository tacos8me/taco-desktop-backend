# Phase C Training Runbook

Operator playbook for the SFT-on-chosen LoRA training pipeline shipped
in v1.18.0-rc3. **The first real training run waits until the corpus
crosses ~1000 pairs (~6-8 weeks from rc3 ship at single-operator
volume).** Until then this runbook is preparatory: validate the ETL,
seed the watermark, dry-run the trainer.

## Section 1 — Pre-flight

Before running anything, verify the schema is at v4 and the corpus
has the lineage signals required for pair construction.

```bash
sqlite3 /mnt/nvme-1/servers/taco-backend/history.db <<EOF
PRAGMA user_version;            -- must be 4
SELECT COUNT(*) FROM generations
  WHERE validator_score IS NOT NULL
    AND validator_version = '$(grep VALIDATOR_VERSION config.py | head -1 | cut -d'"' -f2)';
SELECT COUNT(*) FROM generations WHERE parent_clip_id IS NOT NULL;
SELECT COUNT(*) FROM generations WHERE shot_config_key IS NOT NULL;
SELECT COUNT(*) FROM composition_clips;
SELECT COUNT(*) FROM api_key_metadata WHERE training_opt_in = 1;
EOF
```

You're ready when:

- `parent_clip_id` count > 100 (retake provenance flowing)
- `shot_config_key` count > 500 (cohort lookup viable)
- `composition_clips` count > 50 (kept-vs-not-kept signal viable)
- ≥ 1 opt-in api_key row

If any are 0, capture is not yet writing the lineage columns. Verify
that mcp v0.7.0+ is the deployed orchestrator and that `/v2/retake`
+ `/v2/compositions/{id}/export` are flowing.

## Section 2 — Pair construction (weekly cron)

Installed as a systemd user timer (no crontab entry needed). Runs every
Monday at 04:00 UTC with a randomized jitter up to 10 minutes; missed
runs fire on next boot via `Persistent=true`.

Unit files:

- `/home/ian/.config/systemd/user/preference-pairs.service` — oneshot
  invoking `uv run --no-sync python scripts/construct_preference_pairs.py
  --since-watermark` from the project root. Inherits `HOME` + `PATH` so
  the same `/home/ian/.local/bin/uv` resolves as it does for
  `taco-backend.service`. `TimeoutStartSec=600`.
- `/home/ian/.config/systemd/user/preference-pairs.timer` — `OnCalendar=Mon
  04:00 UTC`, `RandomizedDelaySec=600`, `Persistent=true`.

Operator commands:

```bash
# Disable temporarily (e.g. before a planned DB maintenance window)
systemctl --user stop preference-pairs.timer

# Re-enable
systemctl --user start preference-pairs.timer

# Trigger one immediate run (does NOT touch the timer schedule)
systemctl --user start preference-pairs.service

# Inspect last run's logs
journalctl --user -u preference-pairs -n 200

# Confirm next-run timestamp
systemctl --user list-timers preference-pairs.timer
```

**Expected behavior during corpus accumulation**: at single-operator
volume the four sources will return 0 rows for several weeks after rc1
ships. The pre-rc1 history rows have NULL in `parent_clip_id` /
`shot_config_key` / `composition_id`, and the rc5+ corpus needs to
accumulate ~100 retakes before `user_retake` finds signal. A clean
log line of `source=<name> rows=0 (would-insert)` per source is the
expected steady state until the corpus crosses ~1000 pairs (~6-8 weeks).
A non-zero exit or a SQLite error in `journalctl` is the only thing
that warrants attention.

Manual invocation:

```bash
cd /mnt/nvme-1/servers/taco-backend
uv run --no-sync python scripts/construct_preference_pairs.py --dry-run   # no writes
uv run --no-sync python scripts/construct_preference_pairs.py             # incremental (default)
uv run --no-sync python scripts/construct_preference_pairs.py --full-rebuild  # ignore watermark
uv run --no-sync python scripts/construct_preference_pairs.py --source user_retake  # debug one source
```

Output prints per-source counts. Watermark is `.preference_pairs_watermark`
in the repo root — clip its mtime to inspect last run.

Sanity check after each run:

```sql
SELECT signal_source, COUNT(*) AS n, AVG(signal_strength) AS s
FROM preference_pairs
WHERE validator_version = '<current>'
GROUP BY signal_source;
```

## Section 3 — First training run (operator gate)

The trainer **defaults to dry-run**. To actually consume GPU you must
pass `--execute`. ~50-60 GPU-hours per run on the Blackwell PRO 6000.

**Pre-conditions for the first run:**

1. ≥1000 chosen_clip_ids meeting `signal_strength >= 0.5` AND
   `validator_version = config.VALIDATOR_VERSION` AND
   `used_in_training_run_id IS NULL`.
2. cuda:0 free for ~60 hours (run a turbo-mode-stop first if needed).
3. Off-hours window confirmed (Tuesday 2-10 AM UTC × ~7 nights covers
   one full run; longer if you batch).
4. A snapshot backup of `history.db` exists (`cp history.db history.db.pre-train`).

Run the dry-run first to confirm dataset shape:

```bash
cd /mnt/nvme-1/servers/taco-backend
uv run --no-sync python scripts/train_dpo_sft.py \
    --config configs/sft_quality_lora.yaml
```

The dry-run prints expected `train_on` / `eval_on` counts and exits
without touching the GPU.

When you're sure, execute:

```bash
nohup uv run --no-sync python scripts/train_dpo_sft.py \
    --config configs/sft_quality_lora.yaml \
    --execute > logs/sft-quality-v0.0.1.log 2>&1 &
```

The script:

1. Snapshots the dataset to `training_runs/<run_id>/dataset.jsonl`.
2. Splits 90/10 train/eval.
3. Loads ltx-2.3 base, attaches LoRA via PEFT, freezes base weights.
4. Trains with `bf16 + paged_adamw_32bit + gradient_checkpointing`
   (LoRA-only).
5. Checkpoints every epoch to `training_runs/<run_id>/checkpoint-N/`.
6. Saves the final LoRA to `loras/<run_id>.safetensors`.
7. Persists `training_runs` row with full reproducibility metadata.
8. Marks consumed pairs as `used_in_training_run_id = run_id`.
9. Registers the LoRA in `lora_registry` as a candidate (NOT
   auto-deployed — A/B harness handles promotion).

Resume from a checkpoint:

```bash
uv run --no-sync python scripts/train_dpo_sft.py \
    --config configs/sft_quality_lora.yaml \
    --execute \
    --resume-from training_runs/sft-quality-v0.0.1/checkpoint-2/
```

## Section 4 — A/B monitoring

After training completes, the candidate LoRA needs A/B validation
against the current production. Phase C v1 routes at the MV level
(global toggle in mcp orchestrator), random 50/50 per session.

Activate in the MCP orchestrator's `.env`:

```env
AB_TEST_ACTIVE=1
AB_CANDIDATE_LORA=sft-quality-v0.0.1
AB_CANDIDATE_STRENGTH=0.3
```

Add the weekly decision cron:

```cron
# Phase C: A/B decision (every Monday 5 AM UTC)
0 5 * * 1 cd /mnt/nvme-1/servers/taco-backend && AB_CANDIDATE_LORA=sft-quality-v0.0.1 uv run --no-sync python scripts/ab_decision.py >> logs/ab_decision.log 2>&1
```

Manual check:

```bash
AB_CANDIDATE_LORA=sft-quality-v0.0.1 \
uv run --no-sync python scripts/ab_decision.py
```

Decision matrix:

| Decision | Trigger | Action |
|---|---|---|
| `insufficient_samples` | <30 MVs/arm | nothing; wait |
| `promote` | delta ≥ +10% AND p < 0.05 | sets `deployed_at = now()` on training_runs row |
| `deprecate` | delta ≤ -5% AND p < 0.05 | sets `deprecated_at = now()` |
| `no_action` | thresholds not met | nothing |

To run in operator-gated mode (decisions reported but not applied):

```bash
AB_AUTO_PROMOTE=0 AB_CANDIDATE_LORA=... uv run --no-sync python scripts/ab_decision.py
# OR
uv run --no-sync python scripts/ab_decision.py --no-auto-promote --candidate-lora sft-quality-v0.0.1
```

When a candidate is promoted, manually update the mcp orchestrator's
`.env`:

```env
MCP_PRODUCTION_LORA=sft-quality-v0.0.1
AB_TEST_ACTIVE=0   # OR set a new candidate to start the next experiment
```

## Section 5 — Rollback

When a deployed LoRA produces field regressions, roll back via the
admin endpoint (admin-gated; mirrors `/v1/system/turbo` pattern):

```bash
curl -X POST http://localhost:8090/v1/system/lora/rollback \
    -H "Authorization: Bearer $ADMIN_KEY" \
    -H "Content-Type: application/json" \
    -d '{"lora_id": "sft-quality-v0.0.1", "reason": "20% retake-rate spike post-deploy"}'
```

The endpoint:

1. Verifies `lora_id` is the current `MCP_PRODUCTION_LORA`.
2. Sets `deprecated_at = now()` on the matching `training_runs` row.
3. Finds the previous deployed-and-not-deprecated LoRA.
4. Rewrites `.env` with `MCP_PRODUCTION_LORA=<previous>` (or empty).
5. Returns `{rolled_back_from, rolled_back_to, reason, applied_at, note}`.

**The rolled-back-to value applies only after taco-backend / mcp
processes restart and re-read `.env`**. The endpoint does not hot-swap
in-process state.

```bash
systemctl --user restart taco-backend
systemctl --user restart noodlefinger-mcp
```

## Section 6 — Troubleshooting

### "no eligible chosen_clip_ids" on `--execute`

Either the corpus is too thin (check section 1) or `--config`'s
`min_signal_strength` is too high. Lower it to 0.3 for synthetic
negatives included, but expect noisier training.

### CUDA OOM mid-training

Verify cuda:0 is the only LTX/Flux tenant. Stop sidecars:

```bash
systemctl --user stop joyai-sidecar ernie-image-sidecar
```

If still OOMing, reduce LoRA rank (64 → 32) or
`per_device_train_batch_size` (1 → can't go lower; instead bump
`gradient_accumulation_steps` 4 → 8 to keep effective batch).

### Crash mid-training

Resume from the last checkpoint:

```bash
ls training_runs/<run_id>/checkpoint-*/
uv run --no-sync python scripts/train_dpo_sft.py \
    --config configs/sft_quality_lora.yaml \
    --execute \
    --resume-from training_runs/<run_id>/checkpoint-N/
```

Checkpoint is saved every epoch — at 3 epochs over ~50 hours, you lose
at most ~17 hours of work to a crash.

### Eval-loss / train-loss mismatch (overfitting)

Watch the training log for `eval_loss / train_loss > 1.2`. If sustained
across two epochs, the LoRA is overfitting to single-operator
preferences. Mitigations:

- Drop `epochs` to 2 (config bump).
- Lower learning_rate (5e-4 → 2e-4).
- Increase the held-out ratio (modify `split_held_out` ratio from
  0.9 → 0.85).
- Wait for more pairs before the next training cycle.

### A/B p-value never crosses 0.05

At single-operator 4 MVs/week pace, ~15 weeks calendar to reach
30 MVs/arm. Be patient. If after 20+ weeks the p-value is still
above threshold, the candidate isn't actually better — `no_action`
is the right outcome. Discard the LoRA and try different
hyperparams or wait for a larger pair corpus.
