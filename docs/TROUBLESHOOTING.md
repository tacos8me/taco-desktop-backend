# Troubleshooting + FAQ

> Server version: v1.18.0-rc3 (2026-04-29).
> Audience: operator on-call at 2 AM. Something is broken. You need
> a flowchart, not a textbook.

## How to use this guide

Each section below follows **symptom → check → fix**. Search by the
error string, status code, or symptom you are looking at. If a section
references a deeper doc (operator-tuning, CAPTURE_VALIDATOR,
PHASE_C_TRAINING_RUNBOOK), follow it after you have stabilized — this
file is the triage layer.

The FAQ at the bottom answers the questions that come up after the
fire is out (capacity planning, adding bearers, running concurrent
MVs).

Quick index:

| # | Symptom |
|---|---|
| 1 | HTTP 429 — `per_key_queue_full` / `queue_full` |
| 2 | HTTP 503 from `/v2/embeddings/*` |
| 3 | Validator dispatch silently NULL on completed jobs |
| 4 | Sapiens sidecar returns `{"stub": true}` |
| 5 | `preference_pairs` is empty after weeks of capture |
| 6 | Training run failed mid-flight |
| 7 | LoRA deployed but quality regressed |
| 8 | BFF tee silently fails (`user_signals` empty) |
| 9 | Schema migration failed at startup |
| 10 | Turbo mode won't enable |
| 11 | cuda:0 / cuda:1 OOM |
| 12 | `find_similar_shots` returns empty |

---

## 1. HTTP 429 — rate limit / queue full

**Symptom**

Client receives one of:

```json
{"error": "per_key_queue_full", "detail": "..."}
{"error": "queue_full", "detail": "..."}
{"error": "music_queue_full", ...}
{"error": "batch_queue_full", ...}
```

Response includes `Retry-After: 30` (or similar). Common during
28+ concurrent video submissions from a single bearer or large
`cut_music_video` sessions.

**Check**

```bash
# Live counters and queue depth.
KEY=$(grep -v '^#' .api_keys | grep -v '^$' | head -1)
curl -sS -H "Authorization: Bearer $KEY" http://localhost:8090/v1/system/metrics | jq .

# Configured caps.
python3 -c "import config; print(
    'PER_KEY_QUEUE_CAP', config.PER_KEY_QUEUE_CAP,
    'MAX_QUEUE_DEPTH', config.MAX_QUEUE_DEPTH,
    'PER_KEY_MUSIC_CAP', config.PER_KEY_MUSIC_CAP,
    'PER_KEY_BATCH_CAP', config.PER_KEY_BATCH_CAP,
    'MAX_BATCH_QUEUE_DEPTH', config.MAX_BATCH_QUEUE_DEPTH,
)"
```

If counters are at the cap and `Retry-After` is short, this is
working as intended — the client should back off.

**Fix**

- **Right answer**: client honors `Retry-After`. The cap exists to
  prevent one bearer from starving the queue.
- **If the operator workload genuinely needs higher caps** (e.g. a
  200-clip MV with mcp v0.4.4+ parallel dispatch): bump the env vars
  in `.env`, restart taco-backend.

  ```
  PER_KEY_QUEUE_CAP=100      # default since v1.16.4
  MAX_QUEUE_DEPTH=200        # default since v1.16.4
  PER_KEY_MUSIC_CAP=20
  PER_KEY_BATCH_CAP=20
  MAX_BATCH_QUEUE_DEPTH=30
  ```

  Caveat: `PER_KEY_QUEUE_CAP` should stay at most half of
  `MAX_QUEUE_DEPTH` to preserve single-tenant fairness.

- See `docs/operator-tuning.md` for the full caps matrix and the
  rationale (history of v1.15→v1.16.4 increases).

---

## 2. HTTP 503 from `/v2/embeddings/*` (Phase B)

**Symptom**

```json
{"error": "embedding search not available — install sqlite-vec extension"}
```

or

```json
{"error": "embedding service unavailable"}
```

Returned by `/v2/embeddings/search` or `/v2/embeddings/recommend-loras`.

**Check 1 — sqlite-vec extension installed?**

```bash
uv run --no-sync python -c "
import sqlite3, sqlite_vec
c = sqlite3.connect(':memory:')
c.enable_load_extension(True); sqlite_vec.load(c)
print(c.execute('SELECT vec_version()').fetchone())
"
# Expect: ('v0.1.9',) or newer
```

If this errors, sqlite-vec is missing.

**Check 2 — boot log says it loaded?**

```bash
journalctl --user -u taco-backend | grep -i sqlite-vec | tail -5
```

Healthy line: `INFO history_store: sqlite-vec extension loaded`.
Failure line: `WARN history_store: sqlite-vec extension load failed (...)`.

**Check 3 — llama-swap embeddings endpoint up?**

```bash
curl -sS -X POST http://192.168.1.80:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "gemma-3-12b-nvfp4", "input": "test"}' | head -c 200
```

Healthy: JSON with `data[0].embedding[]` containing 3584 floats.
HTTP 502 / "upstream command exited prematurely" → llama-swap config
is missing the `--embeddings` flag.

**Fix**

- Install sqlite-vec: `uv pip install sqlite-vec` then restart
  taco-backend.
- Re-enable embeddings in llama-swap by adding `--embeddings` and
  `--pooling mean` to the model command in `~/llama-swap/config.yaml`,
  then restart llama-swap.
- Detail walkthrough in `docs/operator-tuning.md` →
  "Embeddings + sqlite-vec" section.

---

## 3. Validator dispatch silently NULL on completed video jobs

**Symptom**

Recently completed video jobs have NULL in `validator_score`:

```bash
sqlite3 history.db "
  SELECT COUNT(*) AS total,
         COUNT(validator_score) AS scored,
         COUNT(*) - COUNT(validator_score) AS null_count
  FROM generations
  WHERE created_at > strftime('%s','now','-1 day')
    AND model LIKE 'ltx-%';
"
```

If `null_count` is large, validator dispatch is silently failing or
being skipped.

**Check 1 — bearer opted in?**

`_on_job_complete` skips dispatch when the bearer is not opted in.

```bash
sqlite3 history.db "
  SELECT api_key_hash, training_opt_in
  FROM api_key_metadata;
"
```

Default for unknown bearers is **opt-out**. The single-tenant deploy
seeded `.api_keys` entries to `training_opt_in=1` on the rc1 v3
migration. External bearers added later need to be INSERTed manually.

**Check 2 — validator dispatch errors in journal?**

```bash
journalctl --user -u taco-backend | grep -i validator | tail -50
```

Look for `validator_dispatch_failed`, `analyze_motion_failed`, RAFT
crashes, Gemma timeouts.

**Check 3 — dispatch counter from `/v1/system/metrics`**

```bash
KEY=$(grep -v '^#' .api_keys | grep -v '^$' | head -1)
curl -sS -H "Authorization: Bearer $KEY" \
  http://localhost:8090/v1/system/metrics | jq '.validator_dispatch'
```

Expect:

```json
{
  "success": 142,
  "failure": 0,
  "skipped_not_video": 31,
  "skipped_opt_out": 0,
  "skipped_validator_disabled": 0,
  "failure_rate_pct": 0.0
}
```

`failure_rate_pct > 0` → tier-1/tier-3 broken (see #4 for tier-2).
`skipped_opt_out > 0` and growing → bearer not in `api_key_metadata`
or has `training_opt_in=0`.

**Fix**

- **Opt-in the bearer** (only if intended, after asking the user):
  ```bash
  sqlite3 history.db "
    INSERT OR REPLACE INTO api_key_metadata
      (api_key_hash, training_opt_in, tier, created_at, updated_at)
    VALUES ('<sha256 of api_key>', 1, 'pro',
             unixepoch(), unixepoch());
  "
  ```
  The hash is `sha256(api_key)` — see `_hash_key` in `history_store.py`.
- **Tier-1 errors**: most often missing CUDA / OOM during RAFT.
  Check `nvidia-smi` for cuda:0 contention.
- **Tier-3 errors**: llama-swap unreachable — `curl http://192.168.1.80:8080/v1/models`.
- **Bulk re-run on a cohort** (admin):
  ```bash
  curl -sS -X POST http://localhost:8090/v2/system/bulk-revalidate \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"target_validator_version": "1.17.0-rc5", "limit": 500, "dry_run": false}'
  ```

---

## 4. Sapiens sidecar returns `{"stub": true}`

**Symptom**

Tier-2 in `validator_payload_json` always shows the stub flag:

```json
{"tier2": {"tier2_skipped": true, "status": "stub", ...}}
```

or

```json
{"tier2": {"tier2_skipped": true, "status": "load_disabled", "latency_s": 0.0}}
```

**Check**

```bash
echo "$LOAD_SAPIENS"
grep '^LOAD_SAPIENS' /mnt/nvme-1/servers/taco-backend/.env

systemctl --user is-active sapiens-sidecar
curl -sS http://127.0.0.1:8096/health 2>/dev/null
```

`LOAD_SAPIENS=0` is the **default in v1.17.0-rc2 / rc3 / rc5**. Stub
output is intended at this stage of the rollout.

`status: "stub"` means the sidecar is up but still in stub mode (not
yet running real Sapiens-2 inference).

`status: "load_disabled"` means `LOAD_SAPIENS=0` — no HTTP call was
made, the validator short-circuits.

**Fix**

- **This is intended today.** Composite scoring already accounts for
  it — tier-2 contributes `0.2 · 1.0` (no penalty) when stubbed or
  disabled. Validator scores remain valid.
- **Do NOT flip `LOAD_SAPIENS=1` until the rc-final sidecar build
  ships real inference.** Until then, flipping the flag only adds an
  HTTP round-trip with no signal.
- When rc-final ships: set `LOAD_SAPIENS=1` in `.env`, then
  `systemctl --user start sapiens-sidecar` and restart taco-backend.
  Verify with a fresh `/v2/video/analyze-motion` call — the tier-2
  block should carry `pose_temporal_stability` or
  `pose_temporal_variance`.

---

## 5. `preference_pairs` is empty after weeks of capture

**Symptom**

```bash
sqlite3 history.db "SELECT COUNT(*) FROM preference_pairs;"
# 0
```

Despite weeks of MV sessions and visible retakes in the UI.

**Check 1 — are there retake rows in `generations`?**

```bash
sqlite3 history.db "
  SELECT COUNT(*) FROM generations WHERE parent_clip_id IS NOT NULL;
"
```

Zero → retake provenance never wrote (rc1 wiring missing or pre-rc1
rows). Non-zero → retakes exist; ETL hasn't run yet.

**Check 2 — `shot_config_key` populated?**

```bash
sqlite3 history.db "
  SELECT COUNT(*) FROM generations WHERE shot_config_key IS NOT NULL;
"
```

Zero → MCP v0.7.0+ shot lineage not flowing through. Check that the
MCP client is actually injecting `shot_config_key` (rc1 contract).

**Check 3 — validator_version match across rows?**

```bash
sqlite3 history.db "
  SELECT validator_version, COUNT(*)
  FROM generations
  WHERE validator_score IS NOT NULL
  GROUP BY validator_version;
"
```

Mixed versions → `construct_preference_pairs.py` filters by
`validator_version`, so mixed-version rows can't be paired.
`validator_version_filter` defaults to current
(`config.VALIDATOR_VERSION`).

**Check 4 — pair construction script ever ran?**

```bash
cat .preference_pairs_watermark 2>/dev/null
ls -la logs/pair_etl.log 2>/dev/null
```

Empty / missing → cron has never executed.

**Fix**

- **Run a dry-run to see per-source counts**:
  ```bash
  uv run --no-sync python scripts/construct_preference_pairs.py --dry-run
  ```
  Expect output broken down by `user_retake`, `composition_kept`,
  `validator_pass`, `validator_fail`. Zero everywhere = capture-side
  data is missing (most likely cause: pre-rc1 rows with NULL
  `parent_clip_id` / `shot_config_key`, or current rows on a stale
  validator_version).

- **Force a full rebuild ignoring the watermark**:
  ```bash
  uv run --no-sync python scripts/construct_preference_pairs.py --full-rebuild
  ```

- **Single-source debug**:
  ```bash
  uv run --no-sync python scripts/construct_preference_pairs.py \
      --source user_retake --dry-run
  ```

- The corpus only fills as new clips with the **current** validator
  version accumulate. At single-operator volume, expect ~6-8 weeks
  before the corpus crosses ~1000 pairs (the first-training-run
  threshold per `docs/PHASE_C_TRAINING_RUNBOOK.md`).

---

## 6. Training run failed mid-flight

**Symptom**

`training_runs` has a row with `lora_output_path IS NULL` long after
the run was supposed to finish:

```bash
sqlite3 history.db "
  SELECT run_id, base_model, num_pairs, trained_at, lora_output_path
  FROM training_runs
  WHERE deployed_at IS NULL
  ORDER BY trained_at DESC
  LIMIT 5;
"
```

**Check 1 — OOM / CUDA error?**

```bash
journalctl --user -u taco-backend --since '2 hours ago' \
  | grep -iE 'oom|cuda|out of memory|train_dpo|sft' | tail -50

# Or, if the script ran outside the unit:
ls -la training_runs/<run_id>/training.log 2>/dev/null
tail -100 training_runs/<run_id>/training.log 2>/dev/null
```

**Check 2 — cuda:0 contention during run**

`train_dpo_sft.py` runs on cuda:0 — same GPU LTX inference uses.
A concurrent video job evicts the trainer's transformer:

```bash
nvidia-smi
```

If LTX is resident, training will OOM or thrash.

**Check 3 — dataset snapshot exists?**

```bash
ls training_runs/<run_id>/dataset.jsonl
```

Missing → script died before snapshot. Pairs were not consumed
(`used_in_training_run_id IS NULL` still); safe to re-run.

**Fix**

- **Pause inference, then re-run training**:
  ```bash
  KEY=$(grep -v '^#' .api_keys | grep -v '^$' | head -1)
  curl -sS -X POST http://localhost:8090/v1/system/pause \
    -H "Authorization: Bearer $KEY"

  # Then re-run with --execute, smaller batch, or different config:
  uv run --no-sync python scripts/train_dpo_sft.py \
      --config configs/sft_quality_lora.yaml --execute

  # When done:
  curl -sS -X POST http://localhost:8090/v1/system/resume \
    -H "Authorization: Bearer $KEY"
  ```

- **Smaller batch / lower rank** if OOM persists: edit
  `configs/sft_quality_lora.yaml` (rank, batch_size).

- **Resume from checkpoint**: `train_dpo_sft.py` writes intermediate
  checkpoints under `training_runs/<run_id>/`. Resume support is
  config-dependent — see `docs/PHASE_C_TRAINING_RUNBOOK.md`.

- **If pairs were consumed but training failed**, manually clear the
  flag so they go back into the candidate pool:
  ```bash
  sqlite3 history.db "
    UPDATE preference_pairs
    SET used_in_training_run_id = NULL
    WHERE used_in_training_run_id = '<run_id>';
  "
  ```

---

## 7. LoRA deployed but quality regressed

**Symptom**

After a new LoRA was deployed (e.g. via setting
`MCP_PRODUCTION_LORA=<id>` and restarting), validator scores dropped
or operator reports visibly worse output.

**Check 1 — confirm regression with metrics**

```bash
sqlite3 history.db "
  SELECT
    DATE(created_at, 'unixepoch') AS day,
    AVG(validator_score) AS mean_score,
    COUNT(*) AS n
  FROM generations
  WHERE validator_score IS NOT NULL
    AND created_at > strftime('%s','now','-14 days')
  GROUP BY day
  ORDER BY day DESC;
"
```

Look for an inflection point at the deploy date.

**Check 2 — A/B comparison if the candidate ran**

```bash
uv run --no-sync python scripts/ab_decision.py \
    --candidate-lora <id>
```

Output includes the paired t-test delta and recommendation
(`promote` / `deprecate` / `no_action` / `insufficient_samples`).

**Fix**

Use the production rollback endpoint (admin-gated):

```bash
ADMIN_KEY=$(grep -v '^#' .admin_keys | grep -v '^$' | head -1)
curl -sS -X POST http://localhost:8090/v1/system/lora/rollback \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"lora_id": "<current_production_lora_id>",
       "reason": "validator score regressed 12% over 3 days"}'
```

What it does:

1. Verifies the `lora_id` matches `MCP_PRODUCTION_LORA` (returns 409
   on mismatch — guards against deprecating non-production candidates).
2. Sets `training_runs.deprecated_at = now()` for that LoRA.
3. Finds the previous deployed-and-not-deprecated LoRA.
4. Atomically rewrites `MCP_PRODUCTION_LORA=<previous>` in `.env`
   (or empty string if no prior production).
5. Returns `{rolled_back_from, rolled_back_to, reason, applied_at, note}`.

**The new value applies on next process restart** — the endpoint does
not hot-swap in-process state. After the rollback response:

```bash
systemctl --user restart taco-backend
```

Investigate the training corpus: re-check `signal_strength`
distribution, verify privacy gate didn't accidentally include
opted-out data, look at sample dataset rows.

---

## 8. BFF tee silently fails (`user_signals` empty)

**Symptom**

`user_signals` table on the BFF is empty despite MCP usage.

```bash
sqlite3 /path/to/bff.db \
  "SELECT COUNT(*) FROM user_signals WHERE signal_type='mcp_event';"
# 0
```

**Check 1 — BFF service running?**

```bash
systemctl --user is-active noodlefinger-bff
curl -sS http://127.0.0.1:8002/health
```

**Check 2 — MCP env points at BFF?**

The MCP subprocess inherits env from its parent (Claude Code). To
verify, find the MCP PID:

```bash
ps aux | grep noodlefinger-mcp
cat /proc/<pid>/environ | tr '\0' '\n' | grep NOODLEFINGER_BFF_URL
```

If unset → tee is a no-op (`_tee_event_fire_and_forget` short-circuits
when `NOODLEFINGER_BFF_URL` is missing).

**Check 3 — BFF reachable from MCP?**

```bash
curl -sS -X POST http://127.0.0.1:8002/api/mcp/events \
  -H 'Content-Type: application/json' \
  -d '{"event":"smoke","data":{},"session_id":"test"}'
```

Expect 200/204.

**Fix**

- **Bring up the BFF**:
  ```bash
  systemctl --user start noodlefinger-bff
  ```
- **Set the env var** in the Claude Code config so the MCP subprocess
  sees it. Example fragment for `~/.claude/mcp.json`:
  ```json
  {
    "mcpServers": {
      "noodlefinger": {
        "command": "...",
        "env": {
          "NOODLEFINGER_BFF_URL": "http://127.0.0.1:8002"
        }
      }
    }
  }
  ```
  Restart Claude Code so the MCP subprocess inherits the new env.
- The tee is **fire-and-forget with a 2 s timeout and swallowed
  errors** by design — events don't queue, they're best-effort. Past
  signals are lost; future ones resume on the next event after the
  BFF is back up.

---

## 9. Schema migration failed at startup

**Symptom**

Backend won't start. journalctl shows an `ALTER TABLE` or
`CREATE TABLE` error, or the unit flips to `failed`.

**Check**

```bash
sqlite3 history.db "PRAGMA user_version;"
```

Compare against `CURRENT_SCHEMA_VERSION` in `history_store.py`
(currently `4` for v1.18.0-rc1+).

```bash
sqlite3 history.db ".tables"
sqlite3 history.db ".schema generations"

# Look for stale WAL locks:
ls -la history.db history.db-wal history.db-shm
fuser history.db history.db-wal 2>/dev/null
```

**Fix**

- **WAL lock from a stale process** (most common):
  ```bash
  systemctl --user stop taco-backend
  fuser -k history.db history.db-wal 2>/dev/null  # kill anyone holding it
  systemctl --user start taco-backend
  ```
- **Migration crashed mid-ladder**: the migrations are idempotent
  (`IF NOT EXISTS`, try/except `OperationalError` on `ALTER`). Just
  restart — `_migrate()` re-runs unconditionally.
- **Backup before any manual surgery**:
  ```bash
  cp history.db history.db.bak.$(date +%s)
  cp history.db-wal history.db-wal.bak.$(date +%s) 2>/dev/null
  ```
- **Last-resort manual ALTER**: if the unit crash logs show a single
  failed `ALTER TABLE`, reproduce it manually inside `sqlite3` while
  the unit is stopped, then start the unit. The `IF NOT EXISTS`
  guards make subsequent boots a no-op.
- Never delete `history.db-wal` — readers depend on it. WAL is the
  record of changes not yet checkpointed; deleting it loses recent
  writes.

---

## 10. Turbo mode won't enable

**Symptom**

```bash
curl -sS -X POST http://localhost:8090/v1/system/turbo \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"enable": true}'
# {"error":"turbo_entry_failed","detail":"cuda:1 not drained..."} or similar
```

Backend logs show `_wait_cuda1_free` timeout, ltx-sidecar `/load`
failure, or a 503 cascade.

**Check 1 — cuda:1 actually drained?**

```bash
nvidia-smi
```

Expect cuda:1 memory usage near 0 MiB after `_stop_cuda1_tenants`.
If something is holding memory, the entry will time out at 20 s and
roll back via `_restore_cuda1_tenants`.

**Check 2 — which cuda:1 sidecars are configured?**

```bash
grep -E '^LOAD_(ACE|JOYAI|ERNIE|MADMOM|SAPIENS)' .env
```

All `LOAD_*=1` units (plus `ltx-sidecar` unconditionally) get
`systemctl stop`d on entry. Any unit that doesn't actually stop
cleanly leaks memory and breaks the drain.

**Check 3 — ltx-sidecar systemd unit healthy?**

```bash
systemctl --user status ltx-sidecar
curl -sS http://127.0.0.1:8093/health 2>/dev/null
```

**Check 4 — turbo log emission (rc4+)**

```bash
journalctl --user -u taco-backend --since '5 minutes ago' \
  | grep -E 'turbo|stop_cuda1|restore_cuda1' | tail -30
```

rc4 added explicit enter/exit log lines naming the gated unit list —
this tells you which units were stopped/restarted.

**Fix**

- **Manually stop cuda:1 tenants and retry**:
  ```bash
  systemctl --user stop ace-step joyai-sidecar ernie-image-sidecar sapiens-sidecar
  nvidia-smi  # confirm cuda:1 is drained

  curl -sS -X POST http://localhost:8090/v1/system/turbo \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"enable": true}'
  ```
- **If a sidecar systemctl-stop hangs** (rare): SIGKILL the pid.
  ```bash
  systemctl --user kill --signal=SIGKILL ace-step
  ```
- **If `_wait_cuda1_free` reports "stuck at <process X>"**: a process
  outside the systemd-managed sidecar set is holding cuda:1. Find with
  `nvidia-smi` and stop it manually.
- **Ltx-sidecar load fails** (cuDNN / CUDA version drift):
  ```bash
  journalctl --user -u ltx-sidecar -n 100
  ```

---

## 11. cuda:0 / cuda:1 OOM

**Symptom**

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```

In journalctl, with a stack trace through `split_model_manager`,
`flux_manager`, or `validator`.

**Check 1 — which model is resident?**

```bash
nvidia-smi
```

LTX active is ~79 GB on cuda:0; Flux Dev forward-pass peak is ~81 GB.
They're mutually exclusive on cuda:0 — combined would be ~160 GB on a
96 GB GPU. The auto-swap helpers (`_ensure_ltx_resident` /
`_ensure_flux_ready`) handle this, but a half-loaded state from a
prior crash can leave both partially resident.

**Check 2 — turbo state**

```bash
curl -sS http://localhost:8090/v1/system/gpu | jq .
```

If `turbo_active=true`, cuda:1 is dedicated to LTX too. ACE / JoyAI /
ERNIE / madmom / Sapiens should NOT be running.

**Check 3 — half-load detection**

```bash
journalctl --user -u taco-backend | grep -iE 'half[_ ]load|reset|evict' | tail -20
```

`_last_load_failed` flag is set on a partial load; the next
`_ensure_ltx_resident` calls `reset()` to fully tear down before
retry. If you see this in a loop, eviction is itself failing.

**Fix**

- **Force-evict the wrong tenant**:
  ```bash
  KEY=$(grep -v '^#' .api_keys | grep -v '^$' | head -1)
  curl -sS -X POST http://localhost:8090/v1/ltx/unload  -H "Authorization: Bearer $KEY"
  curl -sS -X POST http://localhost:8090/v1/flux/unload -H "Authorization: Bearer $KEY"
  ```
  Then make a request — auto-swap reloads only what's needed.

- **Hard restart taco-backend** if eviction itself OOMs:
  ```bash
  systemctl --user restart taco-backend
  ```
  All workers tear down, cuda:0 fully clears, fresh boot reloads
  models cleanly.

- **Validator (RAFT) OOM on cuda:0**: usually means LTX is
  mid-forward-pass when the validator dispatch fires. RAFT uses
  ~1 GB transient — should be safe — but if the LTX transformer is
  swapping right at that moment, the peak collides. Lower validator
  concurrency by spacing dispatches (currently fire-and-forget; no
  knob for this in rc5).

- **Turbo + remote pool collapse**: scale the remote pool to 0
  before exiting turbo if cuda:1 is leaking:
  ```bash
  curl -sS -X POST http://localhost:8090/v1/system/pool/remote-workers \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"modal": 0, "runpod": 0}'
  ```

---

## 12. `find_similar_shots` returns empty

**Symptom**

MCP `find_similar_shots` tool returns `[]` for a valid prompt; or
direct `POST /v2/embeddings/search` returns
`{"results": []}`.

**Check 1 — corpus size**

```bash
sqlite3 history.db "SELECT COUNT(*) FROM clip_embeddings;"
```

Need ~100+ rows minimum for retrieval to surface anything useful.
If 0, embeddings backfill never ran.

**Check 2 — bearer privacy gate**

`/v2/embeddings/search` filters by `api_key_hash` of the caller.
You only see your own clips, ever.

```bash
sqlite3 history.db "
  SELECT api_key_hash, COUNT(*) AS n
  FROM generations
  WHERE id IN (SELECT id FROM clip_embeddings)
  GROUP BY api_key_hash;
"
```

If your bearer's hash isn't here, no embeddings were captured under
this key (privacy gate working as intended).

**Check 3 — validator_version filter**

The endpoint defaults `validator_version_filter` to current
`config.VALIDATOR_VERSION` (currently `1.17.0-rc5`). Rows scored on
older versions are excluded.

```bash
sqlite3 history.db "
  SELECT validator_version, COUNT(*)
  FROM generations
  WHERE id IN (SELECT id FROM clip_embeddings)
    AND validator_score IS NOT NULL
  GROUP BY validator_version;
"
```

**Check 4 — `min_validator_score` too aggressive**

Default request body includes `min_validator_score`; if the caller
set this to e.g. `0.9`, very few clips qualify. Try `0.0`:

```bash
KEY=$(grep -v '^#' .api_keys | grep -v '^$' | head -1)
curl -sS -X POST http://localhost:8090/v2/embeddings/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "...", "k": 5, "min_validator_score": 0.0,
       "validator_version_filter": null}' | jq .
```

`validator_version_filter: null` removes the filter entirely.

**Fix**

- **Empty corpus**: run the backfill.
  ```bash
  uv run --no-sync python scripts/backfill_prompt_embeddings.py --dry-run
  uv run --no-sync python scripts/backfill_prompt_embeddings.py
  ```
- **Privacy-gate empty for this bearer**: the bearer hasn't generated
  enough opted-in clips. Check `api_key_metadata.training_opt_in`.
- **Version mismatch**: pass `validator_version_filter: null` in the
  request body, or wait for the corpus to roll forward to the current
  version. Bulk-revalidate to force re-scoring (admin endpoint
  `/v2/system/bulk-revalidate`, see #3).
- **Lower `min_validator_score`** to broaden the candidate set.

---

# FAQ

### Do I need a GPU on this box for the embeddings backfill?

No. Embeddings are computed by **llama-swap** on the chat host
(192.168.1.80:8080). taco-backend just packs the float32 bytes and
INSERTs into `clip_embeddings`. The backfill script does ~30
prompts/sec via batch-of-64 calls to llama-swap; ~4 minutes for 8000
rows. Rate-limit it with `--sleep-ms 1000` if you want to leave
bandwidth for live MCP/chat traffic.

### Can I run two MVs concurrently?

Yes, with caveats:

- **Same bearer**: gated by `PER_KEY_QUEUE_CAP` (default 100). A
  200-clip MV with mcp v0.4.4 parallel dispatch fits comfortably.
  Two simultaneous 28-clip MVs from the same bearer = 56 in-flight,
  also fine.
- **Different bearers**: gated by `MAX_QUEUE_DEPTH` (default 200) and
  worker capacity. Peak parallelism with turbo + 10 Modal + 2 RunPod
  is **14 concurrent video workers**. Beyond 14, jobs queue.
- **GPU-bound**: with turbo off, only 1 local worker. Two MVs mostly
  serialize on the LTX transformer.
- See `docs/operator-tuning.md` for cap-tuning examples.

### How do I add a new bearer?

```bash
# 1. Generate a key (any random string, 32+ bytes recommended).
NEWKEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# 2. Append to .api_keys (one per line, # for comments).
echo "$NEWKEY  # <description>" >> .api_keys

# 3. Decide opt-in vs opt-out for training.
HASH=$(python3 -c "import hashlib; print(hashlib.sha256('$NEWKEY'.encode()).hexdigest())")
sqlite3 history.db "
  INSERT OR REPLACE INTO api_key_metadata
    (api_key_hash, training_opt_in, tier, created_at, updated_at)
  VALUES ('$HASH', 1, 'pro', unixepoch(), unixepoch());
"

# 4. Restart taco-backend so .api_keys is re-read.
systemctl --user restart taco-backend
```

`.api_keys` is read on startup. The seed of `api_key_metadata` from
`.api_keys` only ran during the v2→v3 migration (one-shot); new
bearers added later need the manual INSERT above to opt them in.

### How do I rotate a bearer key?

Add the new key as above; remove the old line from `.api_keys`;
restart. Existing in-flight jobs under the old key will fail to poll
results — coordinate the rotation with the client.

### What's the difference between turbo and DUAL_GPU_LTX?

- **Turbo** is a runtime toggle (`POST /v1/system/turbo`). It
  evicts cuda:1 tenants, claims cuda:1 for a second LTX worker, and
  restores them on exit.
- **DUAL_GPU_LTX=1** is a boot-time env flag that permanently
  dedicates both GPUs to LTX from startup. Flux / ACE / JoyAI / ERNIE
  / madmom / Sapiens are all disabled. Restart required to change.

Use turbo when you want temporary 2x video throughput. Use
DUAL_GPU_LTX when this box is dedicated to video generation only.

### How long does the validator add to a job's tail?

Fire-and-forget. The job is marked `COMPLETED` and the result is
returned to the caller **before** the validator runs. The dispatch
happens in `_on_job_complete` after the per-key counter decrement,
inside `asyncio.create_task` — the queue worker dequeues the next job
immediately. Validator latency (~150 ms tier-1 + ~2-4 s tier-3) lands
on the row asynchronously and `/v2/history/{id}` reads it back when
ready. The MCP active hook (`_run_quality_validation`) hits the
`validator_runs` cache instantly when it polls ~50 ms later.

### How do I check what a specific bearer has been doing?

```bash
HASH=<sha256_of_api_key>
sqlite3 history.db "
  SELECT
    DATE(created_at, 'unixepoch') AS day,
    model,
    COUNT(*) AS n,
    AVG(validator_score) AS mean_score
  FROM generations
  WHERE api_key_hash = '$HASH'
    AND created_at > strftime('%s','now','-30 days')
  GROUP BY day, model
  ORDER BY day DESC;
"
```

### Where do logs live?

- **Backend**: `journalctl --user -u taco-backend`. Add `-f` to tail.
- **Sidecars**: `journalctl --user -u <unit>` for ace-step,
  joyai-sidecar, ernie-image-sidecar, ltx-sidecar, madmom-sidecar,
  sapiens-sidecar.
- **Pair ETL cron**: `logs/pair_etl.log` (see runbook).
- **Training runs**: `training_runs/<run_id>/training.log`.
- **A/B decision cron**: configured in the cron entry; default
  `logs/ab_decision.log`.

### How do I escalate?

If you can't resolve from this guide:

1. Capture state: `journalctl --user -u taco-backend --since '1 hour ago' > /tmp/backend.log`
2. Capture GPU state: `nvidia-smi > /tmp/gpu.log`
3. Capture queue / metrics: `curl ... /v1/system/metrics > /tmp/metrics.json`
4. File an issue with the three artifacts attached + a clear symptom
   description.

For deeper architecture context:

- `docs/CAPTURE_VALIDATOR.md` — validator pipeline internals.
- `docs/PHASE_C_TRAINING_RUNBOOK.md` — training cron, A/B, rollback.
- `docs/operator-tuning.md` — caps, NOFILE, ffmpeg autodetect.
- `CLAUDE.md` — codebase + GPU topology + version-by-version
  highlights.
- `CHANGELOG.md` — bug taxonomy by version.
