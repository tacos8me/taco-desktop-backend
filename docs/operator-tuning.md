# Operator tuning — rate limits, concurrency, fd ceilings, export quality

> Server version: v1.18.0-rc3 (2026-04-29).
> Audience: backend operator / on-call. Frontend / SDK callers should look at [API.md](./API.md) instead.

This file documents the runtime-tunable knobs originally introduced in **v1.16.1** (after a 28 a2v / 1.5 s-pacing submission tripped `per_key_queue_full` on 24/28 first-pass submissions) and rescaled in **v1.16.4** for heavy-MV operators running 200-clip `cut_music_video` sessions with mcp v0.4.4+ parallel clip dispatch.

## What's tuned

Three layers, all env-overridable:

1. **Application layer** — per-bearer + global queue caps in `config.py`.
2. **HTTP layer** — uvicorn `--limit-concurrency` + `--backlog` flags in `run.sh`.
3. **System layer** — `LimitNOFILE` in the systemd user unit.

To roll back, set the env vars in `.env` and restart the unit.

## Per-key job queue caps (application layer)

`config.py` exposes five ceilings. Each accepts an env-var override of the same name; missing or unparsable values fall back to the default.

| Knob | v1.15.x | v1.16.1 | v1.16.4 default | What it caps |
|---|---|---|---|---|
| `PER_KEY_QUEUE_CAP` | 3 | 15 | **100** | In-flight v2 video jobs per bearer (text-to-video, image-to-video, audio-to-video, retake, video-outpaint, video-hdr). Returns `429 per_key_queue_full` + `Retry-After: 30` when exceeded. |
| `PER_KEY_MUSIC_CAP` | 2 | 5 | **20** | In-flight music submissions per bearer (`/v2/music`). |
| `PER_KEY_BATCH_CAP` | 2 | 5 | **20** | In-flight batch submissions per bearer (`/v2/batch`). |
| `MAX_QUEUE_DEPTH` | 10 | 30 | **200** | Global queue ceiling across all bearers. Returns `429 queue_full`. v1.16.4 made this env-overridable (was hardcoded). |
| `MAX_BATCH_QUEUE_DEPTH` | 5 | 5 | **30** | Concurrent batch submissions (`/v2/batch`). v1.16.4 made this env-overridable (was hardcoded). |

The half-global ratio (per-key 100 vs global 200) preserves the v1.16.1 single-tenant-protection rationale: a single bearer can saturate up to half the queue, but never starve other tenants. A 200-clip `cut_music_video` session with parallel dispatch fits comfortably under 100 in-flight per-bearer.

`PER_KEY_QUEUE_CAP` was the proximate cause of the v1.16.0 regression — 28 sequential video submissions from one bearer overflowed a 3-slot per-key cap. v1.16.1 fixed that for typical 28-job MVs; v1.16.4 scales to 200-clip sessions without forcing operators to hand-tune env vars.

### Override pattern

In `.env` (or any env source loaded by `run.sh`):

```
PER_KEY_QUEUE_CAP=50
PER_KEY_MUSIC_CAP=10
PER_KEY_BATCH_CAP=10
MAX_QUEUE_DEPTH=100
MAX_BATCH_QUEUE_DEPTH=15
```

Then `systemctl --user restart taco-backend`. No code edit required.

## HTTP layer (uvicorn)

`run.sh` invokes uvicorn with explicit flags as of v1.16.1:

```
exec uv run --no-sync uvicorn server:app --host 0.0.0.0 --port 8090 --no-access-log \
    --limit-concurrency 200 --backlog 4096
```

| Flag | Why |
|---|---|
| `--limit-concurrency 200` | Caps in-flight ASGI request handlers. Default uvicorn uses `limit_concurrency=None` (unbounded) — under 28+ concurrent client polls per video job, the unbounded path lets the kernel SYN backlog overflow before ASGI ever sees the request, and clients get `Connection reset by peer` instead of a clean `503`. With the cap, requests beyond 200 in-flight return clean 503s with the standard envelope. |
| `--backlog 4096` | Doubles the kernel TCP listen queue from the uvicorn default of 2048. Shipping with the higher value gives more headroom for short bursts of poll-heavy clients (the same 28-job MV submission with 240 GET polls per job is ~6,720 polls in flight at peak). |

Tunable: bump `--limit-concurrency` higher if your operator workload includes many slow concurrent SSE streams (each holds a handler open). Lower it if you want more aggressive backpressure.

## System layer (systemd)

The taco-backend user unit at `~/.config/systemd/user/taco-backend.service` carries:

```
[Service]
LimitNOFILE=16384
```

Default systemd inherits the user-default `nofile=1024`, which was *exactly* the pre-v1.16.1 ceiling. A single in-flight MV submission opens, at peak:

- ~30 concurrent client SSE streams + result-poll connections,
- ~10 outbound httpx connections to the LTX local + Modal + RunPod sidecars,
- ~3 outbound httpx connections to ACE / JoyAI or ERNIE / madmom sidecars,
- 1 SQLite-WAL writer + N reader fds for `history.db`,
- ~50 transient fds for upload-store reads, thumbnail extraction, and ffmpeg pipes.

Crossed at ~150 fds in steady state during a busy 14-worker turbo session. 16,384 leaves plenty of headroom for accidental fd leaks; the system-wide `nofile-max` is ~1M on a stock Ubuntu 24.04 box so this is not load-bearing on system limits.

## Reversibility

To run more conservatively in a constrained environment (a small VPS, a noisy multi-tenant box):

1. Lower the per-key caps in `.env`:
   ```
   PER_KEY_QUEUE_CAP=15
   PER_KEY_MUSIC_CAP=5
   PER_KEY_BATCH_CAP=5
   MAX_QUEUE_DEPTH=30
   MAX_BATCH_QUEUE_DEPTH=5
   ```
2. Optionally lower the uvicorn limits in `run.sh` (edit `--limit-concurrency` and `--backlog`).
3. Optionally lower `LimitNOFILE` in the systemd unit and `systemctl --user daemon-reload && systemctl --user restart taco-backend`.

Caps are read at process start; restart is required after any of the above.

## Validation — confirm the caps are live

After a config change + restart, verify the live values from outside the running process:

```bash
# (1) Per-key + global caps are loaded into config:
python3 -c "import config; print(config.PER_KEY_QUEUE_CAP, config.MAX_QUEUE_DEPTH)"
# Expect: 15 30   (or your override)

# (2) NOFILE limit is in effect on the live process:
cat /proc/$(systemctl --user show -p MainPID --value taco-backend)/limits | grep "Max open files"
# Expect: Max open files            16384                16384                files
```

If `(1)` reports the v1.15.x defaults `3 10`, the unit was likely started without your `.env` — confirm the file path and that systemd is sourcing it (`EnvironmentFile=` in the `[Service]` block).

If `(2)` reports `1024`, the unit didn't pick up `LimitNOFILE` — `systemctl --user daemon-reload` was probably skipped after the unit edit.

## Cross-references

- Application-layer 429 envelopes are documented in `docs/API.md` → "Queue / rate limit" table.
- The v1.16.0 `/v1/music/analyze` endpoint (which `madmom_client` proxies) is documented in `docs/API.md`.
- The `cut_music_video` MCP orchestrator (which is the most common driver of the 28-job submission pattern) lives in noodlefinger-mcp v0.4.2+; per-shot audio slicing is described in its `flows.json`.

---

## Export quality (v1.16.2)

`POST /v2/compositions/{comp_id}/export` ships a libx264-based default encoder stack as of v1.16.2. Pre-v1.16.2 the export pipeline hardcoded `-c:v libopenh264` with NO CRF / bitrate flags — libopenh264 defaults to ~4-8 Mbps which produced visible blocking on 1080p+ output.

The fix switches the default to:

| Knob | v1.16.2 default |
|---|---|
| Video encoder | `libx264` |
| CRF | `18` (libx264) / `22` (libx265) |
| Preset | `medium` |
| Profile | `high` |
| Pixel format | `yuv420p` (max compat) |
| Audio bitrate | `256k` (was 192k) |

CRF 18 + high profile is the standard "visually transparent" operating point for H.264 — perceptually lossless at typical viewing distances. `yuv420p` maximizes player compatibility (mobile / iOS QuickTime / Chrome all decode it natively).

### Per-export overrides

There are NO env vars for export quality — every knob is per-request. Full schema in `docs/API.md` → `POST /v2/compositions/{comp_id}/export`:

- `output_encoder` ∈ `{libx264, libx265, libopenh264}`
- `output_crf` ∈ `[0, 51]` (lower = higher quality)
- `output_preset` ∈ x264 preset names (`ultrafast` … `placebo`)
- `output_profile` ∈ `{baseline, main, high, high10, high422, high444}`
- `output_video_bitrate` (string `\d+[kMG]?`, e.g. `"12M"`)
- `output_audio_bitrate` (string `\d+[kMG]?`, e.g. `"320k"`)

Setting `output_video_bitrate` on libx264/libx265 switches the encoder from CRF mode to 1-pass ABR with `-maxrate` and `-bufsize 24M` — useful when you have a hard size budget and don't care about per-clip quality variance.

Malformed values return `422` BEFORE the job is enqueued.

### `TACO_FFMPEG_BIN` env var (only operator knob)

This is the ONLY operator-side knob for export quality. Sets the ffmpeg binary used for export. Defaults to autodetect.

The autodetect order is:

1. `$TACO_FFMPEG_BIN` if set.
2. `ffmpeg` from `$PATH` (whatever venv / conda is active).
3. `/usr/bin/ffmpeg` (Ubuntu system ffmpeg).

The exporter probes each candidate with `-encoders` and picks the first one that supports the requested encoder. **Note**: conda-installed ffmpeg in this box's miniconda env was built `--disable-gpl` and ships WITHOUT libx264/libx265 — only libopenh264. The Ubuntu system ffmpeg at `/usr/bin/ffmpeg` has libx264 from the `libx264-*` package, so autodetect picks it for the libx264 path even though it's not first in `$PATH`.

If neither has libx264/libx265, the exporter logs a WARN and falls back to libopenh264 (lower quality at default bitrate). To install:

```bash
sudo apt install libx264-164  # or libx265-199
# Both are dependencies of Ubuntu's `ffmpeg` meta-package.
```

### Validation

After deployment, sanity-check the live encoder via a small export:

```bash
COMP_ID=...
KEY=$(grep -v '^#' .api_keys | grep -v '^$' | head -1)

# Default export.
JOB=$(curl -sS -X POST "http://localhost:8090/v2/compositions/$COMP_ID/export" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{}' | jq -r .id)

# Poll until done, then download:
curl -sS -H "Authorization: Bearer $KEY" \
  "http://localhost:8090/v2/jobs/$JOB/result" -o /tmp/result.mp4

# Confirm libx264 was used and the bitrate is higher than v1.16.1:
ffprobe -v error -show_entries stream=codec_name,bit_rate /tmp/result.mp4
```

## Embeddings + sqlite-vec (v1.18.0-rc2)

Phase B retrieval depends on two pieces: the `sqlite-vec` SQLite
extension (vector storage + cosine/L2 search) and llama-swap's
`/v1/embeddings` endpoint (Gemma 3 12B → 3584-dim float32 vectors).

### Install

`sqlite-vec` is a regular pip dep — no system packages:

```bash
cd /mnt/nvme-1/servers/taco-backend
uv pip install sqlite-vec   # already in pyproject.toml since rc2
```

The Python package ships the `.so` and a loader. Verify:

```bash
uv run --no-sync python -c "
import sqlite3, sqlite_vec
c = sqlite3.connect(':memory:')
c.enable_load_extension(True); sqlite_vec.load(c)
print(c.execute('SELECT vec_version()').fetchone())
"
# → ('v0.1.9',)
```

After taco-backend restarts the boot log carries one of:

```
INFO history_store: sqlite-vec extension loaded
WARN history_store: sqlite-vec extension load failed (...); embedding search endpoints will return 503
```

### llama-swap `/v1/embeddings`

llama.cpp natively supports `/v1/embeddings` when the Gemma model is
loaded. Verify:

```bash
curl -sS -X POST http://192.168.1.80:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "gemma-3-12b-nvfp4", "input": "test"}' | head -c 200
```

A healthy response is JSON shaped like
`{"data":[{"embedding":[...3584 floats...],"index":0}],"model":...}`.

If you see HTTP 502 / `unable to start process: upstream command exited
prematurely`, llama-swap's config is missing the embeddings binding.
Add to llama-swap's model config:

```yaml
# In ~/llama-swap/config.yaml under the gemma model:
gemma-3-12b-nvfp4:
  cmd: |
    /path/to/llama-server
    --model /path/to/gemma-3-12b-Q5_K_M.gguf
    --embeddings           # ← critical: enables /v1/embeddings
    --pooling mean
    -ngl 99
    --port 8080
```

`--embeddings` adds the route; `--pooling mean` matches sentence-
embedding semantics (default for retrieval). Restart llama-swap.

### Backfill historical rows

Once both pieces are working, backfill prompt embeddings for opted-in
rows:

```bash
cd /mnt/nvme-1/servers/taco-backend
uv run --no-sync python scripts/backfill_prompt_embeddings.py --dry-run
# Inspect the count, then:
uv run --no-sync python scripts/backfill_prompt_embeddings.py
```

The script is idempotent and resumable — if it crashes, just re-run.
Runs at ~30 prompts/sec via batch-of-64 calls; ~4 minutes for 8000
rows. Add `--sleep-ms 1000` if you want to leave bandwidth for live
traffic.

### Verification

Issue a search to confirm the surface works:

```bash
KEY=$(grep -v '^#' .api_keys | grep -v '^$' | head -1)
curl -sS -X POST http://localhost:8090/v2/embeddings/search \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a dramatic sunset over the ocean", "k": 5}' | jq .
```

Healthy response shape:

```json
{
  "validator_version_filter": "1.17.0-rc5",
  "results": [
    {"shot_id": "...", "prompt": "...", "similarity_score": 0.83,
     "validator_score": 0.91, "final_score": 0.78, ...}
  ]
}
```

### Rate limit (429s)

`/v2/embeddings/*` and `/v2/system/bulk-revalidate` are gated at 10
req/sec/key with burst 10. Bursts of 11+ in the same instant return
429 + `Retry-After`. Tune burst/rate via the constants
`_EMBEDDINGS_RATE_LIMIT_*` in `server.py` if your workload demands
more — current values are a single-operator sanity floor, not a
multi-tenant fairness ceiling.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| 503 "embedding search not available — install sqlite-vec extension" | extension not loaded | install sqlite-vec; check boot logs |
| 503 "embedding service unavailable" on /v2/embeddings/search | llama-swap `/v1/embeddings` returning 5xx or unreachable | verify llama-swap config has `--embeddings`; restart llama-swap |
| 429 on every request | rate limit too tight for workload | tune `_EMBEDDINGS_RATE_LIMIT_*` |
| Empty `results: []` despite known clips | privacy gate filter, OR opted-out bearer, OR validator_version filter mismatch | check `api_key_metadata.training_opt_in`; verify `validator_version_filter` matches actual rows |
| `total_samples: 0` from /recommend-loras | no clips with `lora_applied_id` populated yet | submit a few jobs with `{"lora": {...}}` after rc2 deploy |

---

## Validator pipeline (v1.17.0-rc2+)

Three-tier validator (RAFT motion + Sapiens pose + Gemma judge) runs
fire-and-forget on every completed video job for opted-in bearers.
Composite score lands in `generations.validator_score`; full payload in
`generations.validator_payload_json`.

### Env vars

All registered in `config.py`. Restart `taco-backend` after changes.

| Env var | Default | What it controls |
|---|---|---|
| `LOAD_SAPIENS` | `0` | Tier-2 sidecar gate. `0` = stub mode (composite uses `0.2·1.0` placeholder, never blocks). `1` = real Sapiens inference; requires `sapiens-sidecar` service running on cuda:1:8096. |
| `SAPIENS_SIDECAR_URL` | `http://127.0.0.1:8096` | Tier-2 sidecar base URL. Override only for non-loopback deployments. |
| `SAPIENS_TIMEOUT_S` | `60` | Per-request timeout in seconds. On timeout/5xx/ConnectError, tier2 returns `None` and the composite drops the tier2 weight (graceful degrade — never blocks the validator). |
| `VALIDATOR_VERSION` | `1.17.0-rc5` | String used as `validator_runs(video_sha256, validator_version)` cache key. Bumping forces re-runs of all clips. Treat as immutable per release. |
| `VALIDATOR_ARTIFACTS_DIR` | `<repo>/validator_artifacts/` | On-disk directory where tier dumps land for debugging. Safe to `rm -rf` between runs. |
| `JUDGE_PROMPT_V1` | (long default in `config.py`) | Tier-3 system prompt for the Gemma judge. Override only when tuning the judge — schema-validated against `JudgeResponseV1` so a malformed prompt that breaks JSON output will surface as `tier3` fallback `verdict=warn / score=0.5`. |
| `NOODLEFINGER_BEAT_MADMOM_MIN` | `0.55` | Minimum madmom downbeat-confidence score to use madmom output; below this the orchestrator falls back to librosa. |
| `NOODLEFINGER_BEAT_LIBROSA_MIN` | `0.40` | Minimum librosa beat-confidence floor before the analyzer returns a degraded-quality warning. |

### Enabling the Sapiens sidecar (tier 2)

Tier 2 ships in stub mode for the rc2 → rc5 window. Flip on after diff
review:

```bash
# 1. Add to .env (or env source loaded by run.sh):
echo "LOAD_SAPIENS=1" >> .env

# 2. Enable + start the sidecar unit:
systemctl --user enable --now sapiens-sidecar

# 3. Restart taco-backend so config picks up LOAD_SAPIENS:
systemctl --user restart taco-backend

# 4. Confirm the sidecar is alive:
curl -sS http://127.0.0.1:8096/health
# → {"status":"ok",...}
```

Turbo-mode coordination is automatic: `sapiens-sidecar` is in
`_stop_cuda1_tenants` + `_restore_cuda1_tenants` (gated on
`LOAD_SAPIENS`), so turbo entry will stop it and exit will restart it
just like ACE / JoyAI / ERNIE.

### Verification

After dispatching a video job, the validator runs as a
fire-and-forget task. Confirm scores are landing:

```bash
sqlite3 history.db \
  "SELECT id, validator_score, validator_version
   FROM generations
   WHERE validator_score IS NOT NULL
   ORDER BY created_at DESC LIMIT 5"
```

For a healthy pipeline, `validator_score` should populate within
~1-3 s of the video MP4 hitting disk. NULL across recent rows means
the dispatch is being skipped — check the metrics counter:

```bash
KEY=$(grep -v '^#' .api_keys | grep -v '^$' | head -1)
curl -sS -H "Authorization: Bearer $KEY" \
  http://localhost:8090/v1/system/metrics | jq .validator_dispatch
```

Counter shape: `{success, failure, skipped_not_video,
skipped_opt_out, skipped_validator_disabled, failure_rate_pct}`.
High `skipped_opt_out` means the bearer's `api_key_metadata.training_opt_in`
is 0 (validator dispatch is gated on training opt-in). High
`failure` with non-trivial `failure_rate_pct` means the
validator itself is erroring — `journalctl --user -u taco-backend |
grep validator` for the WARN line.

### Synchronous one-off — `POST /v2/video/analyze-motion`

Use to validate a single clip without going through the queue.
Reuses `validator.run_all_tiers()` directly. Body:
`{video_uri, prompt?, shot_uuid?, tiers?, validator_version?}`.
Useful for ad-hoc diagnosis without polluting `validator_runs`
cache (synchronous endpoint still writes the cache row, keyed by
the SHA-256 of the resolved file).

---

## Phase C training setup (v1.18.0-rc3+)

Phase C ships training infrastructure (preference-pair ETL +
SFT-on-chosen LoRA training + A/B promotion harness). The first
real training run waits until the preference-pair corpus crosses
~1000 high-strength pairs (~6-8 weeks of organic traffic). All
scripts default to dry-run / opt-in semantics; nothing trains
without explicit operator action.

### Pre-flight checklist

Before kicking off the first training run:

1. **Corpus**: `sqlite3 history.db "SELECT COUNT(*) FROM preference_pairs WHERE signal_strength >= 0.5 AND validator_version = '<your-version>'"` ≥ 1000. (default trainer floor; raise to 0.7 if you want to require validator_pass-or-better signal only)
2. **Window**: schedule the run during off-hours — SFT uses the
   full LTX 22B base for ~50-60 GPU-hours on a single Blackwell.
   Throughout the run cuda:0 is unavailable for inference; turbo +
   cuda:1 sidecar can keep video flowing on a separate GPU but
   throughput is halved.
3. **Backup**: `cp history.db history.db.pre-train-$(date +%F)`.
   Training itself never writes to `generations` rows, but the
   `INSERT` into `training_runs` and `UPDATE preference_pairs SET
   used_in_training_run_id` are post-hoc auditable; keep a
   pristine snapshot anyway.
4. **Validator version freeze**: confirm `VALIDATOR_VERSION` has
   not been bumped mid-corpus — pairs across mixed validator
   versions are filtered out by `construct_preference_pairs.py`,
   but a recent bump can silently halve the eligible corpus.

### Weekly preference-pair construction

Cron candidate (run weekly, e.g. Sundays 03:00):

```bash
cd /mnt/nvme-1/servers/taco-backend
uv run --no-sync python scripts/construct_preference_pairs.py --since-watermark
```

`--since-watermark` (default) is incremental — reads
`.preference_pairs_watermark`, includes only `generations` rows
created after, then writes `time.time()` back on success. Use
`--full-rebuild` once after the rc3 schema migration to backfill
historical rows; subsequent runs should be incremental. `--dry-run`
reports counts without writing.

The script is idempotent (`INSERT OR IGNORE` on
`idx_pp_unique_pair_source`) — re-running is safe.

### First training run

Defense-in-depth: defaults to `dry_run=True`. `--execute` is the
consent gate.

```bash
# (1) Dry run — no GPU, no DB writes; prints expected dataset size:
uv run --no-sync python scripts/train_dpo_sft.py \
  --config configs/sft_quality_lora.yaml

# (2) Real run — claims cuda:0 for ~50-60 hours:
uv run --no-sync python scripts/train_dpo_sft.py \
  --config configs/sft_quality_lora.yaml --execute

# (3) Resume from checkpoint after interruption:
uv run --no-sync python scripts/train_dpo_sft.py \
  --config configs/sft_quality_lora.yaml --execute \
  --resume-from training_runs/sft-quality-v0.0.1/checkpoint-2/
```

Output artifact lands in `loras/<run_id>/` and a row is appended to
`training_runs` with full reproducibility metadata
(seed/hyperparams/dataset snapshot/code SHA/validator_version).
The trained LoRA is registered as a candidate — **NOT** auto-deployed.
A/B promotion is the next gate.

### A/B promotion harness

Set in `.env` and restart taco-backend + mcp:

```
AB_TEST_ACTIVE=1
AB_CANDIDATE_LORA=sft-quality-v0.0.1
AB_AUTO_PROMOTE=1     # set 0 for operator-gated decision-only mode
```

When active, the mcp orchestrator's `_normalize_input` tags each
MV session with `_ab_arm` (baseline vs candidate) at coin-flip
50/50. Weekly cron runs the decision script:

```bash
AB_CANDIDATE_LORA=sft-quality-v0.0.1 \
  uv run --no-sync python scripts/ab_decision.py
```

Promotion thresholds (paired t-test on per-MV mean
`validator_score`):

- **PROMOTE** iff `delta >= +10%` AND `p < 0.05`.
- **DEPRECATE** iff `delta <= -5%` AND `p < 0.05`.
- Otherwise: NO ACTION.

Insufficient-samples rule: ≥ 30 MVs per arm before any decision.
With `AB_AUTO_PROMOTE=0` the decision is logged and a row is
written but `MCP_PRODUCTION_LORA` is **not** rotated — the
operator runs the rollback / promote endpoint manually.

---

## Rollback procedures

### LoRA rollback

`POST /v1/system/lora/rollback` deprecates the current production
LoRA and rolls `MCP_PRODUCTION_LORA` back to the previous
deployed-and-not-deprecated training run. Admin-keyed (per
`.admin_keys` / `TACO_ADMIN_KEY`).

```bash
ADMIN_KEY=$(cat .admin_keys | grep -v '^#' | head -1)
CURRENT=$(grep MCP_PRODUCTION_LORA .env | cut -d= -f2)

curl -sS -X POST http://localhost:8090/v1/system/lora/rollback \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"lora_id\": \"$CURRENT\", \"reason\": \"validator_score regression in arm A vs baseline\"}"
```

Response shape: `{rolled_back_from, rolled_back_to, reason,
applied_at, note}`. The `lora_id` field MUST match the current
production value — guards against accidentally deprecating a
non-production candidate. Mismatch returns `409`.

The new value is written to `.env`. **Restart `taco-backend` and
`mcp` to fully apply** — the endpoint does not hot-swap in-process
state; the next cold start re-reads `.env`.

If no prior good production LoRA exists, `rolled_back_to` is the
empty string and `MCP_PRODUCTION_LORA` is cleared (mcp falls back
to no-LoRA generation).

### Schema rollback

**Don't.** All v1.17.0-rc1+ migrations are additive — new
nullable columns + new tables, no destructive ALTERs. Pre-migration
rows get NULL in new columns and continue to read fine after
upgrade. To "roll back" you would either:

- Run an older taco-backend binary against the upgraded DB —
  works as long as the binary doesn't reject unknown
  `PRAGMA user_version`. v1.16.x readers tolerate v3 / v4
  schemas because all reads use named columns.
- Or restore from a pre-upgrade `history.db` backup if you took
  one. Forward-only is the supported path.

### Validator version downgrade

`config.VALIDATOR_VERSION` is a free-form string; older scores
remain valid because `validator_runs` is keyed by
`(video_sha256, validator_version)` UNIQUE — downgrading just
means new dispatches will look up a different cache row.

```bash
# In config.py, revert the constant:
# VALIDATOR_VERSION = "1.17.0-rc5"  →  VALIDATOR_VERSION = "1.17.0-rc4"
systemctl --user restart taco-backend
```

Old `generations.validator_score` rows scored under rc5 are
preserved — no backfill, no deletion. New dispatches re-score
under rc4 and land in fresh `validator_runs` rows. Phase C
training scripts filter pairs by `validator_version` so a
downgrade does not poison the training corpus; it just shrinks
the eligible window until enough rc4 scores accumulate.
