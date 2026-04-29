# Operator Quickstart

> Audience: a new operator who just got systemd access to the box and wants to
> verify the system is healthy, generate a test clip, find the dashboard, and
> know who to call when things break.
>
> Server version: v1.18.0-rc3 (2026-04-29).

This is the "first 30 minutes on the box" doc. Frontend / SDK callers want
[QUICKSTART.md](./QUICKSTART.md). Operators tuning rate limits or export
quality want [operator-tuning.md](./operator-tuning.md). Architecture nerds
want [ARCHITECTURE.md](./ARCHITECTURE.md). When you want to talk to clients,
[API.md](./API.md) is canonical.

---

## 0. What you're looking at

`taco-backend` is the FastAPI inference server that fronts our LTX video stack
and Flux image stack. It runs on a single Linux box with two RTX PRO 6000
Blackwell GPUs (96 GB each) and proxies a constellation of sidecars (ACE for
music, JoyAI / ERNIE for image edit, madmom for downbeat detection, sapiens
for pose stability, plus a remote LTX worker pool on Modal + RunPod). The
frontend portal (`noodle-i`, `noodle-v`, `noodle-mv`) reaches it via
`api.noodlefinger.io` (Cloudflare-fronted) or `http://<host>:8090` on the LAN.

For the full topology — GPU tenants, sidecar swap rules, turbo mode, validator
pipeline — see [ARCHITECTURE.md](./ARCHITECTURE.md). This file only covers
"how do I drive it day-to-day".

---

## 1. The 5 services

Everything runs as a **systemd user unit** under the `ian` account. All
control commands take the form
`systemctl --user <verb> <unit>`. Status is `systemctl --user status <unit>`,
logs are `journalctl --user -u <unit> -f`.

| Unit | Purpose | Port | GPU | Required? |
|---|---|---|---|---|
| `taco-backend.service` | The main FastAPI server. All `/v1/*` and `/v2/*` HTTP endpoints. | `8090` | cuda:0 (LTX/Flux swap) | Yes — this is the system. |
| `noodlefinger-bff.service` | Backend-for-frontend for the operator portal (`noodlefinger-portal`). Owns `portal.db`. | `8002` (loopback) | none | Only if the operator portal is in use. |
| `sapiens-sidecar.service` | Validator tier-2 (pose temporal stability). Stub-mode in rc2. | `8096` | cuda:1 | Optional. `LOAD_SAPIENS=0` default — flip after diff review. |
| `madmom-sidecar.service` | CPU-only beat / downbeat detector for `/v1/music/analyze?analyzer=madmom`. | `8095` | none (CPU) | Optional. Default-on but degrades gracefully (`/v1/music/analyze` returns 503 if asked for `madmom` and the sidecar is down). |
| `ace-step.service` | ACE music generation (`/v1/music`, `/v2/music`). | `8001` (loopback) | cuda:1 | Optional. `LOAD_ACE=1` to enable. |

There are three more units you'll see but won't usually touch:

- `joyai-sidecar.service` / `ernie-image-sidecar.service` — image-edit / text-to-image
  sidecars on cuda:1, mutually exclusive with each other. Coexist with ACE.
- `taco-dashboard.service` — serves the operator dashboard at
  `http://192.168.1.80:8099`. Lightweight Python proxy that wraps the
  taco-backend `/dashboard` static page and adds LAN-only routing.
- `llama-swap.service` — orchestrator on `:8080`. Owns the Gemma vision /
  chat / embeddings models on a separate box; taco-backend proxies into it via
  `chat_manager.py`.

### Status check for each

```bash
for u in taco-backend noodlefinger-bff sapiens-sidecar madmom-sidecar ace-step; do
  printf "%-22s " "$u"
  systemctl --user is-active "$u"
done
```

Expected: `active` for taco-backend at minimum. The rest depend on which
sidecars you've chosen to load.

---

## 2. Daily health check

Copy-paste this into a shell. It hits all the right endpoints and surfaces
anything weird.

```bash
KEY=$(grep -v '^#' /mnt/nvme-1/servers/taco-backend/.api_keys | grep -v '^$' | head -1)
echo "== /health =="                           ; curl -s http://localhost:8090/health | jq
echo "== /v1/system/gpu =="                    ; curl -s -H "Authorization: Bearer $KEY" http://localhost:8090/v1/system/gpu | jq '.gpus[] | {device, mem_used_mib, util_pct, temp_c}'
echo "== /v1/system/workers =="                ; curl -s -H "Authorization: Bearer $KEY" http://localhost:8090/v1/system/workers | jq '{turbo_active, n: (.workers | length)}'
echo "== /v1/system/pool =="                   ; curl -s -H "Authorization: Bearer $KEY" http://localhost:8090/v1/system/pool | jq '.providers'
```

What you want to see:

- `/health` returns `200` with `{"ok": true, "queue": {...}}` and a non-zero
  uptime. `queue.processing` should track `/v1/system/workers` (any zombie
  PROCESSING > 30 min gets swept by `cleanup_loop`, see v1.15.2 in the
  CHANGELOG).
- GPU memory: cuda:0 idle should sit ~700 MiB (LTX / Flux evicted between
  requests). cuda:1 ~18 GB if ACE is loaded, +50 GB if JoyAI, +33 GB if
  ERNIE. Anything over 80 GB resident on cuda:0 with no in-flight job means
  eviction failed — see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
- Workers: `n: 1` in normal mode; `n: 2 + remotes` under turbo.
- Pool: each provider shows `{configured, target, active}`. `active <= target <= configured`.

---

## 3. API keys

Keys live in `/mnt/nvme-1/servers/taco-backend/.api_keys`. Plain text, one
key per line, `#` comments allowed, blank lines OK.

```
# taco-backend API keys (one per line)
# generate with: python generate_keys.py
abc123def456...
```

### Add a key

```bash
cd /mnt/nvme-1/servers/taco-backend
python generate_keys.py >> .api_keys      # appends one new key
```

Reload the server picks up `.api_keys` lazily on the next request — no
restart required. (The file is re-read on every `check_api_key` call;
this is fine because it's tiny and the kernel keeps it in page cache.)

### Verify a key works

```bash
curl -s -H "Authorization: Bearer <key>" http://localhost:8090/v1/system/gpu | jq -r '.gpus[0].device'
# → "cuda:0"   (anything else → key didn't match)
```

### Auth on / off

If `.api_keys` is **empty** (or contains only comments / blank lines), auth
is **disabled** server-wide — every request goes through. This is the
intentional dev-mode escape hatch. In production, never empty the file.

### Training opt-in metadata (v1.17.0-rc1)

The validator pipeline only fires for keys with `training_opt_in=1` in
`api_key_metadata` (a SQLite table inside `history.db`). The first
v2→v3 schema migration seeds every existing key with `training_opt_in=1`.
New keys added after that boot start as **opt-out** — INSERT a row by hand
if you want validator runs:

```bash
sqlite3 history.db "INSERT INTO api_key_metadata (api_key, training_opt_in) VALUES ('<key>', 1);"
```

---

## 3.5 Phase B + Phase C tooling (v1.18.0)

The retrieval (Phase B) and training-infrastructure (Phase C) ships in v1.18.0 add three operator-facing surfaces. None of these auto-fire — every Phase C invocation is operator-driven.

### Phase B — Retrieval (live since v1.18.0-rc2)

Two endpoints surface the corpus you've already captured:

- `POST /v2/embeddings/search` — privacy-gated semantic search over **the caller's own** clips. Returns ranked `(shot_id, similarity, validator_score, recency, in_final_composition)` tuples.
- `POST /v2/embeddings/recommend-loras` — aggregates LoRA performance via similarity-then-group, ranked by `0.7·mean + 0.3·boost`. Empty list when no rows have `lora_applied_id` populated.

Both are rate-limited to 10 req/sec/key (burst 10). Full reference: [`RETRIEVAL_WORKFLOW.md`](RETRIEVAL_WORKFLOW.md).

A backfill script (`scripts/backfill_prompt_embeddings.py`) populates the `clip_embeddings` virtual table from existing `generations` rows. Idempotent + resumable; run manually post-deploy.

### Phase C — Training infrastructure (live since v1.18.0-rc3, no first run yet)

Three scripts implement the weekly training pipeline:

- `scripts/construct_preference_pairs.py` — ETL across 4 signal sources (`user_retake` 0.9, `composition_kept` 0.5, `validator_pass` 0.7, `validator_fail` 0.3). Run weekly via cron.
- `scripts/train_dpo_sft.py` — defaults to dry-run; `--execute` required to consume GPU (~50-60 GPU-hours per cycle). LoRA-only via PEFT.
- `scripts/ab_decision.py` — paired t-test promote/deprecate harness; thresholds in [ADR-016](DECISIONS.md#adr-016-ab-promotedeprecate-thresholds-10---5--p005--30-mvs).

**No first run is auto-triggered.** First invocation waits until the corpus crosses ~1000 high-strength pairs (~6-8 weeks at single-operator volume against rc5+ validator). Full runbook: [`PHASE_C_TRAINING_RUNBOOK.md`](PHASE_C_TRAINING_RUNBOOK.md).

### LoRA rollback (admin-gated)

If a deployed LoRA regresses, use `POST /v1/system/lora/rollback`:

```bash
ADMIN_KEY=$(grep -v '^#' .admin_keys | grep -v '^$' | head -1)
curl -sS -X POST http://localhost:8090/v1/system/lora/rollback \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"lora_id": "<current_production_lora_id>",
       "reason": "validator score regressed 12% over 3 days"}'
```

Atomically rewrites `MCP_PRODUCTION_LORA=<previous>` in `.env` and returns `{rolled_back_from, rolled_back_to, reason, applied_at, note}`. Apply on next process restart — no in-process hot-swap. Full shape: [`API.md`](API.md) §`/v1/system/lora/rollback`.

---

## 4. The dashboard

URL: **http://192.168.1.80:8099** (LAN-only, served by `taco-dashboard.service`).

If you're not on the LAN, port-forward it:

```bash
ssh -L 8099:192.168.1.80:8099 <user>@<box>
# then open http://localhost:8099
```

### Panels

| Panel | What it shows | Common knob |
|---|---|---|
| **GPU telemetry** | Live nvidia-smi snapshot for cuda:0 / cuda:1 (memory, util, temp). Cached 2 s. | None — read-only. |
| **Live Workers** (v1.13.0) | Per-worker status: `local-0`, `local-1`, `modal-<N>`, `runpod-<N>`. Busy / idle + truncated current_job. | None — read-only. |
| **Turbo toggle** | Big red button. Claims cuda:1 for a second LTX worker, evicts ACE / JoyAI / ERNIE / sapiens. ~20 s entry, ~15 s exit. | Hit it before a heavy MV session. Hit it again to release. |
| **Remote Pool** | Two rows of N+1 buttons (0..MAX) for Modal and RunPod. Click `4` to scale Modal to 4 workers. | Use during turbo to add capacity. Cold-start latency ~30-60 s per worker. |
| **LTX advanced controls** | 14 sliders + dropdowns (sampler, stage 1 steps, scheduler shift, CFG, STG, sigmas, eta). Persisted to `.gen_config.json`. | Most operators leave these alone unless A/B-testing. |
| **Flux config** | `turbo_steps` and `turbo_guidance` for Flux 8-step turbo mode. Persisted to `.flux_config.json`. | Bump steps to 12 if turbo output looks soft. |

The dashboard polls everything every ~2-5 seconds. If a panel goes blank,
check `journalctl --user -u taco-dashboard -f` and
`journalctl --user -u taco-backend -f` side by side — usually the proxy is
fine and the backend died.

---

## 5. First test generation

The shortest path to "I know it works":

### Text-to-image (5 seconds, returns WEBP)

```bash
KEY=$(grep -v '^#' /mnt/nvme-1/servers/taco-backend/.api_keys | grep -v '^$' | head -1)

JOB=$(curl -sS -X POST http://localhost:8090/v2/text-to-image \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
    "prompt": "a small ceramic mug on a wooden table, soft window light",
    "model": "flux2-klein",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 4
  }' | jq -r .job_id)

# Poll once a second until the result_url is populated:
while :; do
  STATE=$(curl -sS -H "Authorization: Bearer $KEY" http://localhost:8090/v2/jobs/$JOB)
  echo "$STATE" | jq -c '{status, progress, phase}'
  case $(echo "$STATE" | jq -r .status) in
    completed) break ;;
    failed|cancelled) echo "FAIL"; echo "$STATE" | jq; exit 1 ;;
  esac
  sleep 1
done

curl -sS -H "Authorization: Bearer $KEY" \
  http://localhost:8090/v2/jobs/$JOB/result -o /tmp/test.webp
file /tmp/test.webp     # → RIFF (little-endian) data, Web/P image
```

### Audio-to-video (~15 s, returns silent MP4)

```bash
# Step 1: upload a short audio clip (any wav/mp3 ≤ 30s)
SLOT=$(curl -sS -X POST http://localhost:8090/v1/upload \
  -H "Authorization: Bearer $KEY" | jq -r '.upload_url + " " + .storage_uri')
UPLOAD_URL=$(echo "$SLOT" | cut -d' ' -f1)
AUDIO_URI=$(echo "$SLOT" | cut -d' ' -f2)
curl -sS -X PUT "http://localhost:8090$UPLOAD_URL" \
  -H "Authorization: Bearer $KEY" --data-binary @/path/to/test.wav

# Step 2: kick off the job
JOB=$(curl -sS -X POST http://localhost:8090/v2/audio-to-video \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"a person walking across a sunlit courtyard\",
    \"audio_uri\": \"$AUDIO_URI\",
    \"model\": \"ltx-2-3-fast\",
    \"resolution\": \"1920x1080\",
    \"duration\": 2.04,
    \"fps\": 24
  }" | jq -r .job_id)

# Step 3: same poll loop as above; result will be MP4
curl -sS -H "Authorization: Bearer $KEY" \
  http://localhost:8090/v2/jobs/$JOB/result -o /tmp/test.mp4
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames /tmp/test.mp4
```

If both come back clean, the system is healthy end-to-end (queue dispatch,
model load, GPU swap, history.db write, upload-store read).

---

## 6. Where the data lives

All paths are absolute. Everything is on `/mnt/nvme-1` (the data NVMe);
nothing important lives on the OS disk.

| What | Path | Notes |
|---|---|---|
| **API keys** | `/mnt/nvme-1/servers/taco-backend/.api_keys` | Plain text, see §3. |
| **Generation history** | `/mnt/nvme-1/servers/taco-backend/history.db` | SQLite + WAL. Schema v3 (rc1). |
| **Thumbnails** | `/mnt/nvme-1/servers/taco-backend/thumbnails/` | 256 px JPEGs, one per history row. |
| **Uploads (capability URIs)** | `/mnt/nvme-1/servers/taco-backend/uploads/` | UUID-named blobs. Referenced by `storage://<uuid>` in requests. |
| **Job results** | `/mnt/nvme-1/servers/taco-backend/uploads/` (same dir) | Output MP4 / WEBP land here too. |
| **LTX LoRAs** | `/mnt/nvme-1/servers/taco-backend/loras/` | UUID `.safetensors` + `registry.json` index. IC-LoRA outpaint + HDR live here. |
| **Flux LoRAs** | `/mnt/nvme-1/servers/taco-backend/flux_loras/` | Folder-drop discovery (no registry). Sidecar `.json` for metadata. |
| **Validator artifacts** | `/mnt/nvme-1/servers/taco-backend/validator_artifacts/` | Per-run JSON dumps from `validator.run_all_tiers`. |
| **Generation config** | `/mnt/nvme-1/servers/taco-backend/.gen_config.json` | LTX sliders. Edited via dashboard or `POST /v1/system/config`. |
| **Flux config** | `/mnt/nvme-1/servers/taco-backend/.flux_config.json` | Flux turbo knobs. |
| **Approved-images manifest** | `/mnt/nvme-1/servers/taco-backend/approved-images/manifest.json` | noodle-i → noodle-v handoff. |
| **BFF portal DB** | `/mnt/nvme-1/projects/noodlefinger-portal/bff/portal.db` | Owned by `noodlefinger-bff`. Sessions, portal state. |
| **Model checkpoints** | `/mnt/nvme-1/huggingface/` | LTX-2.3, Flux 2 Dev/Klein, Gemma snapshots. Read-only at runtime. |

### Quick sanity checks

```bash
# How big is the history?
sqlite3 /mnt/nvme-1/servers/taco-backend/history.db \
  "SELECT type, COUNT(*) FROM generations GROUP BY type ORDER BY 2 DESC;"

# What's the schema version?
sqlite3 /mnt/nvme-1/servers/taco-backend/history.db "PRAGMA user_version;"
# → 3 (v1.17.0-rc1+)

# How many uploads / how much disk?
ls /mnt/nvme-1/servers/taco-backend/uploads/ | wc -l
du -sh /mnt/nvme-1/servers/taco-backend/uploads/

# How many validator runs cached?
sqlite3 /mnt/nvme-1/servers/taco-backend/history.db \
  "SELECT validator_version, COUNT(*) FROM validator_runs GROUP BY 1;"
```

The history retention is **30 days** — older rows + their result files get
swept by `cleanup_loop`. Don't worry about disk creep; do worry if uploads
grows past ~100 GB (something is leaking the per-key daily quota).

---

## 7. Logs

Everything goes to journald. There's no log file on disk.

```bash
# Live tail of the main server:
journalctl --user -u taco-backend -f

# Last 200 lines, no follow:
journalctl --user -u taco-backend -n 200

# Time-bounded:
journalctl --user -u taco-backend --since "2 hours ago"

# All sidecars at once:
journalctl --user -u taco-backend -u sapiens-sidecar -u madmom-sidecar -u ace-step -f
```

### Useful greps

```bash
# Did anyone hit a 429 today?
journalctl --user -u taco-backend --since today | grep -E "queue_full|per_key"

# Slow VAE decode / encode?
journalctl --user -u taco-backend | grep -E "vae_decode|video_decode|flux_webp_encode"

# Turbo entries / exits:
journalctl --user -u taco-backend | grep -E "turbo|_enter_turbo_mode|_exit_turbo_mode"

# OOM recoveries (the @_with_oom_recovery wrapper logs WARN on retry):
journalctl --user -u taco-backend | grep -E "OOM|out of memory|cleanup_memory"

# Validator dispatches / scores:
journalctl --user -u taco-backend | grep -E "_dispatch_validator|validator_runs|composite"

# Modal / RunPod transport errors:
journalctl --user -u taco-backend | grep -E "modal|runpod" | grep -iE "error|timeout|reset"
```

If you need wire-level detail, `--no-access-log` is on by default in `run.sh`
to keep the log readable. To turn access logs back on for one debug session,
edit `run.sh`, drop the flag, `systemctl --user restart taco-backend`. Don't
forget to revert.

---

## 8. Common scenarios

### 8.1. "I want to deploy a new sidecar"

Pattern: each sidecar lives in its own venv at `/mnt/nvme-1/servers/<name>/`,
has a `run.sh` entrypoint, exposes FastAPI on a localhost port, and gets a
matching client module in taco-backend (`<name>_client.py`). The systemd unit
goes at `~/.config/systemd/user/<name>.service`.

Reference templates: `madmom-sidecar.service` (CPU-only) and
`sapiens-sidecar.service` (cuda:1, GPU drain semantics). Both are documented
in CLAUDE.md under their respective sections. Add a `LOAD_<NAME>=1` env var
to `config.py` so it can be turned on/off without code changes, and wire it
into `_stop_cuda1_tenants` / `_restore_cuda1_tenants` in `server.py` if it
holds GPU memory on cuda:1 (otherwise turbo entry will fail to drain it).

For full deploy steps see `docs/DEPLOY_SIDECAR.md` (TBD by infra team).

### 8.2. "I want to scale rate limits"

Don't tune the source. Set env vars in
`/mnt/nvme-1/servers/taco-backend/.env` and restart the unit. The five
knobs you care about (`PER_KEY_QUEUE_CAP`, `MAX_QUEUE_DEPTH`,
`PER_KEY_MUSIC_CAP`, `PER_KEY_BATCH_CAP`, `MAX_BATCH_QUEUE_DEPTH`) plus the
uvicorn / `LimitNOFILE` ceilings are documented in
[operator-tuning.md](./operator-tuning.md). Includes validation snippets to
confirm the new caps are live after restart.

### 8.3. "I want to roll back a bad LoRA"

LTX LoRAs (incl. IC-LoRA outpaint / HDR) are tracked in
`loras/registry.json`. Remove the entry from the registry and either delete
or rename the corresponding `<uuid>.safetensors`. Then either restart the
backend or call `POST /v1/loras/rescan` (no auth needed beyond bearer) so
the registry cache reloads.

Flux LoRAs are folder-drop — just `mv` or `rm` the file in `flux_loras/`
and `POST /v1/flux-loras/rescan`. No registry to edit.

If a fused LTX LoRA has poisoned the in-memory transformer (rare —
fusion is permanent until next reload), `POST /v1/ltx/unload` then
`POST /v1/ltx/reload` resets the cache. Costs ~30 s.

For more recovery patterns see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

### 8.4. "I want to know if the validator is firing"

Three places to look:

```bash
# (1) Dispatch log line per completed video job:
journalctl --user -u taco-backend --since "1 hour ago" | grep _dispatch_validator

# (2) Cache table — should grow with every distinct (video_sha256, validator_version) pair:
sqlite3 /mnt/nvme-1/servers/taco-backend/history.db \
  "SELECT validator_version, recommendation, COUNT(*) FROM validator_runs GROUP BY 1, 2;"

# (3) Composite scores written back onto generations rows:
sqlite3 /mnt/nvme-1/servers/taco-backend/history.db \
  "SELECT id, validator_score, validator_version FROM generations
   WHERE validator_score IS NOT NULL ORDER BY id DESC LIMIT 10;"
```

If `(1)` is silent on a key you expected, that key probably isn't
opted in — see §3 "Training opt-in metadata".

If `(1)` fires but `(2)` / `(3)` are empty, sapiens or the Gemma judge is
returning errors and the validator is short-circuiting. Look for WARN
lines in the same time window.

To run the validator synchronously against a single clip and inspect the
full payload:

```bash
curl -sS -X POST http://localhost:8090/v2/video/analyze-motion \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"video_uri": "storage://<uuid>", "prompt": "..."}' | jq
```

Returns `{video_uri, validator_version, tier1, tier2, tier3, composite_score, recommendation, reasoning_summary, ran_at, latency_s, cached}` — note the field is `composite_score`, not `composite`. See [API.md](API.md) §`/v2/video/analyze-motion` for the full shape.

---

## 9. When to escalate

### Wait-and-see (don't page anyone)

- A single OOM recovery in the logs — `@_with_oom_recovery` evicts and
  re-runs. One per hour is normal under turbo.
- Modal / RunPod transient transport errors. Remote workers are
  best-effort; turbo doesn't auto-exit on remote failure. The local pair
  keeps serving.
- Sidecar 503 on `/v1/music/analyze?analyzer=madmom` — fall back to
  `librosa`.
- Queue depth bumping the cap once or twice during a heavy MV session —
  expected. Clients see clean `429 Retry-After: 30`.

### Page the team

- `taco-backend.service` in a `Restart=on-failure` loop (more than 3
  restarts in 5 minutes). `systemctl --user status taco-backend` shows
  the failure reason; `journalctl --user -u taco-backend -n 200` has the
  traceback.
- `cuda:0` memory stuck high (>80 GB) with no in-flight job. Means
  `evict_all` lost a reference and the model didn't unload. `POST
  /v1/ltx/unload` is the first try; if that doesn't free it, restart the
  service.
- `cuda:1` drain timeout on turbo entry — `_wait_cuda1_free` raises and
  `_restore_cuda1_tenants` runs. If the rollback also fails, you're left
  with cuda:1 empty of services and turbo half-active. Restart sidecars
  manually + turbo off + restart taco-backend.
- History DB write errors / `database is locked` repeated — WAL is
  jammed. Stop the service, `sqlite3 history.db "PRAGMA wal_checkpoint(TRUNCATE);"`,
  start again.
- Disk on `/mnt/nvme-1` over 90% — `cleanup_loop` should be sweeping
  but isn't. Check `journalctl ... | grep cleanup_loop`.
- Any 5xx burst that lasts more than 60 seconds. Clients are already
  feeling it; the longer the burst the worse the blast radius
  (composition retries, MCP orchestrator backoff cascades).

### Who to call

- Backend owner: see `git log --pretty='%an %ae' -n 20` on master for
  the most recent committers.
- Frontend / portal issues that don't reproduce against the LAN URL
  but do against `api.noodlefinger.io`: probably Cloudflare or DNS,
  not us.
- Sidecar-specific repos live next to taco-backend on the same box:
  `/mnt/nvme-1/servers/{ace,madmom-sidecar,sapiens-sidecar,ltx-sidecar}/`.

---

## See also

- [ARCHITECTURE.md](./ARCHITECTURE.md) — GPU topology, swap rules, validator pipeline.
- [API.md](./API.md) — canonical client-facing API contract.
- [QUICKSTART.md](./QUICKSTART.md) — frontend / SDK developer quickstart.
- [operator-tuning.md](./operator-tuning.md) — rate limits, concurrency, fd ceilings, export quality.
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) — recipes for common failures.
- [CHANGELOG.md](../CHANGELOG.md) — what shipped when.
