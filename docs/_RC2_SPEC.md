# taco-backend v1.19.0-rc2 spec — operator depth (logs, debug, schema, per-job)

> **Historical scratchpad** — preserved as the original build spec for v1.19.0-rc2. The features described here shipped with the rc2 release; current operator-facing references are in [INDEX.md](./INDEX.md), [OPERATOR_QUICKSTART.md](./OPERATOR_QUICKSTART.md), and [TRAINING_LOOP_STATUS.md](./TRAINING_LOOP_STATUS.md).

**Audience**: codex (or any single-agent build runner). Read top-to-bottom; everything you need is here.
**Repo**: `/mnt/nvme-1/servers/taco-backend` (ONLY — do not touch other repos for this sprint).
**Branch**: master. Do NOT create a branch. Stage by explicit path. DO NOT commit (operator commits).
**Total scope**: 4 backend endpoints + 1 SSE log tail + 7 dashboard panels + tests. ~600-800 LOC.

---

## Context (where we are)

v1.19.0-rc1 just shipped (commit `b17f1a2`). Operator console exists but is shallow on operational depth. The backend captures massive volumes of state — validator runs, lineage, preference pairs, training runs, embeddings — but the dashboard surfaces only 5 high-level panels (Quality, Experiments, Training, Pair Counter, Validator Failures) plus the v1.13 Live Workers + v1.3 GPU Status / Job Queue / Controls / Advanced.

Operator pain points (the ask):
1. **No live log tail** — must SSH and `journalctl --user -u taco-backend -f`. Should be in the dashboard.
2. **`/v1/system/metrics` not surfaced** — validator_dispatch + embeddings counters land in JSON but nothing renders them.
3. **No schema/storage observability** — operator can't see schema_version, table row counts, or DB size from the UI.
4. **No per-job detail** — clicking a job_id anywhere should drill into params + phases + validator + lineage. Today no such view exists.
5. **No bearer overview** — `api_key_metadata` table not exposed; can't see per-bearer activity or toggle training_opt_in for someone else.
6. **Sidecar health is just badges** — no per-sidecar port/PID/last-error/model-loaded view.
7. **No storage breakdown** — operator guesses at history.db growth + uploads/ disk consumption.

This rc closes those.

---

## Architecture constraints

- Dashboard is `dashboard.html` (vanilla JS, served by `dashboard_server.py` on `192.168.1.80:8099` — LAN-only proxy). No framework. oklch palette. Reuse existing helpers (`getAuthHeaders()`, `callEndpoint()`, button classes). 1969 → 2691 LOC after rc1; you'll add ~700-900 more.
- Backend is `server.py` (FastAPI). Mirror the rc1 pattern for new endpoints (auth-gated by default, `_require_admin(request)` for admin-gated, return JSON).
- Schema is at v6 (taco-backend `history_store.py:CURRENT_SCHEMA_VERSION = 6`). NO schema migration in this rc.
- All new code lives in taco-backend ONLY. Do not touch noodlefinger-portal or noodle-v.
- Two protected files in taco-backend repo to leave untouched: none specifically — the protected files are in the noodlefinger-portal repo, which you should not touch at all this sprint.
- Tests target ≥ 290 (was 276 after rc1).
- All new endpoints carry the existing rate-limit middleware (already covers `/v2/embeddings/*` + `/v2/system/bulk-revalidate` + `/v1/system/*` stats). Confirm by reading the regex in server.py and extending if needed.

---

## Tier 1 — high leverage (must ship)

### 1. Live log tail (SSE)

**Backend** — `server.py`:
- New endpoint `GET /v1/system/logs/stream?source=taco-backend&since=5m&min_severity=INFO`
- Returns Server-Sent Events. Format: `data: {ts, source, severity, message}\n\n` per line.
- Source allowlist: `taco-backend, ltx-sidecar, ace-step, joyai-sidecar, ernie-image-sidecar, madmom-sidecar, sapiens-sidecar, taco-dashboard`. Reject other values with 422.
- Implementation: subprocess.Popen on `journalctl --user -u <source> --since="<since>" --output=json --follow` and stream stdout line-by-line into SSE events. Parse the JSON to extract `__REALTIME_TIMESTAMP`, `_SYSTEMD_UNIT`, `PRIORITY`, `MESSAGE`. Map PRIORITY 0-3 → ERROR, 4 → WARN, 5-6 → INFO, 7 → DEBUG.
- Severity filter: skip events whose severity is below `min_severity`.
- Auth: admin-gated (`_require_admin`) — log content can leak bearer fragments, prompts, file paths.
- Cleanup: on connection close, terminate the subprocess (use httpx StreamingResponse with cleanup callback OR FastAPI background task pattern).
- Backpressure: cap queue at 1000 lines; drop oldest with a synthetic "[N lines dropped]" event when client falls behind.
- Token-bucket rate-limit middleware should NOT apply to this endpoint (it's long-lived). Add to bypass list.

**Frontend** — `dashboard.html`:
- New section "Logs" between "Validator Failures" and "Controls".
- UI: 
  - Source dropdown (multi-select, default `taco-backend`).
  - Severity filter dropdown (INFO/WARN/ERROR/DEBUG, default WARN+).
  - Pause / Resume button.
  - Auto-scroll-to-bottom toggle (default ON).
  - Clear button.
  - Lines container: monospace, color-coded by severity (ERROR red, WARN amber, INFO neutral, DEBUG dim).
  - Cap visible lines at 500 (older lines roll off).
- EventSource API for SSE. Reconnect with exponential backoff (1s → 30s) on disconnect. Surface "(connection lost, retrying...)" inline.
- When source/severity/since changes, close the EventSource and open a new one.
- Auth: EventSource doesn't support custom headers, so use the `?token=<bearer>` query-param fallback (matches the existing `/v2/jobs/{id}/stream` pattern). Bearer must be admin.

### 2. Surface `/v1/system/metrics`

**Backend** — `server.py`:
- Endpoint already exists. Read what it returns. NO backend change unless the endpoint omits anything from the rc1 metrics block (validator_dispatch + embeddings).
- Confirm: it should expose `{validator_dispatch: {success, failure, skipped_not_video, skipped_opt_out, skipped_validator_disabled, failure_rate_pct}, embeddings: {embeddings_search_total, embeddings_search_success, embeddings_search_failure, embeddings_search_rate_limited, embeddings_search_results_avg, embeddings_search_latency_p50_ms, embeddings_search_latency_p95_ms, recommend_loras_total, bulk_revalidate_total}}`.

**Frontend** — `dashboard.html`:
- New section "Metrics" between "Quality" and "Experiments".
- Two-column layout:
  - Left: "Validator dispatch" — 5-row count table + a big "failure rate %" indicator (red if > 5%).
  - Right: "Embeddings" — search totals + p50/p95 latency sparkline + results_avg.
- Refresh: 10 s.
- Click any counter → modal with last-24h time-series (calls a NEW backend endpoint `/v1/system/metrics/history?window=24h&metric=<name>` that buffers per-minute samples — see backend §4 below).

### 3. Schema state panel

**Backend** — `server.py`:
- New endpoint `GET /v1/system/schema-state` (auth-gated):
  ```json
  {
    "schema_version": 6,
    "tables": {
      "generations": {"row_count": 10789},
      "preference_pairs": {"row_count": 0},
      "training_runs": {"row_count": 0},
      "clip_embeddings": {"row_count": 10667},
      "validator_runs": {"row_count": 1234},
      "composition_clips": {"row_count": 0},
      "compositions": {"row_count": 12},
      "api_key_metadata": {"row_count": 2}
    },
    "db_size_bytes": 1234567890,
    "wal_size_bytes": 12345678,
    "indexes": ["idx_pp_unique_pair_source", "..."]
  }
  ```
- Use `SELECT COUNT(*)` per table (cheap on SQLite with WAL).
- DB + WAL sizes via `os.stat(history_store.HISTORY_DB_PATH)` and the `-wal` sibling file.
- Indexes via `SELECT name FROM sqlite_master WHERE type='index'`.

**Frontend** — `dashboard.html`:
- New section "Schema" inside a collapsible card under "Controls" (default collapsed).
- Renders the JSON as a clean table with locale-formatted counts.
- Single fetch on first expand; refresh button re-fetches.

### 4. Per-job detail modal

**Backend** — `server.py`:
- New endpoint `GET /v1/system/jobs/{job_id}` (auth-gated, scoped to caller's api_key_hash unless admin):
  ```json
  {
    "id": "...",
    "type": "audio-to-video",
    "status": "completed",
    "phase": null,
    "params": { ...full body... },
    "result_uri": "...",
    "thumbnail_url": "/v2/history/.../thumbnail",
    "phases": [
      {"name": "queued", "ts": "...", "duration_ms": 234},
      {"name": "denoising", "ts": "...", "duration_ms": 14820},
      ...
    ],
    "validator_run": { ...full validator_runs row + parsed payload... },
    "lineage": {
      "parent_clip_id": "...",
      "shot_uuid": "...",
      "shot_config_key": "...",
      "ab_arm": "candidate" | null,
      "composition_id": "..." | null,
      "lora_applied_id": "..." | null,
      "lora_applied_strength": 0.3 | null,
      "motion_intent": "..." | null,
      "ancestors": [{"id": "...", "validator_score": 0.71}, ...],
      "descendants": [{"id": "...", "validator_score": 0.45}, ...]
    },
    "in_compositions": [{"composition_id": "...", "position_in_comp": 3, "kept": true}]
  }
  ```
- Phases are derived from existing log lines (see CLAUDE.md "Timing logs at every post-denoise phase boundary"). If `phase` field on the row history isn't structured per-phase, fall back to `params.gen_config_json` + `params.params_json` and synthesize from logs OR omit `phases` (degraded but acceptable).
- Lineage walks `parent_clip_id` chain (defensively, max depth 50 to avoid infinite loops).
- `in_compositions` queries `composition_clips` joined with `compositions`.
- 404 on unknown job_id; 403 if scope check fails.
- Add 5 unit tests in `tests/test_v1_19_console.py`.

**Frontend** — `dashboard.html`:
- Refactor existing job_id-bearing tables (Live Workers `current_job`, Validator Failures rows, Experiments per-arm samples if listed, Quality bucket-modal) so every job_id is a `<button class="job-link" data-job-id="...">` that opens the per-job modal.
- Modal layout:
  - Header: id, type, status badge.
  - Tabs: Params | Phases | Validator | Lineage | Compositions.
  - Params tab: pretty-printed JSON.
  - Phases tab: gantt-like horizontal bar chart in SVG (each phase a colored block, width = duration).
  - Validator tab: composite_score + recommendation + tier1/tier2/tier3 nested. Pretty-print payload.
  - Lineage tab: ancestor chain + descendants table + sibling shot_uuid family.
  - Compositions tab: list of comps this clip appears in with thumbnails + position_in_comp + kept flag.

---

## Tier 2 — substantial (ship if time)

### 5. Bearer table

**Backend** — `server.py`:
- New endpoint `GET /v1/system/bearers` (admin-gated):
  ```json
  {"bearers": [{
    "api_key_hash": "8c7f...e2",
    "label": "primary" | null,
    "training_opt_in": true,
    "first_seen_at": "...",
    "last_activity_at": "...",
    "total_clips": 6945,
    "mean_validator_score": 0.687 | null,
    "ab_arm_distribution": {"candidate": 312, "baseline": 309, "null": 6324}
  }]}
  ```
- JOIN `api_key_metadata` ⋈ `generations` GROUP BY api_key_hash. Aggregations: `MAX(created_at) AS last_activity`, `AVG(validator_score) FILTER (WHERE validator_score IS NOT NULL)`, `COUNT(*) GROUP BY ab_arm`.
- New endpoint `POST /v1/system/bearers/{api_key_hash}/training-opt-in` (admin-gated):
  ```json
  { "opted_in": true }
  ```
  Updates `api_key_metadata.training_opt_in` for the named bearer (any bearer, not just `me`).

**Frontend** — `dashboard.html`:
- New section "Bearers" inside the "Controls" group (admin only — hide entire section if 403 on first fetch).
- Table columns: hash (first 8 chars + tooltip with full), label, opt-in toggle (calls POST), clips, mean score, ab_arm pie (mini SVG).
- Refresh: on demand (button) + once on section expand.

### 6. Sidecar health detail

**Backend** — `server.py`:
- New endpoint `GET /v1/system/sidecars` (auth-gated):
  ```json
  {"sidecars": [{
    "name": "ltx-sidecar",
    "url": "http://127.0.0.1:8093",
    "configured": true,
    "active": true,
    "pid": 357856,
    "last_health_at": "...",
    "last_health_status": "ready",
    "last_health_error": null,
    "model_loaded": "ltx-2.3-distilled" | null,
    "vram_resident_gb": 1.78 | null
  }, ...]}
  ```
- Iterate over the known sidecar list (ltx-sidecar, ace-step, joyai-sidecar, ernie-image-sidecar, madmom-sidecar, sapiens-sidecar, taco-dashboard). For each:
  - Determine config: `LOAD_*` env var.
  - PID: `systemctl --user show -p MainPID <unit>` or read /proc.
  - Health: `httpx.get(f"{url}/health", timeout=2)` — if non-200, capture last error text.
  - VRAM: parse `nvidia-smi --query-compute-apps=pid,used_memory` and match by PID.

**Frontend** — `dashboard.html`:
- Replace the existing Model State badges (currently just colored chips) with a card-per-sidecar layout. Each card shows: name, port, status dot (ready/idle/error), last_health_at, model loaded, VRAM resident, "view logs" link → opens the Log Tail panel pre-filtered to that sidecar.
- Refresh: 5 s.

### 7. Storage depth

**Backend** — `server.py`:
- New endpoint `GET /v1/system/storage` (auth-gated):
  ```json
  {
    "history_db_bytes": 1234567890,
    "wal_bytes": 12345678,
    "uploads_bytes": 98765432,
    "uploads_count": 1234,
    "thumbnails_bytes": 5432109,
    "thumbnails_count": 5678,
    "validator_artifacts_bytes": 0,
    "validator_artifacts_count": 0,
    "clip_embeddings_projected_bytes": 174784512
  }
  ```
- Walk each dir with `os.scandir` for size + count. clip_embeddings projection: `row_count × 4096 × 4 + table overhead`.

**Frontend** — `dashboard.html`:
- Inside the "Schema" section, add a "Storage" subsection. Renders the JSON as a stacked-bar chart in SVG (history.db | uploads | thumbnails | clip_embeddings | other) with labels.
- Refresh: 30 s.

---

## Tests

`tests/test_v1_19_rc2.py` (NEW). ~12 tests covering:
- All 4 Tier 1 endpoints + 3 Tier 2 endpoints (response shape, auth, scope check)
- Per-job detail modal: privacy gate (bearer A queries B's job → 403), 404 on unknown id, lineage walker terminates on cycle (defensive)
- Bearer toggle: admin-gated, persists, returns correct shape
- SSE log tail: connect with `?token=`, get at least one event, terminate cleanly on disconnect
- Schema state: row counts match independent sqlite query

Target: 276 + 12 = ≥ 288 green. Run `cd /mnt/nvme-1/servers/taco-backend && uv run --no-sync pytest tests/ -q` and verify.

---

## Critical safety constraints

1. **DO NOT commit.** Operator commits.
2. **NEVER `git add -A` / `git add .` / `git reset --hard` / `git pull --rebase`.** Stage by explicit path or don't stage at all.
3. **DO NOT touch any file outside `/mnt/nvme-1/servers/taco-backend/`.** Especially noodlefinger-portal and noodle-v are out of scope.
4. **Privacy gate on `/v1/system/jobs/{id}`** — non-admin caller MUST be scoped by api_key_hash. Test the 403 path explicitly.
5. **Log tail is admin-gated** — log content leaks bearer fragments, prompts, file paths. Don't expose to regular bearers.
6. **subprocess cleanup** — the journalctl streamer MUST terminate on client disconnect or you'll leak processes. Use a finally block.
7. **No backwards-incompat changes** — every new endpoint is additive. Existing endpoints unchanged.

---

## File inventory

Modified:
- `/mnt/nvme-1/servers/taco-backend/server.py` — 7 new endpoints + SSE log streamer (~400-500 LOC additive)
- `/mnt/nvme-1/servers/taco-backend/dashboard.html` — 7 new panels (~600-800 LOC additive)
- `/mnt/nvme-1/servers/taco-backend/CLAUDE.md` — v1.19.0-rc2 highlights paragraph above rc1
- `/mnt/nvme-1/servers/taco-backend/CHANGELOG.md` — v1.19.0-rc2 entry

New:
- `/mnt/nvme-1/servers/taco-backend/tests/test_v1_19_rc2.py` — ~12 tests

---

## Verification (operator runs after codex finishes)

```
cd /mnt/nvme-1/servers/taco-backend
uv run --no-sync pytest tests/ -q
# expected: ≥ 288 green

systemctl --user restart taco-backend
# /health should return 200 within ~10s

curl -fsS http://localhost:8090/v1/system/schema-state -H "Authorization: Bearer $KEY"
# verify shape

# open dashboard
xdg-open http://192.168.1.80:8099/dashboard
# verify 7 new panels render
```

If any test fails or any panel breaks, do not mark complete. Surface the specific failure clearly so operator can decide whether to ship anyway or fix.

---

## Tag

When this rc lands cleanly: `v1.19.0-rc2`. Operator commits + pushes.
