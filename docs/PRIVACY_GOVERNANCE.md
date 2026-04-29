# Privacy + Governance Posture

**Versions covered**: taco-backend `v1.18.0-rc3` · noodlefinger-bff `v0.3.0` · noodlefinger-mcp `v0.7.0`
**Audience**: operator, auditor, future external bearer
**Sibling docs**: [`CAPTURE_VALIDATOR.md`](CAPTURE_VALIDATOR.md) · [`CAPTURE_VALIDATOR_ROADMAP.md`](CAPTURE_VALIDATOR_ROADMAP.md) · [`API.md`](API.md) · [`operator-tuning.md`](operator-tuning.md)
**Date**: 2026-04-29

This document is the operator's reference for **what the system actually does
with user data**: what is captured per generation, who can see it, what is
used for training, and how the opt-out path works. Cite tables, columns, env
vars, and code paths — not legalese.

`CAPTURE_VALIDATOR.md` describes the validator pipeline as a feature; this
describes the same pipeline as a data-governance surface. Where the two
overlap, this doc is the canonical statement of "what flows where".

---

## 1. Tenancy model

### 1.1 Today: single-operator single-tenant

The deploy at `/mnt/nvme-1/servers/taco-backend/` runs as a single FastAPI
process owned by the operator. `.api_keys` and `.admin_keys` files (one
bearer per non-comment line) gate access. The operator is the only data
subject; "training data" means the operator's own clips, prompts, and
retake decisions.

In this regime the privacy posture is mostly trivial: the operator owns
every row in `history.db`, every signal in the BFF, every artifact under
`uploads/`. The opt-in flag default is **ON globally** (seeded on first
v3 migration) precisely because the operator is opting in to themselves.

### 1.2 Future: multi-tenant via `api_key_hash` scoping

Multi-tenant is the design target, not the current state. The privacy
gate is shaped so that flipping to multi-tenant is a configuration
change, not a refactor. The spine is:

> **Every privacy-critical query filters by `api_key_hash`. Every
> training-data write is gated by `training_opt_in`. Every cross-bearer
> read is rejected at the data layer.**

The relevant invariants live in three places:

- `taco-backend/history_store.py` — `_hash_key(api_key)` (sha256, hex);
  `api_key_hash` column on `generations` (line 54), index
  `idx_api_key_hash` (line 68); every retrieval query filters by it
  (lines 928, 937, 944, 951, 958, 967, 981, 1008).
- `taco-backend/server.py` — `_is_training_opted_in(api_key)`
  (server.py:3642) is the only authority that says "this bearer's
  output may flow into training corpora"; `_on_job_complete` (3724)
  checks it before dispatching the validator.
- `noodlefinger-bff/db.py` — schema v2 (v0.3.0) added
  `user_signals.api_key_hash` + `idx_signals_api_key_hash` so signal
  ingest is per-bearer scopable from day one.

### 1.3 Multi-tenant readiness — what's **already** done vs **not yet** done

| Surface | Status |
|---|---|
| `api_key_hash` on every privacy-critical history query | ✅ since v1.18 (`generations` retrieval/delete paths) |
| BFF `user_signals` carries `api_key_hash` | ✅ since BFF v0.3.0 |
| Cross-bearer history retrieval test | ✅ `test_embeddings_search_privacy_gate` (rc2) |
| Training corpus opt-in filter | ✅ `_is_training_opted_in` gate on validator dispatch (rc3) |
| Per-tenant rate limits | ✅ `PER_KEY_QUEUE_CAP`, `PER_KEY_MUSIC_CAP`, `PER_KEY_BATCH_CAP`, `PER_KEY_LORA_COUNT`, `PER_KEY_UPLOAD_BYTES_PER_DAY` |
| Per-tenant token-bucket rate limit on retrieval endpoints | ✅ since v1.18.0-rc2 — 10 req/sec/key with burst 10 on `/v2/embeddings/*` and `/v2/system/bulk-revalidate`; bucket keyed by `sha256(api_key)` so raw bearers never land in heap dumps. See [`API.md`](API.md) §rate-limits. |
| Multi-tenant load test | ❌ **not done** — Phase D prerequisite |
| Hard-fail audit endpoints (`GET /v1/training-usage/{key_hash}`) | ❌ **not done** — Phase D |
| Right-to-delete cascade across `preference_pairs` | ❌ **not done** — Phase D candidate (see [§7](#7-right-to-delete-forward-looking)) |
| Tenant-bound BFF actor (today: `actor_email` falls back to `"unknown"`) | ❌ **not done** — `bff/routers/mcp_events.py:11` carries the TODO |

Multi-tenant deploy is **not a configuration flip away**. It is a
configuration flip + the four red-row items + counsel-side review of
the Sapiens AUP read (see [§10](#10-security-posture)).

---

## 2. What's captured per generation

Each completed generation lives as one row in
`history.db::generations`. The schema v3 (rc1) layout, with a privacy
read on each column:

### 2.1 Identifying the subject

| Column | Type | What it stores | Privacy note |
|---|---|---|---|
| `id` | TEXT (uuid4) | Generation ID — surface in API, dashboard, MCP | Random; carries no subject information |
| `api_key_hash` | TEXT | `sha256(api_key)` hex, computed at write time via `_hash_key` (history_store.py:132) | **Raw API key never stored.** sha256 is irreversible. Stale keys remain bound to their hash even after rotation. |
| `created_at` | REAL | Unix epoch | timing oracle; combine with bearer to fingerprint a user (see [§11.3](#113-side-channels-the-irreversible-hash-doesnt-protect-against)) |

The `api_key_hash` filter on every retrieval query
(`SELECT … FROM generations WHERE api_key_hash = ?` repeated 8 times
in `history_store.py`) is the load-bearing invariant. A bug that drops
the filter would expose every bearer's history to every other bearer
in a multi-tenant deploy — there is no second-line defense.

### 2.2 Generative inputs (user-supplied)

| Column | Type | What it stores |
|---|---|---|
| `prompt` | TEXT | Verbatim user prompt (or LTX-rewritten when `enhance_prompt=true` — original lives in `params_json`, rewrite in `enhanced_prompt`) |
| `params_json` | TEXT | Full Pydantic-dumped request body — preserves `storage://` URIs, LoRA refs, keyframes, seed |
| `gen_config_json` | TEXT | LTX `_gen_config` snapshot at dispatch time (sampler, steps, CFG, etc.) |
| `enhanced_prompt` | TEXT | Gemma-rewritten prompt (only populated when `enhance_prompt=true`); NULL otherwise |
| `seed` | INTEGER | RNG seed for reproducibility |

**Personal data lives in `prompt`.** Users may type names, email
addresses, descriptions of real people, or other identifiers. There
is no automated PII scrubbing; the operator and any bearer-level data
subject is responsible for their own input hygiene. This is the same
contract every LLM provider operates under.

`storage://` URIs in `params_json` point at user-uploaded reference
images, reference audio, source videos for retake / outpaint / HDR.
Those uploads live under `uploads/` and have their own retention
(see [§7](#7-right-to-delete-forward-looking)).

### 2.3 Generative outputs

| Column | Type | What it stores |
|---|---|---|
| `result_uri` | TEXT | `storage://<uuid>` — points at the output MP4/WEBP/PNG |
| `thumbnail_uri` | TEXT | 256-wide JPEG thumbnail (PyAV first-frame extract for video) |
| `width` / `height` / `model` / `lora_id` / `lora_strength` | mixed | Reproducibility metadata |

Output files live under `uploads/` and are governed by the same
retention sweep as inputs.

### 2.4 Validator scoring (rc2+)

| Column | Type | Populated by | Privacy note |
|---|---|---|---|
| `validator_score` | REAL | `_dispatch_validator` UPDATE | Composite (0..1); fast SQL filtering |
| `validator_payload_json` | TEXT | same | Per-tier payload (RAFT flow stats, Sapiens stub flag, Gemma judge verdict + reasoning + retake hint). **The Gemma `reasoning` field can quote user prompts and describe scene content** — treat as PII-equivalent |
| `validator_version` | TEXT | same | Pinned config.VALIDATOR_VERSION (e.g. `"1.17.0-rc5"`) |

The Gemma judge's `reasoning` and `retake_hint` strings are LLM-
generated descriptions of the user's video and prompt. They carry
the same sensitivity as the prompt itself.

### 2.5 Lineage stamps (rc1+)

| Column | Type | Populated by | Privacy note |
|---|---|---|---|
| `parent_clip_id` | TEXT | `/v2/retake` handler via `find_id_by_result_uri(body.video_uri)` | Retake provenance — links a "rejected" clip to its "chosen" successor |
| `shot_uuid` | TEXT (16-hex or 32-hex) | MCP `_apply_shot_lineage` → backend `_HISTORY_ONLY_PARAMS` strip → history.save | **Deterministic** hash via `hashlib.sha256(prompt + image_uri + position).hexdigest()[:16]` (legacy mcp ≤ v0.7.x, 16-hex) or `[:32]` (mcp v0.8+, 32-hex). Load-bearing for resume safety: the same shot across resumes hashes to the same row in lineage tables. Privacy implication: a `shot_uuid` is **not random** — anyone who can guess the (prompt, image_uri, position) tuple can re-derive it. |
| `shot_config_key` | TEXT (full sha256) | same | DPO pair-matching key (prompt + image_uri + audio_start_s + duration_s + model + lora_id + lora_strength) |
| `composition_id` | TEXT | **never written** today (denorm convenience) | Forward-looking; redundant with `composition_clips` join |
| `lora_applied_id` / `lora_applied_strength` | TEXT/REAL | **never written** today | Captures the *actual* fused LoRA at runtime (vs requested) — Phase B candidate |
| `prompt_embedding` | BLOB | **never written** today | 3584-dim float32 from Gemma; deferred per [Roadmap §2.3](CAPTURE_VALIDATOR_ROADMAP.md). Note: forward-look in roadmap is to introduce a `clip_embeddings` virtual table once `sqlite-vec` is loaded; the column on `generations` may be deprecated in favor of the virtual table |

`shot_uuid` and `shot_config_key` are derived hashes. They are not
reversible to the original prompt unless an attacker can guess the
prompt — which is the expected case for low-entropy prompts.
**Operators should consider these stamps as carrying the same
sensitivity as the prompt itself**, not as anonymized identifiers.

---

## 3. The `training_opt_in` flag — the spine

### 3.1 Schema

```sql
CREATE TABLE IF NOT EXISTS api_key_metadata (
    api_key_hash TEXT PRIMARY KEY,    -- sha256(api_key) — raw keys never stored
    training_opt_in INTEGER NOT NULL DEFAULT 1,
    tier TEXT DEFAULT 'pro',
    notes TEXT,
    created_at REAL,
    updated_at REAL
);
```

Lives in `history_store.py:121-126`. Per-bearer flag with three
material states: `1` (opted in), `0` (opted out), and **row absent**
(treated as opt-out by `_is_training_opted_in`, server.py:3662-3665).

### 3.2 Defaults

- **First v2→v3 migration** (one-shot): `_maybe_seed_api_key_metadata`
  (history_store.py:749) reads `.api_keys`, hashes each non-comment
  line, INSERTs with `training_opt_in=1`. Convenience for the
  single-tenant deploy: the operator is opted in to themselves
  without manual SQL.
- **`.api_keys` itself is never modified** — the file is the auth
  surface, the metadata table is the governance surface. They can
  drift; the metadata table wins.
- **Bearers added to `.api_keys` after the seed run** are opted-out
  by default (defense-in-depth). Operator must explicitly INSERT to
  enable training capture.

### 3.3 Enforcement points

The flag is checked in **one place** — `_is_training_opted_in`
(server.py:3642). It is invoked from:

- `_on_job_complete` (server.py:3745) — gates the passive validator
  dispatch. Opted-out bearers' jobs do not run the validator at all
  (no scoring, no payload, no `validator_runs` cache row, no UPDATE
  on `generations`).

It is **not yet** invoked from:

- `find_similar_shots` retrieval (Phase B 3.2) — when shipped, will
  filter by `api_key_hash` (per-bearer scope) but should **also**
  honor `training_opt_in` if cross-bearer search is ever enabled
  (currently rejected by privacy-gate test, rc2).
- DPO pair construction (Phase C 4.1) — joins `generations` by
  `shot_config_key`. The ETL spec must filter on
  `training_opt_in=1`. Documented as a Phase C requirement; not
  yet enforced because no writer exists.

Single-tenant deploy: this gap is theoretical. Multi-tenant: this
gap must close before flipping to multi-tenant.

### 3.4 Operator commands

**Verify the seed ran**:

```bash
sqlite3 /mnt/nvme-1/servers/taco-backend/history.db \
  "SELECT COUNT(*), SUM(training_opt_in) FROM api_key_metadata"
# → (N, N)  where N matches non-comment lines in .api_keys
```

**Audit who's opted in**:

```bash
sqlite3 history.db \
  "SELECT api_key_hash, training_opt_in, tier, datetime(created_at,'unixepoch')
   FROM api_key_metadata"
```

**Flip a bearer to opt-out**:

```bash
sqlite3 history.db \
  "UPDATE api_key_metadata
   SET training_opt_in=0, updated_at=strftime('%s','now')
   WHERE api_key_hash = '<sha256-of-key>'"
```

**Compute `api_key_hash` for a known bearer**:

```python
import hashlib
hashlib.sha256("the-bearer-token".encode()).hexdigest()
```

**Add a new external bearer with opt-in**:

```bash
echo "new-bearer-token" >> /mnt/nvme-1/servers/taco-backend/.api_keys
sqlite3 history.db \
  "INSERT INTO api_key_metadata
     (api_key_hash, training_opt_in, tier, created_at, updated_at)
   VALUES ('<sha256>', 1, 'pro',
           strftime('%s','now'), strftime('%s','now'))"
systemctl --user restart taco-backend   # reload .api_keys
```

### 3.5 Forward-look: dashboard toggle

The Phase D "Account" page in `noodlefinger-portal` will surface a
training opt-in toggle reading/writing `api_key_metadata`. Today the
flag is operator-managed via SQL. There is no end-user-visible
opt-out UI. **External bearers must trust the operator's manual
enforcement** until that page ships.

---

## 4. Data flow with privacy gates

```
┌──────────────────┐
│ User generates   │  bearer in Authorization header
│ a clip via       │
│ taco-backend API │
└──────────────────┘
         │
         │  [GATE 1: API_KEYS check_api_key middleware]
         │  401 if bearer absent
         ▼
┌─────────────────────────────────┐
│ history_store.save(generation,  │
│   api_key=bearer)               │
│ → api_key_hash = _hash_key(...) │  raw key dropped here
│ → row written to generations    │
└─────────────────────────────────┘
         │
         │  on_complete callback
         ▼
┌─────────────────────────────────┐
│ _on_job_complete(job)           │
│   if job.status != COMPLETED:   │
│       skip                      │
│   if not video type:            │
│       skip ("skipped_not_video")│
│   if not _is_training_opted_in: │  ◄─── [GATE 2: training_opt_in]
│       skip ("skipped_opt_out")  │
│   else:                         │
│       fire-and-forget dispatch  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ validator.run_all_tiers(...)    │
│ tier1 RAFT, tier2 Sapiens,      │
│ tier3 Gemma judge               │
│ → validator_runs cache (sha256, │
│   version)                      │
│ → UPDATE generations.score      │
└─────────────────────────────────┘

╔═══════════════════════════════════╗     ╔══════════════════════════════════╗
║ MCP (Claude Code subprocess)      ║     ║ noodlefinger-bff                 ║
║                                   ║     ║                                  ║
║ Session.append_event(...)         ║─────║▶ POST /api/mcp/events            ║
║   _tee_event_fire_and_forget(...) ║     ║   actor_email = X-User-Email     ║
║   timeout=2s, swallow errors      ║     ║                || payload.email  ║
║   gated on NOODLEFINGER_BFF_URL   ║     ║                || "unknown"      ║
║                                   ║     ║   api_key_hash = session config  ║
║                                   ║     ║   user_signals_log(...)          ║
║                                   ║     ║                                  ║
║                                   ║     ║ [GATE 3: api_key_hash on read]   ║
║                                   ║     ║   future retrieval queries       ║
╚═══════════════════════════════════╝     ╚══════════════════════════════════╝
```

### 4.1 taco-backend `history.db` — per-bearer scoping

Every retrieval/deletion query in `history_store.py` carries
`WHERE api_key_hash = ?`:

- `get_generation(id, api_key)` (history_store.py:966)
- `delete_generation(id, api_key)` (history_store.py:1008)
- `list_generations(api_key, …)` (history_store.py:914-958, 5 paths)

There is no list-all path. There is no admin override path. A bug that
drops the filter would expose every bearer to every other bearer in a
multi-tenant deploy. The privacy posture relies on this being checked
in code review, not at runtime.

### 4.2 BFF `user_signals` — per-actor + per-bearer scoping

Schema v2 (BFF v0.3.0, db.py:81-87):

```sql
ALTER TABLE user_signals ADD COLUMN api_key_hash TEXT;
CREATE INDEX idx_signals_api_key_hash ON user_signals(api_key_hash);
```

Both `actor_email` (v1) and `api_key_hash` (v2) live on every row.
Retrieval should filter by either or both depending on the auth
surface — `actor_email` for portal logins, `api_key_hash` for
machine-bearer auth.

`actor_email` falls back to literal `"unknown"` when the
X-User-Email header is missing AND the payload doesn't carry one
(BFF `routers/mcp_events.py`). In single-tenant this is fine; for
multi-tenant the BFF would need to bind a portal-issued bearer at
MCP install time. Tracked as TODO in `mcp_events.py:11`.

### 4.3 MCP→BFF event tee

`Session.append_event` (mcp/_session.py) calls
`_tee_event_fire_and_forget(payload)` — `asyncio.create_task` of an
httpx POST with 2 s timeout. **Fully gated on `NOODLEFINGER_BFF_URL`
env var; unset → no-op.**

What's teed:

- `clip_submitted` / `clip_completed` / `validator_result` /
  `validator_retake_triggered` / `validator_skipped` /
  `composition_exported` — every state transition the orchestrator
  emits.
- Payload includes `session_id`, `event_kind`, `event_data`, `ts`,
  optional `actor_email`. The `event_data` blob can carry per-clip
  validator scores (composite, recommendation), retake counts,
  prompt fragments — treat as same-sensitivity as the source data.

What's not teed:

- Result MP4 bytes. The tee carries metadata only. Outputs stay in
  taco-backend's `uploads/`.

Failures log WARN and swallow; orchestrator correctness must not
depend on BFF reachability.

### 4.4 Phase B retrieval (forward-looking)

When `find_similar_shots(prompt, k)` ships (Roadmap §3.2), it MUST:

- Embed the caller's prompt via `chat_manager.embed`.
- Vector-search `clip_embeddings` virtual table.
- **Filter results by `api_key_hash = <caller's hash>`** before
  returning. Cross-bearer matches must be excluded at the SQL
  layer, not post-filtered in Python.
- Honor `min_validator_score` to exclude low-confidence rows from
  the recommendation surface.

A regression test (`test_embeddings_search_privacy_gate`, rc2)
already exists for the cross-bearer rejection invariant, even though
the search endpoint isn't yet implemented. Keep that test; it is
the data-layer canary for the multi-tenant flip.

### 4.5 Phase C training (forward-looking)

When DPO pair construction ships (Roadmap §4.1), the ETL job MUST:

- Walk `generations` grouped by `shot_config_key`.
- **Filter on `_is_training_opted_in(...)` for every bearer
  contributing to a pair.** A pair where chosen and rejected come
  from different bearers and one is opted-out must be dropped.
- Stamp `training_runs.run_id` so audit endpoints
  ([§8](#8-opt-out--audit-endpoints-forward-looking)) can answer
  "whose data was in run X?".

The opt-in filter must be applied at pair-construction time, not at
query time. Pairs from opted-out bearers should never enter
`preference_pairs`.

---

## 5. Validator drift + version scoping

The validator pipeline is a data-integrity surface. A version bump
forces re-runs cleanly so historical training corpora are not
contaminated by mixed-version scores.

### 5.1 The cache key

`validator_runs(video_sha256, validator_version) UNIQUE INDEX`
(history_store.py — schema v3 table). Two clips with the same content
hash but different validator versions get two rows. Cache hit logic
in `validator.run_all_tiers` keys on both columns.

### 5.2 What bumps `VALIDATOR_VERSION`

`config.VALIDATOR_VERSION` (currently `"1.17.0-rc5"`). Bump on:

- `JUDGE_PROMPT_V1` text change (changes tier-3 verdict semantics).
- Composite-formula change (changes recommendation thresholds).
- Sapiens model swap (rc-final → real inference).
- Tier-1 RAFT model swap (`raft_small` → `raft_large`).
- Any change that would produce a different score on the same input.

### 5.3 What it doesn't auto-do

- **Does not auto-revalidate historical rows.** Bumping the version
  invalidates cache lookups, but the passive `_dispatch_validator`
  only fires on **new** completions. Historical rows stay at the
  old version forever unless explicitly re-run via MCP
  `resume_music_video(..., revalidate=True)` or (forward-looking) a
  `POST /v2/admin/validator/backfill` maintenance endpoint
  (Roadmap candidate, not built).

### 5.4 Phase C filter

When pair construction lands, it MUST filter
`preference_pairs.chosen_clip_id` and `rejected_clip_id` by
`generations.validator_version = ?` (or in a documented set of
compatible versions) before emitting a pair. Cross-version pairs
would mix scores from incompatible scoring regimes.

### 5.5 Phase B filter

`find_similar_shots` may filter by `validator_version` (default:
current). Older validator scores can be returned with a clear
`validator_version` field on the response so the caller can decide
whether to trust them.

---

## 6. Reproducibility ledger (`training_runs`)

```sql
CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    base_model TEXT,
    base_model_sha TEXT,
    lora_output_path TEXT,
    lora_registry_id TEXT,
    num_pairs INTEGER,
    val_loss REAL,
    eval_metrics_json TEXT,
    trained_at REAL,
    deployed_at REAL,
    deprecated_at REAL
);
```

Forward-looking — no writer wired. When the Phase C training cron
ships (Roadmap §4.2), every deployed LoRA gets a `training_runs` row
with the seed, hyperparameters, dataset snapshot path, base-model
sha, validator version at training time, and deployment / deprecation
timestamps.

### 6.1 Audit story: "whose data was in training run X?"

```sql
SELECT DISTINCT g.api_key_hash
FROM preference_pairs pp
JOIN generations g ON g.id IN (pp.chosen_clip_id, pp.rejected_clip_id)
WHERE pp.used_in_training_run_id = ?;
```

This is the spine of the right-to-be-told question. Note that
`api_key_hash` is irreversible — the operator can map back to the
original bearer only via the `.api_keys` file (which they control).
External bearers asking "was my data used?" must supply their own
bearer; the operator computes the hash and runs the query.

### 6.2 Reproducibility from a `training_runs` row

The plan-of-record (`melodic-sniffing-beacon.md` §C) requires every
training run to be reproducible from its row:

- `training_seed` — RNG seed for DPO + sampler.
- `hyperparams_json` — learning rate, beta, batch size, etc.
- `dataset_snapshot_path` — frozen `preference_pairs` snapshot at
  training time (so re-running on a later corpus gives the same
  result).
- `code_sha` — git SHA of taco-backend at training time.
- `base_model_sha` — sha of the LTX checkpoint trained against.
- `validator_version_at_train` — the version that scored every pair
  in the training set.

These are columns on the row + paths into immutable artifacts. The
schema is wired; the writer is not.

---

## 7. Right-to-delete (forward-looking)

### 7.1 Today: 30-day rolling retention

`history.cleanup()` runs as a periodic background task (the
`cleanup_loop` of v1.15.2 fame). It deletes:

- `generations` rows older than 30 days.
- The associated `result_uri` and `thumbnail_uri` files under
  `uploads/` and `thumbnails/`.

This is a soft form of right-to-delete: data ages out automatically.

### 7.2 Cascades that work today

- `composition_clips.clip_history_id` references `generations.id`.
  When the parent row is deleted, the lineage row becomes a
  dangling pointer. Documented as a Phase D candidate to add
  `ON DELETE CASCADE`.
- `validator_runs` is keyed by `video_sha256`, not `generation_id`.
  Validator score rows survive history deletion. This is intentional
  — the cache is content-addressed, not subject-addressed. To purge
  validator scores for a deleted clip, the operator must explicitly
  DELETE from `validator_runs` matching the (now-orphaned) sha256.

### 7.3 Cascades that don't work today

- **`preference_pairs`** referring to deleted `generations` rows
  are NOT cascaded. If a clip ages out of `generations` after being
  used in pair construction, the pair becomes a dangling pointer
  (the chosen/rejected clip can no longer be re-fetched for
  re-training). Documented as a Phase D candidate to add
  `ON DELETE SET NULL` + a "this pair is orphaned" filter on
  pair-consumer queries.
- **Validator artifacts** (`validator_artifacts/<clip_id>/...`,
  Roadmap forward-look) — not yet wired, so no cascade exists. When
  wired, `history.cleanup()` must be extended to `rmtree` the
  per-clip artifact directory alongside the result file.

### 7.4 Active right-to-delete for a specific bearer

```sql
-- Find every row for the bearer:
SELECT id, result_uri, thumbnail_uri, created_at
  FROM generations WHERE api_key_hash = '<sha256>';

-- Delete them (DELETE cascades into the row only; files unlink
-- in history.delete_generation):
-- (Use the `delete_generation` Python helper per row, NOT raw SQL,
-- so the upload files are unlinked — see history_store.py:1008.)
```

There is no batch right-to-delete CLI. Operator must script it.

---

## 8. Opt-out + audit endpoints (forward-looking)

The following endpoints do not exist yet. They are the multi-tenant
prerequisites; flag them on the Phase D checklist.

### 8.1 `GET /v1/training-usage/{api_key_hash}` (proposed)

Returns: list of `training_runs.run_id` where the bearer's data
appeared in a constituent `preference_pair`.

```json
[
  {
    "run_id": "...",
    "trained_at": 1730000000.0,
    "deployed_at": 1730086400.0,
    "deprecated_at": null,
    "lora_registry_id": "dpo-mv-v1",
    "num_pairs_from_caller": 42,
    "num_pairs_total": 1247
  }
]
```

Auth: caller must be either (a) admin, or (b) the bearer in question
(verified by `_constant_time_match(token, [api_key])` against
`compare_digest(_hash_key(token), api_key_hash)`).

### 8.2 `POST /v1/account/training-opt-out` (proposed)

Flips `training_opt_in=0` for the calling bearer + cascades a delete
mark across `preference_pairs.chosen_clip_id` and `rejected_clip_id`
where the bearer's `api_key_hash` matches. Does NOT delete completed
`training_runs` artifacts (those are already trained and deployed)
but DOES exclude the bearer's data from future training cron passes.

### 8.3 Auditor read-only access

Forward-look: a separate `.audit_keys` file (parity with `.admin_keys`
via SEC P0-2 pattern) gates a read-only auditor surface that can
SELECT from `api_key_metadata` and `training_runs` without seeing
prompts or output URIs. Not yet specified; flag for design when the
first external auditor request lands.

---

## 9. Multi-tenant readiness checklist

Repeated from [§1.3](#13-multi-tenant-readiness--whats-already-done-vs-not-yet-done) for operator convenience as the Phase D
gate.

**Done**:

- [x] `api_key_hash` on every privacy-critical query
- [x] BFF `user_signals` carries `api_key_hash` (v0.3.0)
- [x] Cross-bearer history retrieval test
  (`test_embeddings_search_privacy_gate`)
- [x] Training corpus opt-in filter (`_is_training_opted_in` gates
  `_on_job_complete`)
- [x] Per-tenant rate limits

**Not done — Phase D blockers**:

- [ ] Multi-tenant load test (verify rate limits hold across N bearers)
- [ ] Hard-fail audit endpoints (`GET /v1/training-usage/{hash}`)
- [ ] Right-to-delete cascade across `preference_pairs`
- [ ] BFF `actor_email` no-fallback mode (no `"unknown"` rows)
- [ ] Sapiens AUP read re-confirmation with counsel
  (see [§10](#10-security-posture))
- [ ] Dashboard "Account" page training opt-in toggle in
  `noodlefinger-portal`
- [ ] DPO pair construction enforces `training_opt_in` filter at
  ETL time
- [ ] Phase B `find_similar_shots` enforces `api_key_hash` filter at
  SQL layer (not Python post-filter)

---

## 10. Security posture

### 10.1 Auth surfaces

- **`.api_keys`** — bearer file at the project root. One non-comment
  line per bearer. When empty, auth is **globally disabled**
  (`config.API_KEYS == []` short-circuits `check_api_key` middleware).
  Empty-file mode is intended for development only; a WARN logs at
  boot so operators notice
  (server.py:704-705).
- **`.admin_keys`** — separate bearer file (SEC P0-2, v1.8.2). When
  empty, every `.api_keys` bearer is treated as admin
  (backwards-compat bridge with a startup WARN, server.py:706-711).
  When populated, only admin bearers can hit mutation endpoints.

### 10.2 Admin-gated endpoints (12 of them)

`_require_admin(request)` (server.py:5490-5513) gates:

- `POST /v1/system/turbo`
- `POST /v1/system/pause` / `resume`
- `POST /v1/ltx/{unload,reload}` / `POST /v1/flux/{unload,reload}`
- `POST /v1/system/pool/remote-workers` (and variants)
- `POST /v1/system/config` / `flux-config` (mutations + resets)
- `POST /v1/loras/...` mutation endpoints (upload / register)
- `POST /v2/admin/...` (anywhere in the namespace)
- `DELETE` of others' data is blocked because retrieval already
  filters by `api_key_hash`

403 on mismatch (not 404 — admin endpoints are
known-existent, no existence oracle to protect).

### 10.3 Rate limits + per-key caps

`config.PER_KEY_QUEUE_CAP` (default 100), `PER_KEY_MUSIC_CAP`
(20), `PER_KEY_BATCH_CAP` (20), `PER_KEY_LORA_COUNT`,
`PER_KEY_UPLOAD_BYTES_PER_DAY`. All keyed by `_sha256_key(api_key)`
in in-memory bucket dicts (server.py:5520-5530). Restart resets
counters; documented as acceptable in the v1.8.2 SEC notes.

### 10.4 Sapiens AUP read

Sapiens-2 ships under Meta's custom AUP. Clause §1.b.vi.ii (`for
biometric processing`) is potentially blocking. Operator's narrower
read in `LICENSE_NOTES.md` (2026-04-29) is **internal-use only**:

- Synthetic input only (LTX-generated frames; no real persons).
- No real-person identification.
- No SaaS surface.
- No external bearers.

**Reversibility**: if scope ever changes (multi-tenant, external
bearers, SaaS), revisit the attestation with counsel before
continuing. Substitute candidates:

- DWPose (Apache-2.0)
- ViTPose (Apache-2.0)

The schema is forward-stable: tier-2 returns `pose_temporal_*`
fields whether the backend is Sapiens, DWPose, or ViTPose. The
client-side change in `validator._run_tier2_sapiens` is one
configuration line.

### 10.5 Other relevant security entries

- `secrets.compare_digest` for every bearer comparison (constant-time)
- API keys hashed (`sha256`) before being used as map keys
- `uploads/` directory has UUID-only filenames; no enumeration
- Capability-URL pattern on `storage://<uuid>` (knowing the UUID
  grants access; the UUID is the secret)
- WAL mode on `history.db` (writers don't block readers — but the
  privacy invariant is still "filter every read by `api_key_hash`")

---

## 11. Compliance summary

### 11.1 Where personal data MAY appear

| Location | Sensitivity | Retention |
|---|---|---|
| `generations.prompt` | High — user-typed; may include names, descriptions, identifiers | 30 days |
| `generations.params_json` (storage://URIs of uploads) | High — points at user reference images / audio | 30 days |
| `generations.enhanced_prompt` | High — Gemma-rewritten, derived from prompt | 30 days |
| `generations.validator_payload_json` (Gemma `reasoning`) | High — LLM-described scene/prompt | 30 days |
| `uploads/<uuid>.*` (reference images, audio, source videos, outputs) | High — user-uploaded media | 30 days (cascaded from `generations.cleanup`) |
| BFF `user_signals.metadata_json` | Medium — event payloads can carry prompt fragments | (BFF retention policy — currently unbounded; flag for review) |
| MCP session JSON cache (`~/.cache/noodlefinger-mcp/sessions/*.json`) | Medium — session state with prompts and clip URIs | Cleared on session expiry; no formal retention |

### 11.2 Where personal data does NOT appear

| Location | Why |
|---|---|
| `generations.api_key_hash` | sha256 — irreversible without bearer |
| `validator_runs.video_sha256` | content hash — derived from output bytes |
| `generations.shot_uuid` / `shot_config_key` | derived hash; reversible only if attacker can guess prompt |
| `generations.prompt_embedding` (when wired) | semantic vector; not directly invertible to prompt without an attack model |
| Logs (journalctl) | careful: prompt strings can leak via WARN/ERROR paths — audit log volume periodically |

### 11.3 Side channels the irreversible hash doesn't protect against

- **Timing**: `created_at` per row + bearer-frequency analysis can
  fingerprint a user's session pattern. Not mitigated.
- **Prompt-content correlation**: If the operator can guess a
  bearer's known prompt (e.g., from a public MV), they can confirm
  that bearer's `shot_uuid` by re-hashing. Mitigated only by
  prompt-content secrecy.
- **Output-content correlation**: `validator_runs.video_sha256` is
  the bytes hash. If an attacker has the MP4, they can confirm
  presence in the cache. Mitigated by upload secrecy (capability
  URLs).

### 11.4 Single-tenant deploy summary (today)

The operator is the sole data subject. The subject and the controller
are the same entity. No GDPR / CCPA right-to-delete obligation
beyond the operator's own preferences. The 30-day retention is a
self-imposed cap, not a regulatory one.

### 11.5 Multi-tenant deploy summary (forward-looking)

External bearers introduce a controller/subject distinction. The
operator becomes the controller; each bearer becomes a separate
subject. Required before flipping:

- Phase D checklist items in [§9](#9-multi-tenant-readiness-checklist)
  closed.
- Counsel review of the Sapiens AUP read.
- BFF `actor_email` no-fallback mode (no `"unknown"` rows).
- Audit endpoints exposed.
- Right-to-delete CLI / endpoint.
- Per-bearer ToS.

---

## 12. Cross-references

- [`docs/CAPTURE_VALIDATOR.md`](CAPTURE_VALIDATOR.md) — validator
  pipeline as a feature.
- [`docs/CAPTURE_VALIDATOR_ROADMAP.md`](CAPTURE_VALIDATOR_ROADMAP.md)
  — Phase B / C / D execution plan; this doc's [§9 checklist](#9-multi-tenant-readiness-checklist)
  is the privacy slice of that roadmap.
- [`docs/API.md`](API.md) — endpoint contracts.
- [`docs/operator-tuning.md`](operator-tuning.md) — env vars
  including `VALIDATOR_VERSION` and rate-limit caps.
- `taco-backend/history_store.py` — schema + `_hash_key` +
  `_maybe_seed_api_key_metadata`.
- `taco-backend/server.py:3642` — `_is_training_opted_in`.
- `taco-backend/server.py:5490` — `_require_admin`.
- `noodlefinger-portal/bff/src/noodlefinger_bff/db.py` — BFF schema
  v2 with `api_key_hash` column.
- `/mnt/nvme-1/servers/sapiens-sidecar/LICENSE_NOTES.md` — Sapiens
  AUP read.
- `/home/ian/.claude/plans/melodic-sniffing-beacon.md` — origin plan,
  including §C training requirements that this doc references.
