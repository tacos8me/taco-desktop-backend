# RATING_LORA_INTEGRATION.md — Client integration guide

This is the public-facing integration contract for any dashboard, CLI, bot,
or extension that wants to wire into the v1.19.0+ rating + exemplar-LoRA
endpoints. It is the canonical contract — the leatherjacket-dashboard is
one realization of it, not the spec.

## Audience

Anyone building a client that wants to:

- Surface validator-flagged borderline clips for human rating
- Submit human ratings (pairwise, taxonomy, warn) into the training corpus
- Curate exemplar sets and trigger LoRA fine-tunes
- Apply real-time recommendation feedback (L1 boosts)
- Adjust per-bearer validator thresholds

Examples of clients: a dashboard like leatherjacket-dashboard, a CLI tool
that shells out to curl, a Discord bot that surfaces clips for ratings in
a channel, a VS Code extension that lets the operator rate from inside
their editor, an MCP tool that LLMs call (with consent gates).

## Authentication

All endpoints require `Authorization: Bearer <api_key>`. Two tiers:

- **Bearer auth**: any valid `.api_keys` entry. Caller can rate, curate
  exemplars, and read their own data.
- **Admin auth**: any valid `.admin_keys` entry. Required for:
  bulk-revalidate, lora-rollback, build-LoRA (cost gate), bearer-table
  reads, threshold edits for OTHER bearers.

`api_key_hash = sha256(bearer)` is the per-bearer identity used in
privacy gates.

## Error envelope

All errors return a uniform JSON envelope:

```json
{
  "error": "<short_token>",
  "error_code": "<TOKEN_UPPER>",
  "message": "human-readable description",
  "detail": { /* optional structured info */ }
}
```

`error` is the short snake_case token (e.g. `privacy_gate_blocked`).
`error_code` is the same token in SCREAMING_SNAKE_CASE for clients that
prefer constants (e.g. `PRIVACY_GATE_BLOCKED`). `message` is
human-readable; `detail` is optional structured context for the client to
surface programmatically.

## Roles

- **owner** = the bearer whose `api_key_hash` matches `rater_api_key_hash`
  on the resource (rating, exemplar set, etc). For ratings: owner = rater
  (who submitted). For exemplar sets: owner = creator. NOT the bearer of
  the rated clip — that's the **subject** (the bearer who owns the clip's
  generation row). The privacy gate enforces both perspectives: rater
  identity (who can write) and subject identity (whose data is read).

## 409 conditions

| `error_code` | When |
|---|---|
| `PRIVACY_GATE_BLOCKED` | rater ≠ bearer AND bearer opted-out of training, OR cross-bearer rating without `cross_bearer_rating_consent_at` set; re-checked on PATCH so a mid-session opt-out flip rejects subsequent edits |
| `RATE_LIMIT_TOO_FAST` | per-rater anti-fatigue gate (plan invariant #4): last rating from same rater < `_RATING_MIN_INTERVAL_S` (2.0 s) ago. Distinct from the 429 token-bucket middleware — see `Rate-limit specifics` below |
| `VALIDATOR_COMPOSITE_NULL` | rated clip has `validator_score IS NULL` (validator never rendered an opinion; rating against it is rejected) |
| `SESSION_CAP_EXCEEDED` | per-rater anti-fatigue gate (plan invariant #4): rater hit `_RATING_SESSION_CAP` (200) ratings within the rolling 90-minute window |
| `IDEMPOTENCE_CONFLICT` | re-rating a consumed pair (`used_in_training_run_id NOT NULL`) — the new rating creates a fresh row but the old `pair_id` is immutable; client must surface `pair_consumed: true` |
| `VALIDATOR_VERSION_MISMATCH` | rated clip's `validator_version` differs from the active queue filter (the queue cursor was generated against a stale validator version; client should refresh) |
| `CLIP_ALREADY_IN_SET` | exemplar member add: `(set_id, clip_id)` already present — idempotency conflict against the unique index |

## 422 conditions (selection)

| `error_code` | When |
|---|---|
| `INVALID_THRESHOLD_INVERSION` | submitted threshold pair has `pass <= retake` (would invert recommendation logic) |
| `INVALID_THRESHOLD_RANGE` | threshold is outside global ± 0.15 clamp |
| `MISSING_VALIDATOR_VISIBLE_FLAG` | rating body missing required `payload.validator_visible_at_rating` (no default by design) |
| `INVALID_KIND` | `kind` not in `pair_chose_a` / `pair_chose_b` / `pair_tie` / `tag` / `warn` |
| `EXEMPLAR_SET_FULL` | exemplar set member count ≥ `max_members` (default 200; admin-overridable per-set) |

## Other exemplar error codes

The exemplar endpoints share the same envelope shape as ratings (Phase 1
polish — `_error()` extended with optional `error_code`):

| `error_code` | Status | When |
|---|---|---|
| `EXEMPLAR_SET_NOT_FOUND` | 404 | `set_id` doesn't exist (or, for owner-gated routes, the caller doesn't own it and no admin override) |
| `EXEMPLAR_OWNER_REQUIRED` | 403 | caller is not the set's owner and not admin |
| `EXEMPLAR_ACCESS_DENIED` | 403 | cross-bearer member add: subject bearer hasn't given training opt-in or `cross_bearer_rating_consent_at` |
| `CLIP_NOT_FOUND` | 404 | `clip_id` referenced by member-add doesn't exist |
| `CLIP_NOT_IN_SET` | 404 | member-remove targets a `(set_id, clip_id)` that isn't a current member |

## Rate-limit specifics

Two independent layers guard the rating endpoints; clients should
distinguish them so retries hit the right cooldown:

1. **HTTP 429 — token-bucket middleware** (broad anti-DDoS).
   Process-local, keyed by `sha256(api_key)`. 10 req/s per key with a
   burst of 10. Applied to `/v1/clips/*/rating*`,
   `/v1/ratings/queue`, and the embeddings endpoints. Returns
   `Retry-After` in seconds. Bypassed when `config.API_KEYS` is empty
   (auth disabled). Bursting beyond capacity returns
   `429 rate_limited`.

2. **HTTP 409 `RATE_LIMIT_TOO_FAST` — per-rater anti-fatigue**
   (plan invariant #4). Backend reads
   `MAX(human_ratings.created_at) WHERE rater_api_key_hash = ?` on each
   POST/PATCH; if the delta is < `_RATING_MIN_INTERVAL_S` (2.0 s) the
   request is rejected. Survives process restart because the state lives
   in the DB. Distinct intent from #1: this catches mash-clicking on a
   real touch UI, not script-driven floods.

3. **HTTP 409 `SESSION_CAP_EXCEEDED` — per-rater session cap**
   (plan invariant #4). Same DB-driven path as #2 but counts non-NULL
   `created_at` rows in the rolling
   `now - _RATING_SESSION_WINDOW_S` (90-minute) window. Hard cap is
   `_RATING_SESSION_CAP` (200). The 90-minute window is rolling, not
   wall-clock — the cap relaxes naturally as the oldest rows age out.

## Endpoints — quick reference

```
# Ratings (L3 + L1)
POST   /v1/clips/{clip_id}/rating          submit/upsert (bearer)
PATCH  /v1/clips/{clip_id}/rating/{rid}    change in place (owner | admin)
DELETE /v1/clips/{clip_id}/rating/{rid}    retract (owner | admin)
GET    /v1/clips/{clip_id}/ratings         audit list (clip's bearer | rater | admin)
GET    /v1/ratings/queue                   active-learning feed: borderline-composite (bearer)

# Threshold overrides (L1)
GET    /v1/api-keys/me/validator-thresholds   read self thresholds (bearer)
POST   /v1/api-keys/me/validator-thresholds   write self thresholds (bearer)

# Recommendations (L1; existing endpoint extended with new param)
POST   /v2/embeddings/recommend-loras         add `session_boosts: {<lora_id>: float}` param

# Exemplar sets + LoRA builds (L2.5)
POST   /v1/exemplar-sets                                           create set (bearer)
GET    /v1/exemplar-sets                                           list caller's sets (bearer)
GET    /v1/exemplar-sets/{set_id}                                  detail + members (owner | admin)
POST   /v1/exemplar-sets/{set_id}/members                          add clip (owner)
DELETE /v1/exemplar-sets/{set_id}/members/{clip_id}                remove clip (owner)
POST   /v1/exemplar-sets/{set_id}/build-lora                       kick off training (owner + admin gate)
GET    /v1/exemplar-sets/{set_id}/builds                           training_runs for this set (owner | admin)
POST   /v1/exemplar-sets/{set_id}/build/{run_id}/cancel            stop a running build (admin)
```

## Rating submission contract

```http
POST /v1/clips/abc-123-def/rating HTTP/1.1
Authorization: Bearer $KEY
Content-Type: application/json

{
  "kind": "pair_chose_a",        // or "pair_chose_b" | "pair_tie" | "tag" | "warn"
  "value": 1.0,                   // [-1.0, 1.0]; pair_chose_a = 1.0, pair_chose_b = -1.0, etc.
  "payload": {
    "pair_partner_clip_id": "xyz-789-other",      // required for pair_chose_*
    "validator_visible_at_rating": false,          // REQUIRED — see safety §P0-3
    "lr_seed": 42,                                // required for pair_chose_*; client computes int.from_bytes(sha256(f"{clip_id}|{pair_partner_clip_id}").digest()[:4], 'big')
    "tags": ["stiff_motion", "hand_glitch"],      // optional taxonomy tags
    "comment": "..."                              // optional freetext (audit-only, never trains)
  }
}

→ 200 OK
{
  "rating_id": 1234,
  "clip_id": "abc-123-def",
  "kind": "pair_chose_a",
  "value": 1.0,
  "pair_id": 42,                              // null for warn/tag-only
  "validator_version": "1.19.0-rc1",
  "validator_composite_at_rating": 0.51,
  "created_at": 1683123456.789,
  "superseded_rating_id": null,                // non-null on upsert
  "pair_consumed": false                       // true if existing pair_id was already in a training run; new rating starts fresh divergence
}
```

**Required fields and why**:

- `kind`: discriminator. Values: `pair_chose_a`, `pair_chose_b`,
  `pair_tie`, `tag`, `warn`. `pair_*` lands in `preference_pairs`; `tag`
  and `warn` are audit-only.
- `value`: normalized strength in [-1.0, 1.0]. For pairwise: chose_a=1.0,
  chose_b=-1.0, tie=0.0.
- `payload.validator_visible_at_rating`: REQUIRED. Pass `false` if your
  client hides the validator score during the rating decision
  (recommended, per §P0-3); pass `true` if shown. Backend filters or
  down-weights pairs where this was true. Lying breaks the safety
  contract.

  **WHY this matters**: showing the validator's composite score to the
  operator before they rate creates anchoring bias. Empirically (per the
  adversarial-safety review's Spearman analysis): anchored ratings
  correlate with the validator at ~0.85, while independent
  (validator-hidden) ratings correlate at ~0.5. Anchored ratings provide
  validator-echo signal, not independent training signal — they teach
  the trained LoRA to mimic the validator, which is a circular reward.
  The backend filters or down-weights pairs where this flag is true at
  preference-pair construction. **If your client always shows the score
  before rating, your ratings will be discounted at training time.**
  Recommendation: blur or hide the score until after the operator
  commits the rating, then reveal as a calibration check (so the
  operator can self-audit "did I agree with the validator?" without the
  score influencing the original decision).
- `payload.pair_partner_clip_id`: REQUIRED for `pair_chose_*` kinds. The
  OTHER clip in the comparison.
- `payload.lr_seed`: REQUIRED for `pair_chose_*`. Client computes
  `int.from_bytes(sha256(f"{clip_id}|{pair_partner}").digest()[:4], 'big')`
  to deterministically randomize L/R rendering. Backend doesn't trust
  this for validation but logs it for audit.

**Errors**:

- 401: missing/bad bearer
- 403: `payload.pair_partner_clip_id` belongs to a bearer that hasn't
  given cross-bearer rating consent AND rater ≠ that bearer
- 404: clip_id doesn't exist
- 409: `clip.validator_score IS NULL` (validator never rendered an
  opinion; rating against it is rejected) OR rate-limit (last rating <
  2s ago) OR session cap exceeded
- 422: schema violation (missing `validator_visible_at_rating`, kind not
  in enum, etc.)

## Exemplar curation contract

```http
# Step 1: create a set (idempotent on set_id)
POST /v1/exemplar-sets HTTP/1.1
Authorization: Bearer $KEY
Content-Type: application/json
{ "set_id": "leatherjacket-v80", "description": "Punk-VHS aesthetic, MK23 character anchor" }
→ 201 Created { "set_id": "leatherjacket-v80", "rater_api_key_hash": "...", "created_at": ... }

# Step 2: add members (idempotent per (set_id, clip_id))
POST /v1/exemplar-sets/leatherjacket-v80/members
{ "clip_id": "abc-123-def", "note": "trash compactor scene; love the grain" }
→ 200 OK { "set_id": "leatherjacket-v80", "clip_id": "...", "added_at": ..., "set_size": 23 }

# Step 3: when set_size >= threshold (default 20), kick off build
POST /v1/exemplar-sets/leatherjacket-v80/build-lora
Authorization: Bearer $ADMIN_KEY              # admin-gated due to GPU cost
{
  "rank": 32,                       // optional; defaults to 32
  "steps": 1500,                    // optional; defaults to 1500. ltx-trainer is step-based (default 2000)
  "learning_rate": 1e-4,            // optional
  "base_model": "ltx-2.3-distilled" // optional; default is current-deployed
  "dry_run": false                  // default false; true validates dataset shape + ETA without GPU
}
→ 202 Accepted
{
  "training_run_id": "run_xyz",
  "expected_eta_hours": 12.5,
  "lora_artifact_path_hint": "/mnt/.../loras/leatherjacket-v80-run_xyz.safetensors",
  "scheduled_for": "2026-05-02T02:00:00Z",   // off-hours window
  "config_snapshot_path": "training_runs/run_xyz/config.yaml"
}

# Step 4: poll for status (or use the existing training_runs panel via /v1/system/training-runs)
GET /v1/exemplar-sets/leatherjacket-v80/builds
→ 200 OK { "builds": [{ "run_id": "run_xyz", "status": "running", "progress": 0.42, "eval_loss": [...], ... }] }
```

## Threshold override contract

```http
GET /v1/api-keys/me/validator-thresholds
→ 200 OK
{
  "pass_threshold": 0.65,                       // null means using global default
  "retake_threshold": 0.45,
  "fallback_pass": 0.65,                        // global value (informational)
  "fallback_retake": 0.45
}

POST /v1/api-keys/me/validator-thresholds
{ "pass": 0.70, "retake": 0.40 }                // can pass null to clear
→ 200 OK { ...new values reflected... }
```

Thresholds clamped to global ± 0.15 (rejected with 422 if outside).
Effective on next validator dispatch.

**Sanity validation**: pass must remain strictly greater than retake. If
both `pass` and `retake` are non-null in the submitted body AND `pass <=
retake`, the endpoint rejects with `422 invalid_threshold_inversion` and
an error envelope explaining the constraint. Inverted thresholds would
invert the validator's recommendation logic (clips below the higher
threshold but above the lower would be recommended for retake while
clearly-bad clips passed) — a silent foot-gun the endpoint must refuse.
Single-knob updates (only `pass` or only `retake` in the body) are
validated against the stored value of the other knob (or its global
fallback if NULL).

## Recommend-loras with session boosts (L1)

```http
POST /v2/embeddings/recommend-loras
{
  "prompt": "trash compactor scene with character anchor",
  "k": 5,
  "session_boosts": {              // optional; client-managed L1 state
    "leatherjacket-v80-run_xyz": 0.05,    // boost a candidate LoRA
    "punk-vhs-old": -0.03                  // suppress a stale one
  }
}
→ 200 OK
{
  "recommendations": [
    { "lora_id": "leatherjacket-v80-run_xyz", "rank_score": 0.78, "mean_validator_score": 0.71, "expected_boost": 0.04, "sample_size": 23 },
    ...
  ],
  "total_samples": 6
}
```

Boosts clamped per-LoRA to [-0.10, +0.10] before applying to ranking.

**`session_boosts` is CLIENT-MANAGED state.** The backend stores nothing
about the boost map; the client passes its current boosts on each
`recommend-loras` request. Boosts are session-local by client
convention; resetting on session end is the client's responsibility.
The backend's only role is to clamp each boost to [-0.10, +0.10] before
applying to the ranking formula. This keeps the contract simple: no
session lifecycle on the server, no expiration windows to reason about,
no race between client and server view of the map.

Practical pattern: a client maintains a dict `{lora_id: boost}` in its
own memory or storage; on every rating click it nudges the entry for
the rated LoRA's `lora_applied_id` (±0.01 per click, daily decay
applied client-side); on every `recommend-loras` call it serializes the
current dict into the request body. Reset = drop the dict.

## Rating mutation contracts (PATCH / DELETE / GET)

```http
PATCH /v1/clips/{clip_id}/rating/{rating_id} HTTP/1.1
Authorization: Bearer $KEY                 # owner OR admin
Content-Type: application/json

{
  "kind": "pair_chose_b",                  // optional; may flip kind, supersedes
  "value": -1.0,                           // optional
  "payload": { ... }                       // optional partial; merged with stored payload
}

→ 200 OK
{
  "rating_id": 1235,                       // NEW row; old rating_id has superseded_by=1235
  "supersedes_rating_id": 1234,
  "pair_id": 43,                           // may be a new pair_id if kind flipped
  "pair_consumed": false,
  "validator_version": "1.19.0-rc1",
  "created_at": 1683123500.0
}

→ 403 not_owner / not_admin
→ 404 rating_not_found
→ 409 IDEMPOTENCE_CONFLICT (old pair already in a training run; new pair_id allocated, old row immutable)
```

```http
DELETE /v1/clips/{clip_id}/rating/{rating_id} HTTP/1.1
Authorization: Bearer $KEY                 # owner OR admin

→ 200 OK
{
  "rating_id": 1234,
  "retracted_at": 1683123600.0,
  "preference_pair_deleted": true,         // false if the pair was already consumed
  "pair_id": 42
}

→ 403 not_owner / not_admin
→ 404 rating_not_found
```

Retraction soft-deletes the `human_ratings` row (sets `retracted_at`)
and DELETEs the corresponding `preference_pairs` row IF it has not been
consumed (`used_in_training_run_id IS NULL`). Already-consumed pairs are
immutable; the rating is retracted but the trained-against artifact is
unchanged. See §"What this plan deliberately does NOT do" — retraction
does not unwind shipped LoRAs.

```http
GET /v1/clips/{clip_id}/ratings HTTP/1.1
Authorization: Bearer $KEY                 # clip's bearer OR rater OR admin

→ 200 OK
{
  "clip_id": "abc-123-def",
  "ratings": [
    {
      "rating_id": 1234,
      "rater_api_key_hash": "a3f9c2..."   // truncated to 8 chars unless caller is admin
        ,
      "kind": "pair_chose_a",
      "value": 1.0,
      "payload": { "pair_partner_clip_id": "...", "tags": ["stiff_motion"], "validator_visible_at_rating": false },
      "validator_composite_at_rating": 0.51,
      "created_at": 1683123456.789,
      "retracted_at": null,
      "superseded_by": null
    },
    ...
  ]
}

→ 403 not_authorized (caller is not the clip's bearer, the rater, or an admin)
→ 404 clip_not_found
```

## Active-learning queue contract

```http
GET /v1/ratings/queue?limit=20&cursor=<opaque_b64> HTTP/1.1
Authorization: Bearer $KEY                 # bearer

→ 200 OK
{
  "items": [
    {
      "clip_id": "abc-123-def",
      "validator_composite": 0.52,
      "validator_version": "1.19.0-rc1",
      "shot_config_key": "leatherjacket-v80-shot-04",
      "thumbnail_uri": "/v2/history/abc-123-def/thumbnail"
    },
    ...
  ],
  "next_cursor": "eyJjbGlwX2lkIjoieHl6In0=",  // base64; null when end-of-stream
  "total_borderline": 47                       // approximate; 0.45 ≤ composite ≤ 0.65 not yet rated by caller
}
```

Items are ordered by `ABS(validator_composite - 0.55) ASC` (closest to
band-center first), with a secondary order on `clip_id` for stable
cursor pagination. Cross-cohort sampling is enforced: at least 30% of
items per page come from `shot_config_key`s the caller hasn't recently
rated (mitigates cohort-collapse reward-hacking, P0-2c).

**Diversity quota implementation.** Server-side, the endpoint pulls a
4×-oversampled distance-ordered candidate window, tags each row with
`is_novel_cohort = (shot_config_key NOT IN <caller's last 100 rated
cohorts>)` via SQL, and merges in Python (Option B in the prereq
write-up): first fills `ceil(min_diversity * limit)` slots from
novel-cohort rows (preserving their distance order), then backfills
the remainder from the full distance-ordered stream. When the corpus
is too cohort-thin to meet the quota the endpoint serves what it has
— degraded diversity, never zero results. The fraction is
configurable via the `RATING_QUEUE_MIN_COHORT_DIVERSITY` env var
(default `0.3`, clamped to `[0.0, 1.0]`); the recent-cohort window
size is fixed at 100 rated clips.

```
→ 401 missing/bad bearer
→ 422 INVALID_CURSOR (cursor doesn't decode or doesn't match the active validator_version)
```

## Exemplar set list / detail contracts

```http
GET /v1/exemplar-sets HTTP/1.1
Authorization: Bearer $KEY                 # bearer (caller's sets only)

→ 200 OK
{
  "sets": [
    {
      "set_id": "leatherjacket-v80",
      "description": "Punk-VHS aesthetic, MK23 character anchor",
      "member_count": 23,
      "max_members": 200,
      "last_built_lora_id": "leatherjacket-v80-run_xyz",
      "last_built_at": 1683100000.0,
      "created_at": 1682500000.0
    },
    ...
  ]
}
```

```http
GET /v1/exemplar-sets/{set_id} HTTP/1.1
Authorization: Bearer $KEY                 # owner OR admin

→ 200 OK
{
  "set_id": "leatherjacket-v80",
  "description": "...",
  "rater_api_key_hash": "a3f9c2...",
  "max_members": 200,
  "members": [
    { "clip_id": "abc-123-def", "added_at": 1682600000.0, "note": "trash compactor scene", "validator_composite": 0.81 },
    ...
  ],
  "builds_summary": { "total": 3, "last_status": "completed", "last_run_id": "run_xyz" }
}

→ 403 not_owner / not_admin
→ 404 set_not_found
```

```http
DELETE /v1/exemplar-sets/{set_id}/members/{clip_id} HTTP/1.1
Authorization: Bearer $KEY                 # owner

→ 200 OK
{
  "set_id": "leatherjacket-v80",
  "clip_id": "abc-123-def",
  "removed_at": 1683200000.0,
  "set_size_after": 22
}

→ 403 not_owner
→ 404 set_or_member_not_found
```

## Build cancel contract

```http
POST /v1/exemplar-sets/{set_id}/build/{run_id}/cancel HTTP/1.1
Authorization: Bearer $ADMIN_KEY           # admin
Content-Type: application/json

{ "reason": "operator dialed back exemplar count; refining set" }

→ 200 OK
{
  "training_run_id": "run_xyz",
  "status": "cancelled",                   // training_runs.status flipped from 'running' → 'cancelled'
  "cancelled_at": 1683210000.0,
  "artifact_dir_removed": true,            // training_runs/run_xyz/ deleted; row preserved for audit
  "reason": "operator dialed back exemplar count; refining set"
}

→ 403 not_admin
→ 404 build_not_found
→ 409 build_already_terminal (status was already 'completed' / 'failed' / 'cancelled')
```

## Polling vs streaming for build status

The plan currently exposes polling via
`GET /v1/exemplar-sets/{set_id}/builds`. SSE streaming is a P2
follow-on; mention as future API surface but not blocked-on for v1.

## Privacy gate cheat-sheet

| Action | Allowed if |
|---|---|
| Rate own clip | always (rater = bearer) |
| Rate another bearer's clip | bearer has `cross_bearer_rating_consent_at` non-NULL AND opted in for training |
| Add foreign clip to exemplar set | same as above (rater must have access; cross-bearer needs consent) |
| Build LoRA from set with foreign clips | same; build orchestrator re-validates at submit time |
| Read another bearer's ratings | admin only |
| Read own threshold | always |
| Read another bearer's threshold | admin only |

**Mid-session edge cases** (the 24h staging window plus opt-in
mutability creates three edges clients must handle):

1. **Bearer opts out mid-session**. Existing ratings (already in
   `human_ratings`) remain audited — retraction is a separate operator
   action. Future ratings against that bearer's clips are REJECTED with
   `409 PRIVACY_GATE_BLOCKED` from the moment the opt-in flips. Active
   in-flight queue items become 404 / 409 on submit.

2. **Operator A rates Bearer B's clip, then Bearer B retracts opt-in**.
   The 24h staging window means the latest 24h of A's ratings against
   B's clips are vulnerable to retraction. On the next
   `construct_human_rating_pairs` run after B's flip, the
   construction-time invariant re-check (§Path C′ invariant #1) finds
   `training_opt_in = 0` for B's clip and DELETEs the staged
   `preference_pairs` row. Older ratings whose
   `pending_construction_until` was already cleared remain in the
   corpus until the operator runs an explicit cleanup;
   `delete_rater_corpus(rater_hash)` and a future
   `delete_subject_corpus(bearer_hash)` cascade can purge them.

3. **Operator A rates Bearer B's clip, then Bearer B grants
   cross-bearer consent retroactively**. Ratings submitted before
   consent was granted but still within the staging window auto-promote
   to the training corpus on the next construction-cron run (the
   construction-time invariant re-check now passes). Ratings rejected
   at write time (because consent wasn't yet granted) are NOT
   recoverable — the backend never created the row, so retroactive
   consent doesn't resurrect them.

## Integration testing recipe

1. Spin up taco-backend at localhost:8090.
2. Set up two test bearers in `.api_keys`: `bearer_A`, `bearer_B`.
   Default both `training_opt_in=1`.
3. Run the integration test suite in your client: validate that all 5
   P0 + 10 stop-the-line invariants from §safety are testable from the
   OUTSIDE (your client should be able to reproduce them).
4. Cross-bearer test: bearer_A submits rating for bearer_B's clip →
   expect 403 unless consent flag set.
5. Validator-anchor test: submit rating with
   `validator_visible_at_rating=true` and
   `validator_visible_at_rating=false`; query
   `/v1/clips/{id}/ratings` and confirm both stored, then check
   `train_dpo_sft.py` SQL filters out the visible-true rows correctly
   (the construction filter is documented in
   `scripts/construct_preference_pairs.py`).

## Common gotchas

- **Forgetting `validator_visible_at_rating`**: 422 error. There's no
  default — clients MUST decide and declare.
- **Re-rating same clip with different kind**: produces a new `pair_id`
  (DELETE old + INSERT new) if old wasn't consumed; otherwise creates a
  new row and the old one is now historical. Audit chain via
  `superseded_by`.
- **Exemplar set with all clips below validator threshold**: build will
  succeed but produce a low-quality LoRA. No automatic gate —
  operator's responsibility. Future: surface a "set quality" warning in
  the UI when median validator_score in set < 0.5.
- **Building too many LoRAs**: storage grows. 5-builds-per-set
  retention default; tune via admin endpoint.
- **Session boosts persisting across sessions**: client bug — boosts
  are session-local by design; reset on session end.

## Reference implementation

The reference client is the **leatherjacket-dashboard** at the
operator's deploy. Source is not currently public. For a working
integration template, request the operator's reference client repo URL
or contact the maintainer. Treat this integration guide (the
`RATING_LORA_INTEGRATION.md` doc) as the canonical contract — the
reference client is one realization of it, not the spec.
