# UX Gaps + Roadmap

**Audience**: future UI/UX engineer + product owner.
**Source versions**: taco-backend `v1.18.0-rc3` · noodle-v `current` · noodlefinger-portal `current` · noodlefinger-bff `v0.2.0`.
**Generated**: 2026-04-29 (Wave-2 docs push).

This is a buildable plan. Section 1 inventories the UI surfaces that exist
today; section 2 maps every Phase-B/C capability to the surface it should live
on (and notes the gap); sections 3-5 are the prioritized build queue (P0/P1/P2)
with rationale, location, tech approach, effort estimate, and acceptance
criteria for each panel. Section 6 sequences the work; section 7 explains the
"which app does this belong in?" decision tree; section 8 captures forward-look
debt that isn't worth scoping yet but should be tracked.

If you only read one section: **§3 (P0)** is the critical path. Everything else
is supporting context.

---

## 1. UI surface inventory (what exists today)

The system has four user-visible surfaces today, plus one API-only service
(BFF) that is intended to grow a UI but currently has none. These are the
*only* places where new UI can land — there is no "fifth dashboard" hiding
somewhere.

| Surface | URL | Stack | Audience | Auth | Source of truth |
|---|---|---|---|---|---|
| **taco-dashboard** | `http://192.168.1.80:8099/dashboard` (LAN-only via `dashboard_server.py` proxy) | vanilla JS SPA, single-file `dashboard.html` (~1970 LOC) | operator (you) | Bearer token in `localStorage` | `dashboard.html` + `/v1/system/*` endpoints on taco-backend |
| **noodle-v** | `http://localhost:5173` (dev, Vite) — local only | React 18 + TypeScript + Tailwind + Zustand | end user (creator) | Bearer in `system-store` | `/mnt/nvme-1/servers/noodle-v/src/` |
| **noodlefinger-portal** | `https://noodlefinger.io` (prod) | React + Babel-in-browser, single-file pages, no build step | end user (consumer / API-customer) | WorkOS session cookie via portal BFF | `/mnt/nvme-1/projects/noodlefinger-portal/src/` |
| **BFF** | `http://127.0.0.1:8002` | FastAPI | API only (no UI) | optional Bearer | `/mnt/nvme-1/servers/noodlefinger-bff/` |

### 1.1 taco-dashboard (`dashboard.html` + `dashboard_server.py`)

LAN-only operator dashboard, served by a 145-line FastAPI proxy
(`dashboard_server.py`) that forwards bearer-authed JSON to taco-backend on
:8090. Single HTML file, ~1970 LOC of vanilla JS — no framework, no build step.
Zinc/oklch dark theme. Sections currently rendered (from
`dashboard.html` `<h2>` walk):

- `GPU Status` — per-GPU VRAM bar / temp / utilization (cuda:0, cuda:1)
- `Model State` — colored badges for LTX / Flux / JoyAI / ACE / Chat
- `Job Queue` — Queued / Processing / Completed / Failed counts + active-job
  progress bar
- `Live Workers` (v1.13.0) — per-worker grid (local + Modal + RunPod), busy /
  idle, current-job summary
- `Controls` — System pause/resume, model unload/reload, Turbo Mode toggle,
  Modal Pool / RunPod Pool count grids, Sampler picker (Euler / CFG++)
- `Advanced` (collapsible) — 14 LTX generation knobs (steps, sigmas, scales,
  eta) persisted to `.gen_config.json`
- `Flux` (collapsible) — 2 Flux-turbo knobs persisted to `.flux_config.json`
- `Quick Generate` — bare-bones t2i / t2v form

**Telemetry cadence**: GPU 2 s, queue 1 s, workers 2 s, pool 5 s.

**What it's good at**: at-a-glance GPU + queue health. Single-pane operator
console.

**What it's *bad* at**: no historical view (no charts, only spot values), no
quality observability (no validator scores), no training-run progress, no
preference-pair counter. The "Quality" axis is entirely absent — this is the
single biggest gap relative to v1.17.0-rc2's validator pipeline ship.

### 1.2 noodle-v (`/mnt/nvme-1/servers/noodle-v/src/`)

The MV authoring frontend. React 18 + TypeScript, Tailwind, Zustand stores
(`generation-store`, `music-store`, `musicvideo-store`, `retake-store`,
`timeline-store`, `system-store`, `dataset-store`). Component layout:

- `components/musicvideo/` — `music-video-workspace`, `keyframe-editor`,
  `keyframe-tray`, `beat-timeline`, `video-timeline`, `strategy-bar`,
  `action-bar`, `results-section`, `song-strip`, `song-drop-zone`,
  `autofill-button`
- `components/timeline/` — `canvas`, `canvas-clip`, `timeline-view`,
  `transition-overlay`, `snap-guide`, `drag-ghost`, `export-button`
- `components/retake/` — `retake-workspace`, `retake-setup`, `retake-compare`,
  `retake-player`, `retake-scrubber`, `retake-progress`, `retake-empty`,
  `retake-mode-selector`, `retake-prompt-bar`
- `components/music/` — music-generation panels
- `components/` — `video-bucket`, `video-clip`, `video-lightbox`,
  `keyframe-strip`, `prompt-bar`, `prompt-optimizer`, `media-upload`,
  `approved-feed`, `ernie-modal`, `trigger-word-menu`, `turbo-toggle`,
  `header`, `api-key-gate`, `settings-row`

**What it's good at**: end-to-end MV authoring (song → shot list → clips →
timeline → export). Retake flow is mature.

**What it's *bad* at**: no awareness of validator scores on clips (every clip
in the bucket / timeline is shown identically — no quality affordance), no
visualization of the retake provenance graph (you can retake a clip 5 times
but you can't *see* the variant tree), no way to surface "the LoRA we'd
recommend for this prompt" even though `/v2/embeddings/recommend-loras` exists.

### 1.3 noodlefinger-portal (`/mnt/nvme-1/projects/noodlefinger-portal/src/`)

The public-facing API portal. React + Babel-in-browser, no build step (`.jsx`
files served as text and transpiled at runtime via `tweaks-panel.jsx` host
protocol). Sidebar navigation defined in `shell.jsx`:

```
OVERVIEW      Dashboard
BUILD         API docs · Models (badge: 6) · API keys · MCP server (NEW) · Webhooks (NEW)
WORKSPACE     Usage · Team (badge: 7) · Audit log
YOU           Account · Support
```

Pages in `src/`: `dashboard.jsx`, `data.jsx`, `data-real.jsx`, `docs.jsx`,
`gallery.jsx`, `mcp.jsx`, `pages-1.jsx`, `pages-2.jsx`, plus
`mcp-landing/`. Auth via WorkOS cookie; bearer credentials stay server-side at
the portal BFF.

**What it's good at**: docs / API browsing / billing / team management.
Gallery view of past generations.

**What it's *bad* at**: no per-clip validator score in the gallery (it's all
in `history_store` but never surfaced), no compositions page (a user can't
look up "MV I exported on 2026-04-21 — what 14 clips made it in?"), no opt-in
toggle for `api_key_metadata.training_opt_in` (the table exists per rc1 schema
but there's no UI), no drift / quality-trend visualization, no "your data
contributed N preference pairs to the next training run" pat-on-the-back.

### 1.4 BFF (`/mnt/nvme-1/servers/noodlefinger-bff/`)

FastAPI service on `127.0.0.1:8002`. Holds `user_signals` table (gallery
downloads / exports / MCP events) per `CAPTURE_VALIDATOR.md` §4.4. Portal proxies
through it. **No UI**. Any new visualization that wants to lean on user-signal
telemetry needs (a) a new BFF endpoint and (b) a portal page to render it.

---

## 2. Capability → UI mapping (the gap table)

Every Phase-B and Phase-C capability that ships in v1.17.0-rc1+rc2 (and is
slated for the rest of the
melodic-sniffing-beacon plan) gets a row. Backend surface is the source of
truth; "Current UI" is what's surfaced *today*; "Gap" is what a user / operator
*can't see or do* despite the data being there. Priority: P0 = required for
Phase B/C usability (not having this blocks the operator), P1 = 4-week
quality-of-life, P2 = polish.

| Capability | Backend Surface | Current UI | Gap | Priority |
|---|---|---|---|---|
| Per-clip validator score | `generations.validator_score` + `validator_payload_json` (rc1 schema; populated by rc2 dispatch) | NONE | operator can't see quality distribution; can't filter clips by score; can't spot a regression after a config change | **P0** (CRITICAL) |
| A/B test in-progress stats | `training_runs` rows + `lora_registry` ab-experiment metadata | NONE | when Phase C activates, operator can't see "candidate LoRA score 7.4 vs baseline 7.1, n=312, p=0.04" live; can't pause / abort an experiment from a UI | **P0** (BLOCKER) |
| Retake provenance graph | `generations.parent_clip_id` chain + `shot_uuid` group | NONE | user can't see "I made 5 retakes of clip-3; this one's the keeper"; tree exists in data, never rendered | **P0** (BLOCKER) |
| Composition lineage view | `composition_clips` inverted-index table (rc1) | NONE | user has 50 historical clips, can't tell which 14 ended up in the final MV (export_handler writes the lineage but no read-side UI) | **P0** (BLOCKER) |
| Training run progress (50 hr) | `training_runs.eval_metrics_json` + `status` ladder | NONE | a 50-hour training run is invisible — operator has to `tail` server logs; no ETA, no loss curve | **P0** (BLOCKER) |
| Preference pair counter | `preference_pairs` table | NONE | operator's Phase-C go/no-go is "do we have ≥1000 pairs?" — needs to be a glanceable badge, not a SQL query | **P0** (BLOCKER) |
| Validator dispatch failure alerts | log lines from `_dispatch_validator` + `/v1/system/metrics` (TBD endpoint) | NONE | RAFT crash / Sapiens unreachable / Gemma judge schema-violation are silent until you grep journalctl | **P1** (MISSING) |
| Bearer opt-in toggle | `api_key_metadata.training_opt_in` | NONE | external bearers can't toggle their own opt-in/out; operator has to INSERT manually | **P1** (IMPORTANT) |
| LoRA recommendation sidebar | `/v2/embeddings/recommend-loras` (Phase-B endpoint) | NONE | embedding-based "you'll probably want this LoRA" is computed but never shown to the user authoring a prompt | **P1** (NICE-TO-HAVE) |
| Drift detection (validator scores trending down) | derived: rolling mean of `generations.validator_score` over time, group by model/lora/config | NONE | a config regression that drops mean score 0.78 → 0.62 is invisible until users complain | **P1** (NICE-TO-HAVE) |
| Similar-shots visual gallery | `/v2/embeddings/search` (Phase-B endpoint) returns IDs only | JSON only (devtools) | a "find shots like this one" UI doesn't exist; operators with 5k clips can't browse semantically | **P2** (POLISH) |
| GPU-hour cost per MV | derived: sum of `generations.gpu_seconds` (TBD column) per `composition_id` | NONE | "this 32-clip MV cost $4.20 in GPU time" — feeds pricing decisions; not yet exposed | **P2** (POLISH) |
| User-signal telemetry dashboard | `user_signals` table on BFF (`download` / `export` / `share` events) | NONE | engagement intel exists per row, never aggregated into a chart | **P2** (POLISH) |
| Active-learning UI (borderline thumbs) | derived: clips where `0.45 ≤ validator_score < 0.55` | NONE | the validator's "warn" band is exactly where human label is most valuable; we should be asking the user | **P2** (POLISH) |
| Multi-tenant org-id scoping | not yet in schema | NONE | every UI today is single-tenant; growth path needs org-id scoping front-and-center | future / not yet prioritized |

The table is intended to be exhaustive across the rc1+rc2 surface area — each
Phase-B/C capability shows up in one row. New rows added when new capabilities
ship.

---

## 3. P0 — required for Phase B/C usability

Ordered by criticality within P0. The cumulative effort is the critical path
(see §6). Each item lists rationale, location (which app / where in that app),
tech approach, effort, and acceptance criteria.

### 3.1 Validator Score Dashboard (taco-dashboard)

- **Rationale**: operator can't see per-clip validator scores; quality gates
  are invisible. v1.17.0-rc2 ships a 3-tier validator that writes
  `validator_score` on every completed video, and *nothing* in the UI surfaces
  this. A config change that quietly drops mean composite from 0.78 to 0.62 is
  silent until users complain.
- **Location**: `dashboard.html`, new `<section><h2>Quality</h2></section>`
  inserted between `Live Workers` and `Controls`.
- **Tech**: vanilla JS (match existing dashboard style — no new dependency).
  New backend endpoint `GET /v1/system/validator-stats?window=24h` returning
  `{histogram: [{bucket, count}], mean, p50, p95, by_version: {<v>: {mean, n}},
  recent_failures: [...]}`. Frontend fetches every 5 s, renders an inline SVG
  histogram (10 buckets, 0.0-1.0). Click a bucket → modal listing the worst-N
  clips in that bucket with `result_uri` thumbnails. Per-validator-version
  breakdown shown as a horizontal mini-bar (catches version-skew drift).
- **Effort**: 2 days (1 day backend endpoint + tests, 1 day frontend SVG +
  modal).
- **Acceptance**:
  - [ ] histogram renders the 100 most recent validator runs in <300 ms
  - [ ] per-version breakdown shows mean + n for each `validator_version`
        present in the window
  - [ ] threshold-highlighted clips: bucket [0.0, 0.45) is red, [0.45, 0.65)
        amber, [0.65, 1.0] green (matches `validator.py` recommendation bands)
  - [ ] click-to-drill-in: bucket click opens a modal with thumbnails sourced
        from `/v2/history/{id}/thumbnail`

### 3.2 A/B Test Cockpit (taco-dashboard)

- **Rationale**: when Phase C lights up, an A/B run is "candidate LoRA vs
  baseline LoRA, dispatch alternates by hash, mean validator score on each arm
  is the metric." Operator currently has zero visibility — has to query the
  DB. Without this, you cannot run an experiment with confidence; you cannot
  pause / abort one without restarting the server.
- **Location**: `dashboard.html`, new `<section><h2>Experiments</h2></section>`
  below `Quality`. Modal for full detail.
- **Tech**:
  - Backend: new endpoint `GET /v1/system/ab-status` → `{active: [{id, name,
    candidate, baseline, candidate_n, baseline_n, candidate_mean,
    baseline_mean, p_value, started_at, status}], history: [...]}`. Computes
    Welch's t-test on `generations.validator_score` filtered by
    `lora_applied_id`. New endpoint `POST /v1/system/ab-status/{id}/abort` to
    halt an experiment (sets status=aborted; new dispatches stop routing to
    candidate arm).
  - Frontend: 1-row-per-active-experiment table with sparkline
    (`candidate_mean - baseline_mean` over time), p-value tooltip
    (Bonferroni-corrected if multiple active), pause / resume / abort buttons.
- **Effort**: 3 days (2 days backend incl. stats + dispatch hook, 1 day
  frontend).
- **Acceptance**:
  - [ ] live mean ± std-err for each arm, refreshing every 5 s
  - [ ] p-value tooltip shows the test used and df
  - [ ] abort button immediately halts new candidate dispatches; in-flight
        candidate jobs complete and contribute to final stats
  - [ ] history table shows resolved experiments with final n, mean, p-value,
        outcome (winner / null / aborted)

### 3.3 Retake Provenance Graph (noodle-v)

- **Rationale**: user retakes a clip 5 times to land the right one. Today the
  bucket shows 5 sibling tiles with no relationship — you can't see "clip-3-v4
  is the parent of clip-3-v5". The data is in `parent_clip_id`. UI is missing.
- **Location**: noodle-v, `components/video-bucket.tsx` clip-inspect side
  panel. New `<RetakeTree clipId={...} />` React component sourced from
  `components/retake/retake-tree.tsx` (new file).
- **Tech**: React component. Calls existing `GET /v2/history?shot_uuid=<u>` to
  fetch siblings; walks `parent_clip_id` chain locally to build a tree.
  Rendered as a vertical D3-force-style tree (or simpler: nested CSS-flex
  cards). Click a node → loads that clip into the active retake-compare view.
  Visual indicator on the "winner" node (the one used in the active timeline /
  composition).
- **Effort**: 2 days (1.5 day component + state, 0.5 day integration with
  retake-workspace).
- **Acceptance**:
  - [ ] given a clip with N retakes, the tree renders all N+1 nodes connected
        by `parent_clip_id`
  - [ ] selecting a node sets it as the "active" variant in `retake-store`
  - [ ] the winner (currently-on-timeline) node has a distinguishing border /
        badge
  - [ ] empty state (no retakes yet) renders cleanly without an error

### 3.4 Composition Lineage View (noodlefinger-portal)

- **Rationale**: a user exports a 32-clip MV. Three weeks later they want to
  know "which clips made the cut and which didn't?" The
  `composition_clips` inverted-index has the answer — it's written
  best-effort by `POST /v2/compositions/{id}/export`. There is no read-side
  UI.
- **Location**: noodlefinger-portal, new sidebar item under WORKSPACE:
  `Compositions`. New page `compositions.jsx` matching the existing portal
  page conventions (Babel-in-browser, single-file, follows `gallery.jsx`
  scaffolding).
- **Tech**:
  - Backend: new endpoint `GET /v1/compositions/{comp_id}/clips` →
    `{composition_id, exported_at, total_clips, clips: [{id,
    historyId|storage_uri, thumbnail_url, validator_score, kept,
    deprecated_reason}]}`. `kept` = "in current export"; `deprecated_reason`
    = `"replaced_by_retake" | "trimmed_in_edit" | null`.
  - BFF: passthrough proxy with cookie auth swap.
  - Frontend: list-view page. Each composition is a card with title +
    timestamp + thumbnail strip; click → detail page with full clip list +
    validator score badges. Filter: kept / deprecated / all.
- **Effort**: 2.5 days (1 day backend + tests, 0.5 day BFF, 1 day portal page).
- **Acceptance**:
  - [ ] all of the user's compositions list, paginated
  - [ ] detail page shows every clip in `composition_clips` for the comp,
        flagged kept vs deprecated
  - [ ] thumbnail strip resolves both `historyId` and `storage_uri` clip
        references (matches v1.16.3 export-handler behavior)
  - [ ] validator score badge on each clip (sourced from
        `generations.validator_score`)

### 3.5 Training Run Progress Monitor (taco-dashboard)

- **Rationale**: 50-hour training runs need progress monitoring. Operator
  currently `tail`'s server logs and visually parses `eval_metrics_json` line
  noise. There is no ETA, no loss curve, no "this run is regressing — abort
  it."
- **Location**: `dashboard.html`, new
  `<section><h2>Training</h2></section>` below `Experiments`.
- **Tech**:
  - Backend: new endpoint `GET /v1/system/training-runs` → list of recent
    rows from `training_runs` table; parses `eval_metrics_json` blob into
    `{step, train_loss, eval_loss, lr}` time series.
  - Frontend: table with one row per run (`id`, `name`, `status`, `started`,
    `progress %`, `eta`, `loss sparkline`). Click → modal with full
    train_loss + eval_loss line chart (inline SVG, ~100 data points), config
    JSON tab, abort button.
- **Effort**: 3 days (1.5 day backend incl. eval-metrics parser + tests, 1.5
  day frontend chart).
- **Acceptance**:
  - [ ] in-flight runs visible with current step / total steps + computed ETA
  - [ ] loss curve sparkline renders inline in the row
  - [ ] modal shows full train+eval loss curves with hover tooltips
  - [ ] abort button on in-flight runs (writes status=aborted; trainer polls
        and shuts down cleanly)

### 3.6 Preference Pair Counter (taco-dashboard)

- **Rationale**: Phase-C go/no-go gate is "do we have ≥1000 preference pairs
  with `signal_source ∈ {retake, gallery_download, mcp_chosen}`?" Operator
  needs to glance, not query.
- **Location**: `dashboard.html` header — small badge next to `taco-backend
  v1.2 — GPU Dashboard` title. Click → modal with breakdown.
- **Tech**:
  - Backend: new endpoint `GET /v1/system/preference-pairs-count` →
    `{total, by_source: {retake: N, gallery_download: N, mcp_chosen: N},
    last_24h, last_7d}`.
  - Frontend: badge component with count; pulse-green when count crosses 1000.
    Modal shows breakdown + sparkline of pairs/day.
- **Effort**: 1 day (0.5 day backend, 0.5 day frontend).
- **Acceptance**:
  - [ ] header badge renders count, refreshes every 30 s
  - [ ] click opens breakdown by `signal_source`
  - [ ] crossing 1000 visually pulses (CSS animation, dismissable)
  - [ ] sparkline shows pairs/day for the last 14 days

---

## 4. P1 — 4-week quality-of-life

Not blocking Phase B/C, but the operator + creator experience is meaningfully
worse without these. Build queue: 4-week sprint following P0 completion.

### 4.1 Validator Dispatch Failure Alerts

- **Rationale**: `_dispatch_validator` is fire-and-forget. RAFT crash, Sapiens
  unreachable, Gemma schema-violation — all silent. Operator finds out via
  customer complaint or grep'ing journalctl.
- **Location**: `dashboard.html`, integrate into `Quality` section (§3.1).
  Failures roll up as a red ticker above the histogram.
- **Tech**: backend endpoint `GET /v1/system/validator-failures?window=1h` →
  list of `{job_id, error_class, error_message, timestamp, video_uri}`.
  `_dispatch_validator`'s exception handler writes a row to a new
  `validator_failures` table (lightweight; truncated to last 1000 rows).
  Frontend: dismissable banner; badge count + "View all" link to modal list.
- **Effort**: 1.5 days.
- **Acceptance**: any tier-1/2/3 exception surfaces in the dashboard within
  10 s; modal shows full traceback (truncated to 1 KB); dismissable per-row.

### 4.2 Bearer Opt-In Toggle (portal Account page)

- **Rationale**: external bearers can't toggle their own
  `training_opt_in`. Operator has to manually
  `INSERT INTO api_key_metadata`. This is an opt-in/opt-out for the user's
  data being used to train future LoRAs — it must be self-serve.
- **Location**: noodlefinger-portal, `Account` page (`pages-2.jsx` or new
  `account.jsx`). New section: "Training data".
- **Tech**:
  - Backend: new endpoints
    - `GET /v1/api-keys/me/training-opt-in` → `{opted_in: bool, since: ts}`
    - `POST /v1/api-keys/me/training-opt-in` body `{opted_in: bool}` →
      writes `api_key_metadata` upsert
  - BFF: passthrough.
  - Frontend: clear toggle + plain-language explainer ("Your clips will be
    used to train future quality-improvement models. We never use your audio
    or your prompts; only the generated video and the validator score. Opt
    out at any time.") + link to `PRIVACY_GOVERNANCE.md` (when written).
- **Effort**: 1.5 days.
- **Acceptance**: toggle reflects current state; flipping persists across
  reloads; toggling off stops new dispatches within 1 minute (the
  `_is_training_opted_in` cache TTL).

### 4.3 LoRA Recommendation Sidebar (noodle-v)

- **Rationale**: `/v2/embeddings/recommend-loras` (Phase-B retrieval endpoint)
  computes "given this prompt's embedding, here are the 5 LoRAs whose corpus
  is most similar." Today nothing surfaces it. The user authoring a prompt
  for "synth-pop neon city night drone" should *see* the suggestion.
- **Location**: noodle-v, `components/prompt-bar.tsx` — collapsible sidebar
  to the right of the prompt input.
- **Tech**: React component, debounced (700 ms) call to
  `/v2/embeddings/recommend-loras` on prompt change. Renders top-5 LoRAs as
  cards with similarity score + "Apply" button (which `set_adapters`'s the
  store to attach the LoRA at strength 1.0).
- **Effort**: 1.5 days.
- **Acceptance**: typing a prompt populates suggestions within 1 s of pause;
  click-Apply attaches the LoRA; "Why this?" tooltip shows the top-3
  most-similar past prompts from the corpus.

### 4.4 Drift Detection Sparkline (noodlefinger-portal)

- **Rationale**: a regression that drops mean validator score from 0.78 to
  0.62 over a week is invisible. Need a 30-day rolling-mean sparkline,
  per-bearer, with a "your average is X% below the global median" callout.
- **Location**: noodlefinger-portal, Dashboard page (`dashboard.jsx`) — new
  card "Quality trend".
- **Tech**: backend endpoint `GET /v1/quality/trend?window=30d` →
  `{daily_mean: [...30...], my_mean, global_mean, median}`. Inline SVG
  sparkline with horizontal global-median line.
- **Effort**: 2 days.
- **Acceptance**: sparkline renders 30-day rolling window of the logged-in
  bearer's mean validator score; global-median line visible; tooltip on
  hover shows day + n + mean.

---

## 5. P2 — polish / nice-to-have

Tracked but not on the critical path. Build when the team has a slow week or a
specific user complaint surfaces them.

### 5.1 Similar-Shots Gallery (visual browse)

- **Rationale**: with 5k+ clips in `history.db`, semantic browse beats
  keyword. `/v2/embeddings/search` returns IDs but nothing renders them as a
  gallery.
- **Location**: noodlefinger-portal, new sub-page under `gallery.jsx`:
  "Similar shots". Right-click any clip → "Find similar" → grid view.
- **Tech**: existing endpoint; portal-side React grid; thumbnails via existing
  `/v2/history/{id}/thumbnail`.
- **Effort**: 2 days. Acceptance: 24 most-similar clips render in a 6×4 grid;
  clicking a clip opens lightbox.

### 5.2 GPU-hour Cost Tracker

- **Rationale**: feeds pricing + capacity decisions. "This 32-clip MV cost
  $4.20" is a useful number both internally and as a future user-visible
  receipt.
- **Location**: taco-dashboard, new card under `Quality`. Aggregate
  `generations.gpu_seconds` (TBD column — needs schema add) by
  `composition_id` and apply per-GPU $/hr config.
- **Tech**: schema-additive column on `generations`; new endpoint; new
  dashboard card.
- **Effort**: 2 days incl. schema migration. Acceptance: per-MV cost visible;
  per-day GPU-hour totals chart.

### 5.3 User Signal Telemetry Dashboard

- **Rationale**: BFF's `user_signals` table has download / export / share
  events. Aggregating them into "which clips are users actually shipping?"
  is intel that should drive validator threshold tuning.
- **Location**: noodlefinger-portal, Usage page extension OR new operator-only
  page.
- **Tech**: BFF endpoint to aggregate `user_signals` by event type / time
  bucket. Portal renders bar chart.
- **Effort**: 2 days. Acceptance: 30-day chart of download / export / share
  counts; click-through to a specific day's events.

---

## 6. Recommended Implementation Order

P0 critical path (the P0 set is the *minimum* for Phase B/C usability):

```
Week 1 ─────────────────────────────────────────
3.6 Preference Pair Counter            (1d)  ← warmup, low-risk, unblocks confidence
3.1 Validator Score Dashboard          (2d)  ← biggest single visibility win
3.3 Retake Provenance Graph (noodle-v) (2d)  ← independent track, parallelizable

Week 2 ─────────────────────────────────────────
3.4 Composition Lineage (portal)       (2.5d)  ← independent of week-1 work
3.5 Training Run Progress              (3d)   ← prereq: 3.1 stats endpoint pattern
3.2 A/B Test Cockpit                   (3d)   ← depends on 3.1 stats endpoint
                                              ← depends on 3.5 training infra

Week 3-4 ──────────────────────────────────────
4.x P1 work (4 items, ~6.5d)

Week 5+ ───────────────────────────────────────
5.x P2 polish as time / signal warrants
```

**Critical path**: 1 + 2 + 2 + 2.5 + 3 + 3 = **13.5 working days ≈ ~2.5 weeks**
at 1 FTE. With one full-stack engineer parallelizing the noodle-v track
(3.3) against the dashboard track (3.1, 3.2, 3.5, 3.6), it collapses to
**~2 weeks**.

**Dependency graph**:

- 3.1 (Validator Score Dashboard) → blocks 3.2 (A/B Cockpit) — same stats
  endpoint pattern; new histogram component reused.
- 3.5 (Training Run Progress) → blocks 3.2 — A/B cockpit reads `training_runs`
  to surface candidate-LoRA metadata.
- 3.3 (Retake Provenance) → independent.
- 3.4 (Composition Lineage) → independent (portal track).
- 3.6 (Preference Pair Counter) → independent, smallest, lowest-risk —
  warm-up task.

Sequence-of-record: **3.6 → 3.1 → 3.3 (parallel) → 3.4 (parallel) → 3.5 →
3.2**.

---

## 7. UI Architecture Recommendations

When you're deciding "which surface does this panel live on?", the heuristic is
**audience first, technology second**.

### 7.1 Decision tree: which app does this panel belong in?

```
Is this panel for the operator-only (you, ops, infra)?
├─ YES → taco-dashboard
│         (LAN-only; vanilla JS; no auth complexity beyond bearer in
│          localStorage; ship a panel in hours by editing one HTML file)
│
└─ NO → Is it for a creator authoring an MV?
        ├─ YES → noodle-v
        │         (React + TS + Tailwind; deep state via Zustand; the
        │          authoring loop *is* the product; component library
        │          mature; add to the existing component tree)
        │
        └─ NO → Is it for an end user managing their account / API
                usage / past work?
                ├─ YES → noodlefinger-portal
                │         (public on the open internet; WorkOS-cookie
                │          auth via portal BFF; Babel-in-browser, no
                │          build step — adds friction for complex
                │          components but keeps deploys trivial)
                │
                └─ NO → does it need a UI at all? Reconsider.
```

### 7.2 Per-panel placement (concrete)

| Panel | App | Why |
|---|---|---|
| Validator Score Dashboard | taco-dashboard | operator-only quality-ops; LAN-only is fine; vanilla JS keeps the dashboard one file |
| A/B Cockpit | taco-dashboard | operator-only experiment control; pause/abort buttons should not be exposed externally |
| Training Run Progress | taco-dashboard | 50-hour runs are operator-side; abort button is privileged |
| Preference Pair Counter | taco-dashboard | aggregate operator metric; not user-facing intel |
| Validator Dispatch Failures | taco-dashboard | infra alerting; same surface as Quality |
| Retake Provenance Graph | noodle-v | creator-facing during authoring; deep state already lives in `retake-store` |
| LoRA Recommendation Sidebar | noodle-v | creator-facing during prompt authoring |
| Composition Lineage View | portal | end-user "where did my work go"; surfaces past compositions across sessions |
| Bearer Opt-In Toggle | portal | account-level setting |
| Drift Detection Sparkline | portal | per-user trend |
| Similar-Shots Gallery | portal (extension of gallery.jsx) | semantic browse over historical work |
| GPU-hour Cost Tracker | taco-dashboard (operator) + portal (user-facing receipt) | dual-surface — operator sees raw, user sees per-MV total |
| User Signal Telemetry | portal Usage page | engagement intel for the user; could also have an operator view |

### 7.3 Tech approach per surface

**taco-dashboard** — keep it vanilla JS. The single-file constraint is a
*feature*: any operator can `vim dashboard.html`, restart the proxy, ship a
fix. Resist the temptation to introduce React. Use inline SVG for charts
(histogram / sparkline / loss curve); the existing styling vocabulary is
sufficient.

**noodle-v** — keep using Zustand; new components go under `components/` in
the appropriate sub-folder. New stores only when state genuinely outlives a
component tree. For graph rendering (retake provenance), prefer plain
nested-flex CSS over D3 — the trees are small (≤16 nodes).

**portal** — Babel-in-browser is the constraint. New pages follow `pages-1`
/ `pages-2` / `gallery` conventions: single `.jsx` file, top-level component
exported via the host protocol, no build step. The BFF mediates all
authentication — never call taco-backend directly from the portal.

---

## 8. Long-tail UI debt (forward-look)

Things to track but not yet prioritized. Revisit at each version cut.

- **Multi-tenant org-id-based UI scoping**: every UI today is single-tenant.
  Growth path needs org-id scoping front-and-center across all three
  surfaces — workspace switcher, audit-log filtering, per-org training
  opt-in. Schema-additive; new `organizations` table; every existing scoping
  helper needs a `org_id` parameter. Estimated: 2-week sprint when
  multi-tenancy becomes business-critical.
- **Active-learning UI (borderline scores → user thumbs-up/down)**: when
  the validator says `warn` (composite ∈ [0.45, 0.65]), the human label is
  worth 100× the certain ones. Surface these clips in noodle-v or portal
  with a thumbs-up/down ask. Feeds back into preference pairs at higher
  signal-quality.
- **Mobile-friendly portal**: noodlefinger-portal is desktop-first. As the
  user base grows, mobile gallery + composition browsing becomes
  table-stakes. Tailwind already responsive; the constraint is
  Babel-in-browser bundle size on mobile networks.
- **i18n**: zero strings extracted. When the second non-English-speaking user
  shows up, this becomes urgent. Start with a `locales/` JSON convention in
  each app; defer until then.
- **Real-time job stream in noodle-v**: today noodle-v polls every 2 s. SSE
  endpoint `/v2/jobs/{id}/stream` already exists (v1.1.7). Migrating noodle-v
  to SSE would cut backend load by ~10× during heavy authoring sessions.
- **Operator audit log**: who toggled turbo, who flipped opt-in? No log
  today. Becomes important once a second human is doing ops.
- **Validator score on gallery thumbnails (portal)**: small badge in the
  corner of each thumbnail. Tiny scope but high-density value.
- **Phase-D self-serve LoRA training UI**: when end users get to train their
  own LoRAs, the training-run UI from §3.5 becomes user-facing — needs a
  "your training" view in portal with quota, ETA, abort, download.
- **Webhooks panel** (already in nav as NEW badge): not yet wired —
  webhooks for `validator_failed`, `composition_exported`,
  `training_run_complete` would be high-leverage for power users.
- **Dark/light theme toggle in portal**: portal is always light; noodle-v
  and dashboard are always dark. Decide on a global stance (the user's
  global pref is dark/Zinc) and unify.

---

**End**. Sections 1-2 are the audit; section 3 is the build queue; section 4
is the next sprint; section 5 is polish. Implementation lead should start at
3.6 and work through §6's sequence-of-record.
