# Capture + Validator Machine

**Versions covered**: taco-backend `v1.17.0-rc1`/`-rc2`/`-rc3` · noodlefinger-mcp `v0.7.0` · noodlefinger-bff `v0.2.0`
**Plan of record**: `/home/ian/.claude/plans/melodic-sniffing-beacon.md`
**Status (2026-04-29)**: schema + dispatch + endpoint shipped; tier-2 Sapiens runs in stub mode (`LOAD_SAPIENS=0` default); `validator_artifacts/` directory configured but not yet populated; DPO pair construction + training cron not yet wired.

This document is the canonical reference for the capture+validator subsystem. `CLAUDE.md` describes how to work in the code; this describes the feature: what it is, how it flows, the operator knobs, the privacy story, and the forward-looking surface for extension.

---

## 1. Why this exists

### 1.1 The thesis: the validator is the labeler

The user has a 32-clip music-video pipeline. Until v1.16 the loop was open: a clip was generated, the user eyeballed it, retook it by hand if it was bad. Three weeks of that session was hand-debugging one specific failure mode (the "first half static, then motion" defect, traced back to segment-pin propagation in v0.4.6 of the MCP and uniform audio slicing in v0.5.0).

The capture+validator machine **closes the loop**. Every clip generation now produces:

- A **validator score** (RAFT optical-flow + Sapiens pose stability + Gemma vision judge → composite + recommendation), computed in the same pass that gates the user's "this clip is bad → retake" decision.
- A **shot identity tuple** (`shot_uuid`, `shot_config_key`, `parent_clip_id`) that lets us group attempts at the same shot across resumes and turn (kept, retake) into chosen-vs-rejected DPO pairs.
- **User signals** (gallery downloads, exports, MCP orchestrator events) teed into the BFF's `user_signals` table.

The strategic frame: every clip the operator ships today is a labeled positive. Every retake is a labeled (chosen, rejected) preference pair. The 32-clip MVs the operator is shipping are training data for the next LoRA we fine-tune; we just need the data infrastructure. **That data infrastructure is what shipped in v1.17.0-rc1 + rc2 + rc3 + MCP v0.7.0 + BFF v0.2.0.**

### 1.2 What problem it solves

The v0.4.6 "static-in-first-half then motion" debug was 3 weeks of human attention on a defect class that, in retrospect, RAFT optical flow detects automatically (the `dynamic_degree` percentile + `flow_windows[4]` contour catches it cleanly). Tier-3 Gemma judge with `JUDGE_PROMPT_V1` is explicitly tuned for the same defect: "Watch for: static-in-first-half then motion (the v0.4.6 defect we fixed), identity drift, prompt-output mismatch."

The validator pipeline turns the manual eyeball-and-retake loop into an automated detect-score-recommend loop, with a hard escape hatch (`on_failure="warn"`) for when the operator wants visibility without a closed-loop retake.

---

## 2. Architecture overview

### 2.1 Three components, three repos

| Repo | Role | Storage |
|---|---|---|
| `taco-backend` | Validator pipeline, schema v3, on-complete dispatch, `/v2/video/analyze-motion` endpoint | `history.db` (SQLite WAL) — `generations` (12 new cols), `composition_clips`, `validator_runs`, `preference_pairs`, `training_runs`, `api_key_metadata` |
| `sapiens-sidecar` | Tier-2 pose stability service at `127.0.0.1:8096` (currently stub) | none — stateless |
| `noodlefinger-mcp` | Active validation hook in `cut_music_video` orchestrator + MCP→BFF event tee | session JSON (`current_state.clip_quality`) |
| `noodlefinger-bff` | `user_signals` table + `POST /api/mcp/events` ingest | portal SQLite (`user_signals` table) |

### 2.2 Three-tier validator (taco-backend `validator.py`)

```
                 ┌────────────────────────────────────────┐
                 │  validator.run_all_tiers()             │
                 │  (cache lookup → dispatch → composite) │
                 └────────────────────────────────────────┘
                          │           │           │
        ┌─────────────────┘           │           └────────────────┐
        ▼                             ▼                            ▼
┌────────────────┐         ┌────────────────────┐        ┌────────────────────┐
│ Tier 1: RAFT   │         │ Tier 2: Sapiens    │        │ Tier 3: Gemma      │
│ (in-process,   │         │ (sidecar :8096,    │        │ judge (chat_mgr,   │
│  cuda:0)       │         │  stub-mode now)    │        │  llama-swap)       │
│                │         │                    │        │                    │
│ raft_small     │         │ POST              │        │ 5 keyframes →      │
│ optical flow   │         │ /v1/analyze-pose  │        │ multimodal LLM     │
│ ~150 ms/clip   │         │ stub: {stub:true} │        │ ~2-4 s/clip        │
└────────────────┘         └────────────────────┘        └────────────────────┘
        │                             │                             │
        ▼                             ▼                             ▼
   dynamic_degree              pose_temporal_*                judge_score,
   flow_windows[4]             (or stub flag)                 verdict, hint
   motion_smoothness                  │                             │
        │                             │                             │
        └─────────────────┬───────────┴─────────────────────────────┘
                          ▼
                ┌──────────────────────┐
                │ composite() →        │
                │ 0.4·t1 + 0.2·t2      │
                │ + 0.4·t3             │
                │                      │
                │ recommendation:      │
                │  pass / warn / retake│
                └──────────────────────┘
```

### 2.3 Two capture pathways

```
                   ┌───────────────────────┐
                   │ User generates a clip │
                   │ (any video JobType)   │
                   └───────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────────────┐           ┌───────────────┐
        │  PASSIVE      │           │   ACTIVE      │
        │  on_complete  │           │ MCP hook      │
        │  callback     │           │ (cut_music_   │
        │  (always-on,  │           │  video,       │
        │  best-effort) │           │  opt-in)      │
        └───────────────┘           └───────────────┘
                │                           │
                │ fire-and-forget           │ synchronous (with budget)
                ▼                           ▼
        validator.run_all_tiers()   /v2/video/analyze-motion
                │                           │
                └─────────────┬─────────────┘
                              ▼
                  validator_runs cache
                  (sha256, version) → payload
                              │
                              ▼
                  generations.validator_score
                  generations.validator_payload_json
                  generations.validator_version
```

The cache layer means **the active hook (synchronous) and the passive hook (fire-and-forget) cooperate**: passive runs first asynchronously after job completion, populates the row keyed by `(video_sha256, validator_version)`; active hits the cache instantly when MCP calls `/v2/video/analyze-motion` ~50 ms later from `_run_quality_validation`. The sha256 stream-hashes the file (1 MiB chunks) so large clips don't balloon RAM.

### 2.4 BFF user_signals + MCP→BFF event tee

```
   noodlefinger-mcp (Claude Code subprocess)
   └── Session.append_event(event, data)  [SCHEMA_VERSION=1]
       └── _tee_event_fire_and_forget(payload)
           └── if NOODLEFINGER_BFF_URL:
               POST {bff}/api/mcp/events  (timeout=2.0s, swallow errors)

   noodlefinger-bff
   └── /api/mcp/events handler (anonymous-tolerant)
       └── user_signals_log(db, signal_type="mcp_event", target_id=session_id, metadata={...})
           └── INSERT INTO user_signals VALUES (...)
```

---

## 3. Component reference

### 3.1 Validator pipeline (`taco-backend/validator.py`)

#### Tier 1 — RAFT optical flow (in-process, cuda:0)

`_run_tier1_raft(video_path)` — torchvision `raft_small` lazy-loaded, evicted after each call so LTX reclaims cuda:0 cleanly.

- **Decode**: PyAV at the clip's native fps (no resampling), downsampled to 256-wide via PIL bilinear. CPU tensor `(T, 3, H, W)` uint8.
- **Forward**: per-pair (frame[i], frame[i+1]) through `raft_small`'s 12-iteration refinement; final flow magnitude is `sqrt(flow_x² + flow_y²)`, mean-reduced per pair.
- **Outputs**:
  - `dynamic_degree`: top-5% percentile of mean flow magnitudes across all pairs.
  - `flow_windows[4]`: clip split into 4 equal time-windows, mean flow per window. The "static first half" defect class shows up as `flow_windows[0..1] ≈ 0` while `[2..3] >> 0`.
  - `motion_smoothness`: `1 / (1 + var(diff(per_pair_flow)))`. Penalizes jerky motion changes.
  - `latency_s`: wall-clock.
- **VRAM lifecycle**: model dropped + `torch.cuda.empty_cache()` in the executor's `finally` block. Weights download lazily from pytorch.org on first run (~22 MB, cached at `~/.cache/torch/hub/checkpoints/`). RAFT-small was chosen over RAFT-large because the validator runs on every completed video; the +1 GB transient VRAM and ~50 ms/frame difference matter at 28-job concurrency. One-line switch in `_run_tier1_raft` if accuracy regresses.
- **CPU fallback**: not implemented. RAFT on CPU is >30 s/clip; we'd rather degrade tier1 to `None` and let composite collapse to a partial score.
- **Failure mode**: any exception → `tier1=None`. `composite()` notes `tier1_failed`.

#### Tier 2 — Sapiens pose stability (sidecar :8096)

`_run_tier2_sapiens(video_path)` → `sapiens_client.analyze_pose(video_path)`.

- **Stub-tolerant**: if the response is `{"stub": true, ...}` (the rc2 ship state), the client returns it verbatim; the validator wraps it as `{"status": "stub", "tier2_skipped": True, "raw": {...}, "latency_s": ...}`. Composite scoring treats this as **0.2·1.0** in the tier2 slot — no penalty, no boost.
- **Real failure**: any `SapiensError` (unreachable / 5xx / timeout) → tier2=None. Composite notes `tier2_unreachable`.
- **Sidecar-disabled** (`LOAD_SAPIENS=0`, the rc2/rc3 default): tier2 is short-circuited as `{"tier2_skipped": True, "status": "load_disabled", "latency_s": 0.0}`. No HTTP call is made. Same composite arithmetic as the stub path.
- **Forward-stable schema**: when rc-final lands real Sapiens-2 inference, the response gains `human_detected: bool`, `pose_temporal_variance: float`, `identity_drift_frames: int[]`, and `keypoints: [(T, 308, 3)]`. `validator.composite` already reads `pose_temporal_stability` (preferred) or `pose_temporal_variance` → `1/(1+var)` (fallback) so the rc-final swap is a drop-in.

#### Tier 3 — Gemma judge (chat_manager, `CHAR_VISION_MODEL`)

`_run_tier3_judge(chat, video_path, prompt, tier1_summary, tier2_summary)`.

- **Keyframe sampling**: `_sample_keyframe_indices(num_frames, count=5)` picks `{0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1}` — first / quartile / middle / 3-quartile / last. Deterministic per clip → cache-friendly across replays.
- **Frame extraction**: reuses `history_store._extract_frames_as_pils` (PyAV), encoded as JPEG q=80, base64-inlined as `data:image/jpeg;base64,...` blocks.
- **Multimodal request**: system prompt `config.JUDGE_PROMPT_V1`, user message is text (prompt + tier1/tier2 summary) + 5 image_url blocks. Sent via `chat.generate_chat_completion(messages=..., temperature=0.1, max_tokens=512, model=config.CHAR_VISION_MODEL)`.
- **Schema validation**: response is regex-extracted (`r"\{[\s\S]*\}"`) and validated against `JudgeResponseV1` (Pydantic):

```python
class JudgeResponseV1(BaseModel):
    verdict: Literal["pass", "warn", "retake"]
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=4000)
    retake_hint: str | None = None
```

- **Failure modes** (any of these → `tier3` returns a fallback dict, never raises):
  - LLM unreachable / timeout → `{error: "judge_call_failed: ...", judge_score: 0.5, verdict: "warn"}`.
  - Schema violation / parse error → `{error: "schema_violation: ...", judge_score: 0.5, verdict: "warn"}`.
  - Frame-count probe fails → `{error: "frame_count_failed: ...", verdict: "warn"}`.
  - No frames decoded → `{error: "no_frames", verdict: "warn"}`.

#### `JUDGE_PROMPT_V1` (verbatim from `config.py`)

```
You are a video-quality judge for AI-generated music-video clips. Given:
- The original prompt the user wrote
- Optical-flow summary (tier1) showing motion magnitude over time
- Pose temporal-stability summary (tier2, may be skipped)
- 3-5 keyframes from the generated clip

Decide if the clip's actual motion matches the prompt's motion intent.
Watch for: static-in-first-half then motion (the v0.4.6 defect we fixed),
identity drift, prompt-output mismatch.

Return STRICT JSON: {"verdict": "pass"|"warn"|"retake", "score": 0.0-1.0,
"reasoning": "...", "retake_hint": "specific prompt rewrite suggestion or null"}
```

#### Composite scoring

`validator.composite(tier1, tier2, tier3) → {composite_score, recommendation, reasoning_summary}`.

Formula:

```
composite = 0.4·tier1_norm + 0.2·tier2_stability + 0.4·tier3_score
where:
  tier1_norm   = min(dynamic_degree / 5.0, 1.0)   # empirically healthy clips at 256-wide sit in 1-10 range
  tier2_stab   = pose_temporal_stability ∈ [0,1]  # or 1/(1+pose_temporal_variance)
                 OR 1.0 when stub / unreachable / LOAD_SAPIENS=0
  tier3_score  = judge_score ∈ [0,1]
```

Recommendation:

```
if tier3.verdict == "retake":      recommendation = "retake"   # verdict-level override
elif composite >= 0.65:            recommendation = "pass"
elif composite >= 0.45:            recommendation = "warn"
else:                              recommendation = "retake"
```

When tier1 or tier3 is missing, composite collapses to a partial weighted score (`raw / weight_total`) and `reasoning_summary` flags `tier1_failed` / `tier3_failed`.

#### Caching

`validator_runs` table keyed UNIQUE on `(video_sha256, validator_version)`. SHA-256 is computed via streaming reads (1 MiB chunks). Bumping `config.VALIDATOR_VERSION` (currently `"1.17.0-rc2"`) forces re-runs of every clip on next access. Cache hits inject `cached: true` into the returned payload.

### 3.2 Schema v3 additions (`history_store.py`)

`CURRENT_SCHEMA_VERSION = 3`. Migration is single-startup, idempotent, additive.

#### New columns on `generations` (all nullable)

| Column | Type | Purpose | Populated by |
|---|---|---|---|
| `validator_score` | REAL | Composite score (0..1) for fast SQL filtering | `_dispatch_validator` UPDATE after `run_all_tiers` |
| `validator_payload_json` | TEXT | Full per-tier payload JSON (re-readable from `/v2/history/{id}`) | same |
| `validator_version` | TEXT | Pinned version for cache lookup | same |
| `parent_clip_id` | TEXT | Retake provenance — points at the row this retake supersedes | `/v2/retake` handler via `find_id_by_result_uri(body.video_uri)` |
| `shot_uuid` | TEXT | Cross-session shot identity — `sha256(prompt ⌷ image_uri ⌷ position)[:16]` | MCP orchestrator `_apply_shot_lineage` (forwards in body); backend strips via `_HISTORY_ONLY_PARAMS` and threads to `history.save()` |
| `shot_config_key` | TEXT | Full DPO pair-matching hash (sha256 over prompt + image_uri + audio_start_s + duration_s + model + lora_id + lora_strength) | same |
| `composition_id` | TEXT | Forward-looking — denorm convenience to skip the `composition_clips` join | not yet wired (composition export uses `composition_clips` instead) |
| `lora_applied_id` | TEXT | Which LoRA actually fused at runtime | not yet wired (TODO — currently NULL) |
| `lora_applied_strength` | REAL | LoRA strength at fusion time | same |
| `prompt_embedding` | BLOB | 3584-dim float32 embedding from Gemma | not yet wired (forward-looking — Phase B retrieval) |

Plus three indexes: `idx_gen_shot_config_key`, `idx_gen_parent_clip_id`, `idx_gen_composition_id`.

#### New tables

```sql
CREATE TABLE IF NOT EXISTS composition_clips (
    comp_id TEXT NOT NULL,
    clip_history_id TEXT,            -- NULL for synthetic flash inserts (storage_uri only)
    position INTEGER NOT NULL,
    was_final INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    PRIMARY KEY (comp_id, clip_history_id, position)
);
CREATE INDEX IF NOT EXISTS idx_comp_clips_clip ON composition_clips(clip_history_id);
```

Inverted-index of clips included in each composition export. Written by `history.record_composition_clips(comp_id, clips)` from `POST /v2/compositions/{id}/export` (server.py:5514) before submitting the export job. Best-effort — `INSERT OR IGNORE` so re-exports don't explode on the PK. Flash inserts (synthetic clips with `storage_uri` only, per v1.16.3) get a row with `clip_history_id=NULL`.

```sql
CREATE TABLE IF NOT EXISTS validator_runs (
    run_id TEXT PRIMARY KEY,
    video_uri TEXT,
    video_sha256 TEXT,
    payload_json TEXT,
    latency_s REAL,
    validator_version TEXT,
    ran_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_validator_runs_video_version
    ON validator_runs(video_sha256, validator_version);
```

Cache-by-content for the validator pipeline. UNIQUE ensures concurrent runs of the same clip return the same row (last-write-wins via `INSERT OR REPLACE` in `_validator_runs_persist`).

```sql
CREATE TABLE IF NOT EXISTS preference_pairs (
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chosen_clip_id TEXT,
    rejected_clip_id TEXT,
    signal_source TEXT,         -- "user_retake"|"validator"|"composition_kept"|"explicit_rating"
    signal_strength REAL,
    used_in_training_run_id TEXT,
    created_at REAL
);
```

**Forward-looking — no writer wired in v1.17.0-rc3.** The Phase C DPO pair construction job will join `generations` (grouped by `shot_config_key`) with `composition_clips` and `parent_clip_id` to emit (chosen, rejected) rows.

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

**Forward-looking — no writer wired.** The Phase C training cron will INSERT here after each DPO LoRA training run, and `lora_registry_id` will FK back to the existing `lora_registry.json` once the new LoRA is staged.

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

Per-key training opt-in. **Seeding** (one-shot, on first v2→v3 migration): `_maybe_seed_api_key_metadata()` reads `.api_keys`, hashes each non-comment line with `_hash_key`, INSERTs with `training_opt_in=1`. The seed only runs when `api_key_metadata` is empty — `.api_keys` itself is never modified. New external bearers added to `.api_keys` after the seed get **opt-out by default** until explicitly INSERTed.

#### Migration mechanics

```python
def _migrate(self) -> None:
    current = self._conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= CURRENT_SCHEMA_VERSION:
        return
    if current < 3:
        # 11 ALTER TABLE adds (idempotent via try/except OperationalError)
        # 3 CREATE INDEX
        # 5 CREATE TABLE IF NOT EXISTS (composition_clips, validator_runs, ...)
        # _maybe_seed_api_key_metadata()
    self._conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
```

Single-startup. WAL mode means readers don't block during the ALTER. Re-running after a crash mid-migration is safe — each statement is `IF NOT EXISTS` or guarded.

### 3.3 New endpoint: `POST /v2/video/analyze-motion`

Synchronous. Lives at `server.py:4503-4540`.

**Auth**: Bearer required when `config.API_KEYS` is set; mirrors `/v2/video/extract-frames`.

**Request**:

```json
{
  "video_uri": "storage://b1c0d4e8-...",
  "prompt": "wide shot of a dancer pirouetting under neon lights",
  "shot_uuid": "a3f9c2e1d8b740a6",         // optional, 16-hex truncated sha256
  "tiers": ["raft", "sapiens", "gemma"],   // optional subset; default = all
  "validator_version": "1.17.0-rc2"        // optional override; defaults to config.VALIDATOR_VERSION
}
```

**Response (200)**:

```json
{
  "video_uri": "storage://b1c0d4e8-...",
  "validator_version": "1.17.0-rc2",
  "tier1": {
    "dynamic_degree": 4.821,
    "flow_windows": [4.12, 4.93, 5.41, 4.83],
    "motion_smoothness": 0.72,
    "latency_s": 0.148
  },
  "tier2": {
    "tier2_skipped": true,
    "status": "load_disabled",
    "latency_s": 0.0
  },
  "tier3": {
    "verdict": "pass",
    "score": 0.81,
    "reasoning": "Consistent motion across all four windows; pose stable; matches prompt's pirouette intent.",
    "retake_hint": null,
    "judge_score": 0.81,
    "latency_s": 2.341
  },
  "composite_score": 0.732,
  "recommendation": "pass",
  "reasoning_summary": "tier2_stub",
  "ran_at": 1730240000.123,
  "latency_s": 2.501,
  "cached": false
}
```

**Latencies** (single-clip, no cache, 5-second clip @ 24fps):
- Tier-1 RAFT: ~150 ms (lazy-load amortizes over many calls; ~1 s on first call after restart).
- Tier-2 stub: <10 ms; real Sapiens (rc-final) projected ~500 ms.
- Tier-3 Gemma judge: ~2-4 s (multimodal LLM via llama-swap).
- Cache hit: <50 ms (sha256 stream + SQLite SELECT).

**Failure modes**:
- `404 video_not_found` — `video_uri` doesn't resolve via `uploads.resolve(...)`.
- `401 Missing API key` — bearer required but absent (only when `API_KEYS` is set).
- `503` — none directly raised by this endpoint; tier-2 sidecar errors are absorbed inside `_run_tier2_sapiens` (returns `None` → composite degrades).
- `500 analyze_motion_failed: <exc>` — orchestration-level exception (history writer fails, etc.). Tier failures themselves never bubble up.

### 3.4 Passive validator dispatch (`server.py` `_on_job_complete`)

The `worker_loop` `on_complete(job)` callback (added in v1.8.2 / SEC P1-3) now chains through `_decr_queue_on_complete → _on_job_complete → _dispatch_validator`.

#### Eligibility (server.py:3455-3476)

```python
def _on_job_complete(job: Job) -> None:
    if job.status != JobStatus.COMPLETED:               # FAILED / CANCELLED skipped
        return
    if job.type not in _VALIDATOR_VIDEO_TYPES:
        return                                           # non-video jobs skipped
    if not _is_training_opted_in(job.api_key or ""):
        return                                           # opt-out bearers skipped
    asyncio.create_task(_dispatch_validator(job))        # fire-and-forget
```

`_VALIDATOR_VIDEO_TYPES = {TEXT_TO_VIDEO, IMAGE_TO_VIDEO, AUDIO_TO_VIDEO, RETAKE, VIDEO_OUTPAINT, VIDEO_HDR}`.

The wrapping `_decr_queue_on_complete` swallows exceptions from `_on_job_complete` so a validator failure never affects the per-key queue counter release.

#### `_dispatch_validator` mechanics

1. Resolve `job.result_uri` → on-disk path via `uploads.resolve(...)`.
2. Pull the clip prompt from `job.params["prompt"]` or `job.raw_request["prompt"]`.
3. `await validator.run_all_tiers(video_uri, video_path, prompt, chat, history)`.
4. UPDATE the matching `generations` row (by `id == job.id`) with `validator_score` + `validator_payload_json` + `validator_version`. **This is the only writer to those columns.**
5. Best-effort. Any exception logs WARN and returns; the queue worker has already moved on.

**Why fire-and-forget**: a 3-4 s tier-3 judge call would otherwise block the per-key counter decrement, throttling the next job. The cost: validator scores show up in `/v2/history/{id}` ~3 s after the result_uri appears. MCP's active hook (next section) can wait if it cares.

### 3.5 MCP `quality_validation` hook (`orchestrator.py`)

Active validation, opt-in per `cut_music_video` call. Lives in `_run_quality_validation(session, store, clip_idx)` (orchestrator.py:2248-2359).

#### Schema (orchestrator.py:415-477, normalized via `_normalize_quality_validation`)

```python
quality_validation: {
  enabled: bool,                       # default false; when false, byte-identical to v0.6
  max_retakes_per_clip: int,           # default 1; range [0, 3]
  tiers: ["raft", "sapiens", "gemma"], # default = all three
  on_failure: "warn" | "retake",       # default "warn"
  motion_intent_map: {                 # optional per-clip motion-intent override
    "0": "static_portrait",
    "5": "fast_action",
    ...
  },
  apply_retake_hint: bool,             # default false; if true, append tier3.retake_hint to the shot's prompt
  validator_version: str,              # optional; defaults to "1.17.0-rc2"
}
```

#### Hook timing

Synchronous call, runs after tail-extraction in `_run_clip_step` and after a re-resume in `resume_music_video` (orchestrator.py:2089, 2200).

#### Retake budget logic (`_maybe_retake_from_payload`, orchestrator.py:2362-2440)

```
if on_failure == "retake" AND payload.recommendation == "retake":
    used = clip.quality_retakes (default 0)
    budget = qv.max_retakes_per_clip
    if used >= budget:
        emit "validator_retake_budget_exhausted" event; return False (accept clip)
    clip.quality_retakes = used + 1
    clip.job_id = None
    clip.status = "pending"            # re-submit on next pass
    if qv.apply_retake_hint AND tier3.retake_hint:
        shot.prompt = f"{base} [retake: {hint}]"
    emit "validator_retake_triggered" event
    return True
```

When `on_failure="warn"` (default), the score is recorded but no retake fires — the clip is accepted as-is. This is the "visibility without closed-loop intervention" mode.

#### Cache for resume idempotence

`current_state["clip_quality"]` keyed by `str(clip_idx)`. On `resume_music_video(..., revalidate=False)`, cached scores short-circuit; on `revalidate=True`, the cache is bypassed and tiers re-run for every clip (intended for "I bumped `validator_version` and want fresh scores").

#### `quality_telemetry` block in result

When `quality_validation.enabled=true`, the final `cut_music_video` payload gains a `quality_telemetry` block (orchestrator.py:703-739):

```json
{
  "quality_telemetry": {
    "enabled": true,
    "validator_version": "1.17.0-rc2",
    "tiers": ["raft", "sapiens", "gemma"],
    "on_failure": "warn",
    "max_retakes_per_clip": 1,
    "per_clip": [
      {"clip_idx": 0, "composite_score": 0.83, "recommendation": "pass", "quality_retakes": 0},
      {"clip_idx": 1, "composite_score": 0.41, "recommendation": "retake", "quality_retakes": 1},
      ...
    ],
    "composite_score_avg": 0.71,
    "total_retakes": 3,
    "validation_overhead_s": 47.2
  }
}
```

When `enabled=false` (or absent), the block is absent — byte-identical to v0.6 callers.

### 3.6 Shot lineage forwarding (orchestrator.py + server.py)

Two stable hashes ride in every clip body so taco-backend can group attempts at the same shot:

- **`shot_uuid`** (16-hex): `sha256(prompt ⌷ image_uri ⌷ position)[:16]`. Cross-session shot identity. Two attempts with the same prompt + image + position get the same `shot_uuid` regardless of session.
- **`shot_config_key`** (full sha256): `sha256(prompt + image_uri + audio_start_s + duration_s + model + lora_id + lora_strength)`. **DPO pair-matching key** — two attempts with the same `shot_config_key` are pair-eligible (chosen vs rejected) for downstream training. Differs from `shot_uuid` by including the full generative config (model, LoRA, audio window).

Computed by `_apply_shot_lineage(body, ...)` (orchestrator.py:494-517) on every video-job submission from `cut_music_video`. Backend handles them via `_HISTORY_ONLY_PARAMS` (server.py:353-356):

```python
_HISTORY_ONLY_PARAMS = (
    "parent_clip_id", "shot_uuid", "shot_config_key",
    "composition_id", "lora_applied_id", "lora_applied_strength",
)
```

Three dispatch sites (`_dispatch_job`, `_dispatch_job_turbo`, `_dispatch_job_turbo_remote`) call `_strip_history_params(job.params)` so manager kwargs stay clean. `worker_loop`'s history.save() picks them out of `job.params` and persists them on the row.

Forward-and-backward compatible: older backends (pre-v1.17.0-rc1) silently ignore unknown body fields via Pydantic `extra="ignore"`.

### 3.7 BFF user_signals + MCP→BFF event tee

#### `user_signals` table (`bff/db.py`)

```sql
CREATE TABLE IF NOT EXISTS user_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,                -- ISO8601 UTC
  actor_email TEXT NOT NULL,        -- "unknown" when bearer-less
  signal_type TEXT NOT NULL,        -- e.g. "mcp_event", "gallery.download", "export.complete"
  target_id TEXT,                   -- session_id for mcp_event; clip_id / comp_id elsewhere
  metadata_json TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON user_signals(ts DESC);
CREATE INDEX IF NOT EXISTS idx_signals_actor ON user_signals(actor_email, ts DESC);
```

Distinct from the existing `audit` table (operator/admin actions). Signals are **training-data inputs and product-engagement metrics**.

#### `POST /api/mcp/events` ingest endpoint (`bff/routers/mcp_events.py`)

Anonymous-tolerant. Accepts a single event or a batch:

```json
{
  "session_id": "06b3a9f2-...",
  "event_kind": "validator_result",
  "event_data": {"clip_idx": 5, "composite_score": 0.71, "recommendation": "pass"},
  "ts": 1730240000.123,
  "actor_email": null
}
```

Auth fallback chain: `X-User-Email` header → `actor_email` from payload → literal `"unknown"`. Future hardening: bind a portal-issued bearer at MCP install time.

Returns `202 Accepted` with `{"accepted": N}`.

#### MCP-side tee (`mcp/_session.py`)

`Session.append_event(event, data)` calls `_tee_event_fire_and_forget(payload)` — `asyncio.create_task` of an httpx POST with a 2 s timeout. Fully gated on `NOODLEFINGER_BFF_URL` env var; unset → no-op.

```python
def _tee_event_fire_and_forget(payload: dict[str, Any]) -> None:
    if not _bff_url():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return                          # no loop (unit test) → silent skip
    loop.create_task(_send_event_to_bff(payload))
```

Failures log WARN and swallow; orchestrator correctness must not depend on BFF reachability. The tee captures every state transition the orchestrator emits — `clip_submitted`, `clip_completed`, `validator_result`, `validator_retake_triggered`, `validator_skipped`, `composition_exported`, etc. — giving the BFF a per-user signal stream without coupling MCP latency to BFF availability.

### 3.8 Sapiens sidecar (stub mode)

Lives at `/mnt/nvme-1/servers/sapiens-sidecar/`. Mirrors the `madmom-sidecar` template: separate venv (no shared deps with taco-backend), FastAPI app on `127.0.0.1:8096`, systemd user unit `sapiens-sidecar.service` ordered `Before=taco-backend.service`.

- **v1.17.0-rc1/rc2/rc3 status: STUB.** `service.py:120-163` returns a canned payload:

```json
{
  "keypoints": [],
  "confidence": [],
  "frame_count": 0,
  "model_version": "sapiens2-pose-0.4b",
  "stub": true,
  "stub_reason": "v1.17.0-rc1: real inference deferred to rc2 once validator dispatch is proven",
  "target": "/dev/null",
  "latency_s": 0.0001
}
```

Schema is forward-stable: rc-final (the next rc swap) replaces the stub with PyAV decode + per-frame Sapiens-2 forward pass. `keypoints` becomes `(T, 308, 3)` and `confidence` becomes `(T, 308)`. The `stub` flag flips to `false`. No client-side changes required beyond removing the stub guard (which already exists in `validator.composite`).

- **License**: cleared for **internal-use only** per operator attestation in `LICENSE_NOTES.md` (2026-04-29). Sapiens-2 ships under Meta's custom AUP; clause §1.b.vi.ii (`for biometric processing`) is potentially blocking but the operator's narrower internal-use read is defensible: synthetic input only, no real-person identification, no SaaS, no external bearers. **Reversibility**: if scope ever changes (multi-tenant, external bearers, SaaS), revisit this attestation with counsel before continuing. Recommended substitutes if the read tightens: DWPose (Apache-2.0) or ViTPose (Apache-2.0).

- **Known limitations** (per `SETUP.md`):
  - Unit uses `Type=exec`; `/health` readiness is NOT a startup gate. Operator deployment validation should always include an external `/health` poll.
  - Cross-unit ordering (`Before=taco-backend.service`) requires the consumer side to also declare `After=sapiens-sidecar.service`. Currently in place; if either side is removed, ordering silently becomes a no-op.

---

## 4. Operator runbook

### 4.1 Enabling the passive validator (LOAD_SAPIENS flip)

The validator pipeline already runs on every completed video job — but tier-2 returns the stub payload until the operator flips `LOAD_SAPIENS=1`.

**Minimum-impact trial run**:

1. The schema migration runs on first restart after `v1.17.0-rc1` was deployed (already done if you're reading this in production).
2. Verify schema v3:
   ```bash
   sqlite3 /mnt/nvme-1/servers/taco-backend/history.db "PRAGMA user_version"
   # → 3
   ```
3. Verify the seed:
   ```bash
   sqlite3 /mnt/nvme-1/servers/taco-backend/history.db "SELECT COUNT(*), SUM(training_opt_in) FROM api_key_metadata"
   # → (N, N) where N == non-comment lines in .api_keys
   ```
4. Generate a video and check:
   ```bash
   sqlite3 /mnt/nvme-1/servers/taco-backend/history.db "SELECT id, validator_score, validator_version FROM generations WHERE validator_score IS NOT NULL ORDER BY created_at DESC LIMIT 5"
   ```
   Should show non-NULL scores for any opt-in bearer's recent video jobs.

**Flip to real Sapiens** (when rc-final lands):

1. `bash /mnt/nvme-1/servers/sapiens-sidecar/setup.sh` — installs the venv, downloads weights, registers + starts the unit.
2. Edit `taco-backend/.env` to add `LOAD_SAPIENS=1`.
3. `systemctl --user restart taco-backend`.
4. Verify:
   ```bash
   curl http://127.0.0.1:8096/health
   # → {"ready":true}
   curl -X POST http://127.0.0.1:8096/v1/analyze-pose -H 'Content-Type: application/json' -d '{"video_path":"/dev/null"}'
   # → with stub: true (rc1/rc2/rc3) or real keypoints (rc-final)
   ```
5. Inspect `validator_runs`:
   ```bash
   sqlite3 history.db "SELECT video_uri, validator_version, ran_at FROM validator_runs ORDER BY ran_at DESC LIMIT 5"
   ```

### 4.2 Activating MCP `quality_validation`

```bash
uvx --refresh noodlefinger-mcp@v0.7.0
```

In your `cut_music_video` call, pass:

```json
{
  "music_prompt": "...",
  "shot_list": [...],
  "quality_validation": {
    "enabled": true,
    "max_retakes_per_clip": 1,
    "on_failure": "warn"
  }
}
```

The result will include `quality_telemetry` (see [3.5](#35-mcp-quality_validation-hook-orchestratorpy)).

For closed-loop retakes:

```json
{
  "quality_validation": {
    "enabled": true,
    "max_retakes_per_clip": 2,
    "on_failure": "retake",
    "apply_retake_hint": true
  }
}
```

`apply_retake_hint=true` mutates the shot's prompt by appending `[retake: <tier3.retake_hint>]` before re-submission. **Caveat**: this silently rewrites user prompts in-session. Audit the `validator_retake_triggered` events to see what was applied.

### 4.3 Privacy gate management

**Default policy**: single-tenant deploy, ON globally for seeded keys, opt-OUT for new external bearers.

**Flip a bearer to opt-out**:

```bash
sqlite3 /mnt/nvme-1/servers/taco-backend/history.db
> UPDATE api_key_metadata SET training_opt_in=0, updated_at=strftime('%s','now')
  WHERE api_key_hash = '<sha256-of-key>';
```

To compute `api_key_hash` for a known bearer:

```python
import hashlib; hashlib.sha256("the-bearer-token".encode()).hexdigest()
```

**Add a new external bearer with opt-in**:

```bash
echo "new-bearer-token" >> /mnt/nvme-1/servers/taco-backend/.api_keys

# Then explicitly INSERT to opt them in (the seed only runs once on v2→v3 migration):
sqlite3 history.db "INSERT INTO api_key_metadata (api_key_hash, training_opt_in, tier, created_at, updated_at) VALUES ('<sha256>', 1, 'pro', strftime('%s','now'), strftime('%s','now'))"
```

**Audit who's opted in**:

```bash
sqlite3 history.db "SELECT api_key_hash, training_opt_in, tier FROM api_key_metadata"
```

### 4.4 Turbo-mode coordination

`sapiens-sidecar` is in both `_stop_cuda1_tenants()` and `_restore_cuda1_tenants()` (server.py:1894, 1918). Both branches gated on `config.LOAD_SAPIENS` (rc3 fix).

When `LOAD_SAPIENS=0` (default), turbo entry skips the spurious `systemctl stop sapiens-sidecar` and turbo exit skips the spurious `systemctl start`.

When `LOAD_SAPIENS=1`, turbo entry calls `systemctl stop sapiens-sidecar` and `_wait_cuda1_free` confirms cuda:1 drains; turbo exit re-starts the sidecar via `systemctl start`. Same pattern as ACE / JoyAI / ERNIE.

**Edge case**: turbo entry may fail in `_wait_cuda1_free` (timeout). When that happens, `_restore_cuda1_tenants()` runs as rollback before raising — so `sapiens-sidecar` comes back up automatically. No operator intervention needed.

### 4.5 Validator artifact retention

`config.VALIDATOR_ARTIFACTS_DIR = /mnt/nvme-1/servers/taco-backend/validator_artifacts/` (overridable via env). **Forward-looking**: the directory is configured but no validator currently writes per-clip artifacts (the `validator_artifact_uri` column on `generations` is also unwired, NULL for all rows). Phase B candidate: rc-final Sapiens may dump per-frame keypoints there as `.npz` for downstream temporal-stability re-analysis.

When wired, cleanup will extend `history.cleanup()`'s 30-day retention sweep — same lifetime as result MP4s and thumbnails.

---

## 5. Telemetry / observability

### Per-clip

```sql
SELECT id, type, prompt, validator_score, validator_version,
       json_extract(validator_payload_json, '$.recommendation') AS recommendation,
       json_extract(validator_payload_json, '$.tier3.verdict') AS verdict,
       json_extract(validator_payload_json, '$.tier3.retake_hint') AS hint
FROM generations
WHERE validator_score IS NOT NULL
ORDER BY created_at DESC
LIMIT 50;
```

### Per-MV (from `cut_music_video` result)

The `quality_telemetry` block (see [3.5](#35-mcp-quality_validation-hook-orchestratorpy)) gives per-clip composite scores, recommendations, retake counts, and total validation overhead.

### Validator-vs-user disagreement rate

**Queryable but no dashboard panel yet** (Phase B candidate):

```sql
-- Clips the user kept in compositions that the validator flagged retake
SELECT g.id, g.validator_score, json_extract(g.validator_payload_json, '$.recommendation') AS rec
FROM generations g
JOIN composition_clips cc ON cc.clip_history_id = g.id
WHERE json_extract(g.validator_payload_json, '$.recommendation') = 'retake';

-- Clips the user retook that the validator passed
SELECT child.id AS retake_id, child.validator_score AS retake_score,
       parent.id AS parent_id, parent.validator_score AS parent_score
FROM generations child
JOIN generations parent ON parent.id = child.parent_clip_id
WHERE child.parent_clip_id IS NOT NULL
  AND json_extract(parent.validator_payload_json, '$.recommendation') = 'pass';
```

### Per-bearer signal counts (BFF)

```sql
-- in portal db
SELECT actor_email, signal_type, COUNT(*) AS n
FROM user_signals
WHERE ts > datetime('now', '-7 days')
GROUP BY actor_email, signal_type
ORDER BY n DESC;
```

### journald log lines

```bash
journalctl --user -u taco-backend | grep -E "validator|analyze-motion|tier1|tier2|tier3"
journalctl --user -u sapiens-sidecar -f
journalctl --user -u taco-backend | grep -E "_dispatch_validator|_on_job_complete"
```

Useful filters:
- `analyze-motion` — every endpoint hit (request + per-tier latency).
- `tier1 raft failed` / `tier3 judge schema/parse failure` — degradation events.
- `validator_runs persist failed` — DB writer issue (rare).

---

## 6. Data flow walkthrough — single clip end-to-end

Trace: a single clip from MCP `cut_music_video` invocation to populated training-data row.

1. **MCP submits job** — `cut_music_video` orchestrator builds the clip body, calls `_apply_shot_lineage(body, ...)` to inject `shot_uuid` + `shot_config_key`, POSTs to `/v2/audio-to-video`.
2. **taco-backend dispatches** — `_submit_job` enqueues; `_dispatch_job` calls `_strip_history_params(job.params)` to remove `shot_uuid`/`shot_config_key`/`parent_clip_id` from manager kwargs, then dispatches to LTX denoiser (local cuda:0 or remote pool worker).
3. **LTX completes** — `worker_loop` saves the result MP4, computes the thumbnail, writes the history row with `shot_uuid` + `shot_config_key` populated, calls `on_complete(job)` → `_decr_queue_on_complete(job)`.
4. **Validator dispatch fires** — `_decr_queue_on_complete` chains `_on_job_complete(job)`. Eligibility checks (COMPLETED + video type + opt-in) pass. `asyncio.create_task(_dispatch_validator(job))` spawns; queue worker dequeues the next job immediately.
5. **`_dispatch_validator` runs** — resolves result_uri to path, calls `validator.run_all_tiers(...)`. Cache miss (new clip), so:
   - sha256 streamed from disk (~50 ms for a 5 MB MP4).
   - Tier 1 RAFT executes on cuda:0 (~150 ms; LTX has just evicted, so cuda:0 is free).
   - Tier 2 returns stub (`LOAD_SAPIENS=0` → synthetic-skipped).
   - Tier 3 multimodal Gemma call (~3 s).
   - `composite()` produces score + recommendation.
   - `_validator_runs_persist` INSERTs row keyed by (sha256, version).
   - UPDATE on `generations` row sets `validator_score` + `validator_payload_json` + `validator_version`.
6. **MCP active hook** — `_run_clip_step` finished tail-extraction. If `quality_validation.enabled=true`, calls `nf_http.post_json("/v2/video/analyze-motion", body, timeout=30)`. Cache hit (sha256 already in `validator_runs`). <50 ms response. `current_state["clip_quality"][str(clip_idx)] = payload`. `session.append_event("validator_result", {...})`.
7. **BFF tee** — `Session.append_event` calls `_tee_event_fire_and_forget`, POSTs `validator_result` event to `{NOODLEFINGER_BFF_URL}/api/mcp/events`. BFF inserts into `user_signals` with `signal_type="mcp_event"`, `target_id=session_id`.
8. **Composition export** — user assembles N clips into a comp, calls `/v2/compositions/{id}/export`. Backend calls `history.record_composition_clips(comp_id, params["clips"])` (server.py:5514) → INSERTs N `composition_clips` rows linking the comp to each constituent `generations.id` (or NULL for synthetic flash inserts).
9. **User downloads** — gallery download triggers BFF `user_signals_log` with `signal_type="gallery.download"`, `target_id=clip_id`.
10. **Future — pair construction** (Phase C, not yet wired): a cron walks `generations` grouped by `shot_config_key`, joins `composition_clips` (clips that survived to a comp = chosen) and `parent_clip_id` (clips that got retaken = rejected), emits `preference_pairs` rows with `signal_source="user_retake"` + `signal_strength=1.0`.
11. **Future — DPO LoRA training** (Phase C): a training cron consumes `preference_pairs` (where `used_in_training_run_id IS NULL`), trains a DPO LoRA against the base LTX checkpoint, writes a `training_runs` row, registers the LoRA in `lora_registry.json`, marks the consumed pairs.

---

## 7. What's NOT yet wired (forward-looking)

| Surface | Status | Notes |
|---|---|---|
| Sapiens real inference | Stub mode (rc1/rc2/rc3) | Schema forward-stable; rc-final swap is drop-in. License read in `LICENSE_NOTES.md`. |
| `prompt_embedding` lazy-fill | Column exists, never written | Phase B retrieval — requires extending llama-swap config with `/v1/embeddings` endpoint hitting the already-loaded Gemma 3 12B (~3584-dim float32). User-locked design choice. |
| `validator_artifact_uri` column | Exists, NULL for all rows | Will hold rc-final Sapiens per-frame keypoint dumps. |
| `composition_id` on `generations` | Column exists, never written | Denorm convenience; redundant with `composition_clips` join. May never be wired if the join is fast enough. |
| `lora_applied_id` / `lora_applied_strength` | Columns exist, never written | Need a manager-side hook on LoRA fusion to record which adapter actually fused at runtime. |
| DPO pair construction | `preference_pairs` table exists, no writer | Phase C — joins `generations` by `shot_config_key`, walks `composition_clips` + `parent_clip_id`. |
| DPO training cron | `training_runs` table exists, no writer | Phase C. |
| Dashboard panel for validator scores | Not built | Operator currently SQLs directly. Phase B candidate. |
| Validator-vs-user disagreement metric | Queryable but no view/dashboard | Phase B. |
| Active-learning loop (borderline → user prompt) | Not built | When composite is in `[0.45, 0.65]`, surface to user for explicit thumbs-up/down → `preference_pairs` with `signal_source="explicit_rating"`. |
| Frame-level CLIP embeddings | Not built | Phase B retrieval. |
| ~~`quality_validation.motion_intent_map` consumed by tier-3~~ | **Wired in v1.17.0-rc5** | `AnalyzeMotionRequest.motion_intent` declared, threaded through `validator.run_all_tiers()` → `_run_tier3_judge()`, rendered conditionally into the tier-3 user-message text. `JUDGE_PROMPT_V1` instructs the judge to reconcile its verdict against intent. Passive `_dispatch_validator` does NOT carry intent — only the synchronous `/v2/video/analyze-motion` endpoint receives it from MCP. |

---

## 8. Cross-references

- [`docs/API.md`](API.md) — `/v2/video/analyze-motion` endpoint contract (TODO if missing — verify on next API.md sync).
- [`docs/MCP.md`](MCP.md) — MCP v0.7.0 `quality_validation` field, `cut_music_video` tool surface.
- [`docs/operator-tuning.md`](operator-tuning.md) — env vars + turbo coordination.
- [`docs/MV_EDITING.md`](MV_EDITING.md) — composition lineage (the `composition_clips` writer hook lives in `export_composition`).
- [`CHANGELOG.md`](../CHANGELOG.md) — v1.17.0-rc1 / rc2 / rc3 entries.
- The plan: `/home/ian/.claude/plans/melodic-sniffing-beacon.md` (strategic frame).
- BFF: `noodlefinger-portal/bff/README.md` (v0.2.0 — `user_signals` capture).
- MCP: `noodlefinger-portal/mcp/README.md` (v0.7.0 — quality validation hook + BFF event tee).
- Sapiens sidecar: `/mnt/nvme-1/servers/sapiens-sidecar/SETUP.md` + `LICENSE_NOTES.md`.

---

## 9. Troubleshooting

### "Validator never runs on my jobs"

Check the eligibility chain:

```bash
# 1. Is the bearer opted in?
sqlite3 history.db "SELECT api_key_hash, training_opt_in FROM api_key_metadata WHERE api_key_hash='<sha256-of-bearer>'"

# 2. Are completed video jobs in the queue?
journalctl --user -u taco-backend | grep "_on_job_complete"

# 3. Is the dispatch task actually running?
journalctl --user -u taco-backend | grep "_dispatch_validator"

# 4. Is the result_uri resolvable?
journalctl --user -u taco-backend | grep "result file gone"
```

Common cause: bearer added to `.api_keys` post-seed and not explicitly INSERTed into `api_key_metadata`. See [4.3](#43-privacy-gate-management).

### "Tier-2 always returns stub"

Expected when `LOAD_SAPIENS=0` (default). To verify:

```bash
echo "$LOAD_SAPIENS"   # in taco-backend's environment
systemctl --user is-active sapiens-sidecar
curl http://127.0.0.1:8096/health
```

If `LOAD_SAPIENS=1` AND sidecar is `active` AND `/health` returns `{"ready":true}` AND payload still says `stub: true` → you're on rc1/rc2/rc3. Real inference lands in rc-final.

### "Gemma judge times out"

Tier-3 calls the existing chat_manager + llama-swap. Check:

```bash
curl http://192.168.1.80:8080/v1/models
journalctl --user -u taco-backend | grep "tier3 judge call failed"
```

The judge has a fallback path (`{verdict: "warn", score: 0.5}`) so timeouts don't crash the validator — composite still computes. But chronic timeouts mean tier-3 is contributing constant 0.5, dragging composite down. Investigate the chat infrastructure.

### "MCP quality_telemetry missing from result"

Confirm `quality_validation.enabled=true` was passed:

```bash
# In MCP server logs (Claude Code subprocess):
cat ~/.cache/noodlefinger-mcp/sessions/<session_id>.json | jq '.input.quality_validation'
```

If it's `null` or `enabled: false`, the block is intentionally absent — byte-identical to v0.6.

### "user_signals empty"

Check `NOODLEFINGER_BFF_URL`:

```bash
# In the MCP process env — not the system env:
ps aux | grep noodlefinger-mcp
cat /proc/<pid>/environ | tr '\0' '\n' | grep NOODLEFINGER_BFF_URL
```

If unset, the tee is a no-op. Set in your Claude Code MCP config:

```json
{
  "noodlefinger-mcp": {
    "command": "uvx",
    "args": ["noodlefinger-mcp"],
    "env": {
      "NOODLEFINGER_BFF_URL": "http://localhost:7100"
    }
  }
}
```

### "Validator score in `generations` is stale (older than result_uri)"

Bump `config.VALIDATOR_VERSION` and restart taco-backend. Future calls to `/v2/video/analyze-motion` (active hook) will re-run; passive dispatch only fires on **new** completions, so historical rows are not auto-revalidated. To force backfill:

```bash
# MCP-side (one-shot for a session):
resume_music_video(session_id="...", revalidate=True)
```

There is no taco-backend-side bulk re-validation tool yet. Phase B candidate: a maintenance endpoint `POST /v2/admin/validator/backfill` that walks rows where `validator_version < current` and dispatches.

---

## 10. Next steps

The capture+validator machine is the substrate. The next two phases consume it.

**Wave 2 (immediate)**: flip `LOAD_SAPIENS=1` after rc-final lands real Sapiens-2 inference. Validator scores gain a real tier-2 signal; recommendation accuracy improves on identity-drift cases. No client-side changes required (the stub-tolerant path collapses once the response no longer carries `stub: true`).

**Phase B (weeks 3-4)**: prompt embeddings via `/v1/embeddings` on Gemma 3 12B → fill `prompt_embedding` lazily; CLIP frame embeddings on a thin sidecar; semantic-search MCP tools (`/v2/history/search?q=...`); validator-score dashboard panel.

**Phase C (weeks 5-6)**: DPO pair construction job (joins `generations` by `shot_config_key` + `composition_clips` + `parent_clip_id` → `preference_pairs`); training cron that consumes pairs, trains a DPO LoRA against the LTX base, writes `training_runs`, registers the new LoRA in `lora_registry.json`; explicit thumbs-up/down rating endpoint to feed `signal_source="explicit_rating"` rows.

For the full forward roadmap including dependency graph and timing, see the plan: `/home/ian/.claude/plans/melodic-sniffing-beacon.md`. A separate roadmap doc (forthcoming, third agent's deliverable) will track Phase B / C execution.

---

## 11. Surprises / gaps surfaced during this documentation pass

For the roadmap agent's investigation queue:

1. ~~**`motion_intent_map` round-trip is half-wired**~~. **Resolved in v1.17.0-rc5**: `AnalyzeMotionRequest.motion_intent` declared, threaded through `validator.run_all_tiers()` → `_run_tier3_judge()`, rendered conditionally into the tier-3 user-message text block (line omitted when `None` so rc4 baseline is byte-identical for non-MCP callers). `JUDGE_PROMPT_V1` updated to instruct the judge to reconcile against intent. MCP side is unchanged — it was always passing the right field name. Active hook (synchronous `/v2/video/analyze-motion`) carries intent end-to-end; passive dispatch (`_dispatch_validator`) does not (no source for it).

2. **`prompt_embedding` is in the schema but has zero writers**. The plan calls for lazy-fill via llama-swap `/v1/embeddings`, but that endpoint isn't yet exposed and no taco-backend code path attempts to populate the column. Phase B blocker.

3. **`validator_artifact_uri` column exists with no writer**. Suggests an earlier intent to dump per-clip artifacts (RAFT flow tensors? Sapiens keypoints?) but no module touches it. Phase C may want this for retraining-time provenance.

4. **`lora_applied_id` / `lora_applied_strength` columns are unwired**. The captured `shot_config_key` includes `lora_id + lora_strength` from the request, but the *actual* fused LoRA at runtime is never recorded. If LoRA-cache eviction surprises ever happen (different strength than requested), we'd silently miss it. A `_run_t2v` / `_run_a2v` hook would fix.

5. **Passive validator dispatch has no rate limiting or back-pressure.** `_on_job_complete` fires `asyncio.create_task` per completed job — under heavy MV throughput (28 concurrent video workers in turbo+remote), tier-3 Gemma calls could pile up against the single llama-swap instance. No semaphore, no queue. Phase B candidate: a `_VALIDATOR_SEMAPHORE` cap analogous to `_FRAME_EXTRACT_SEMAPHORE(2)`.

6. **`composition_id` column on `generations` is never written**. Phase 1 plan called it "denorm convenience" but the active path uses `composition_clips` joins instead. Probably safe to drop the column in v1.18 to reduce schema surface.

7. **No taco-backend bulk-revalidate tool**. Bumping `VALIDATOR_VERSION` only affects new dispatches; historical rows stay on the old version forever unless MCP `revalidate=True` is used per-session. A maintenance endpoint or CLI would close the gap.

8. **Sapiens systemd unit uses `Type=exec`, not `Type=notify`**. `/health` readiness is not a startup gate. If rc-final's weight-load fails, the process stays alive with `/health` returning `{"ready": false}` indefinitely; `Restart=on-failure` doesn't fire. Documented in `SETUP.md` but no action item open.

9. **MCP→BFF tee is anonymous-tolerant**. `actor_email` falls back to literal `"unknown"` when the X-User-Email header is missing. In single-tenant deploy this is fine; for multi-tenant the BFF would need to bind a portal-issued bearer at MCP install time. `mcp_events.py:11` flags this in a TODO comment.

10. **Tier-1 RAFT downsamples to 256-wide unconditionally**. A 1080p clip's flow gets computed at ~256×144. For very-low-motion clips with fine details this may under-detect motion; for static-then-motion (the v0.4.6 defect) it's fine. Consider an env flag `RAFT_FLOW_TARGET_WIDTH` for tuning.
