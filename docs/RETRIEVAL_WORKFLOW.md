# Phase B Retrieval Workflow

**Status**: shipped in `taco-backend v1.18.0-rc2` + `noodlefinger-mcp v0.8.0`.
**Audience**: LLM authoring agents (Claude Code, Cursor, Codex) + the
operators who run them. **Scope**: how to use `find_similar_shots`,
`recommend_loras`, and `bulk_revalidate` during a daily MV authoring
session.

This doc is the workflow guide. For wire shapes see
[`API.md`](API.md) (`/v2/embeddings/search`,
`/v2/embeddings/recommend-loras`, `/v2/system/bulk-revalidate`). For tool
schemas see [`MCP.md`](MCP.md).

---

## 1. Why retrieval

Every completed video clip lands in `generations` with a validator score
(`validator_score ∈ [0, 1]`, see
[`CAPTURE_VALIDATOR.md`](CAPTURE_VALIDATOR.md)) and an embedded copy of
its prompt (`prompt_embedding` blob, indexed in the `clip_embeddings`
sqlite-vec virtual table). The retrieval tools turn that corpus into a
working memory: **"for prompts like yours, here's what worked and what
didn't."** An LLM authoring a shot list can borrow validated prompt
language and pick LoRAs by historical score instead of guessing from the
LoRA name.

The signal is per-bearer: every query is filtered by `api_key_hash`, so
a teammate's corpus never bleeds into yours.

---

## 2. The three tools

### `find_similar_shots`

```
find_similar_shots(
    prompt: str,                       # 5-2000 chars, required
    k: int = 5,                        # 1-20
    min_validator_score: float = 0.5,  # [0, 1]
    validator_version_filter: str | None = None,  # default: current VALIDATOR_VERSION
    genre: str | None = None,          # optional
)
```

Wraps `POST /v2/embeddings/search`. Embeds `prompt` via the backend's
`/v1/embeddings` proxy (Gemma 3 12B reused), then runs an
sqlite-vec L2 distance scan over your bearer's `clip_embeddings`,
filtered by `validator_score >= min_validator_score` and matching
`validator_version`. Re-ranks with the composite formula:

```
final = 0.50 · similarity
      + 0.35 · validator_norm        # max(0, score-0.5)/0.5
      + 0.10 · recency               # exp(-age_days / 30)
      + 0.05 · composition_kept      # 1 if shot ended up in a final composition
```

Returns up to `k` ranked results:

```json
{
  "validator_version_filter": "1.17.0-rc5",
  "results": [
    {
      "shot_id": 12453,
      "prompt": "masked drummer in smoky red volumetric light, slow 85mm tracking zoom-in",
      "similarity_score": 0.82,
      "validator_score": 0.78,
      "lora_applied_id": "kinetic-v2",
      "lora_applied_strength": 0.9,
      "model": "ltx-2-3-fast",
      "shot_uuid": "ab12...32hex",
      "in_final_composition": true,
      "created_at": 1756849210.4,
      "final_score": 0.736
    }
  ]
}
```

### `recommend_loras`

```
recommend_loras(
    prompt: str,
    motion_intent: str | None = None,  # free-form: "static", "dynamic", "slow tracking", ...
    k: int = 3,                         # 1-10
)
```

Wraps `POST /v2/embeddings/recommend-loras`. Same vector search as above
but limited to the top-50 nearest opt-in clips, then aggregates by
`lora_applied_id` (NULL is the no-LoRA baseline). Each candidate gets:

- `mean_validator_score` (0..1) — average score on similar prompts
- `sample_count` — n
- `mean_strength` — mean strength used historically
- `expected_boost` = `mean_score - no_lora_baseline_mean`
- `rank_score` = `0.7 · mean_score + 0.3 · max(0, expected_boost)`

Top-`k` returned. Empty array means no opt-in similar clips with a
LoRA — fall back to default.

### `bulk_revalidate` (ADMIN)

```
bulk_revalidate(
    since_validator_version: str,      # MCP arg name; backend body field is target_validator_version
    dry_run: bool = True,
    confirm_admin: bool = ...           # MCP defense-in-depth gate
)
```

Wraps `POST /v2/system/bulk-revalidate`. Selects rows where
`validator_version IS NULL OR validator_version != target` AND
`result_uri IS NOT NULL`, ordered by `created_at DESC`, capped at
`limit` (default 100, max 10000). When `dry_run=true` returns a count
and a sample of `id`s; when `dry_run=false` spawns fire-and-forget
`_dispatch_validator(...)` tasks that re-run RAFT (Tier 1) + Gemma
judge (Tier 3) and UPSERT `validator_runs` + UPDATE
`generations.validator_score / validator_payload_json /
validator_version` per row.

The MCP wrapper requires `confirm_admin=true` on top of the backend's
admin-auth check (`_require_admin`).

---

## 3. Prerequisites

Before retrieval returns anything useful:

- **Backend** v1.18.0-rc2 or later running. Verify:
  `curl http://localhost:8090/health` and look for
  `"validator_version"` and `"sqlite_vec_available": true`.
- **sqlite-vec extension** loaded by `history_store`. If it failed to
  load (missing shared library, glibc mismatch), the two embedding
  endpoints return a clear `503` —
  `"embedding search not available — install sqlite-vec extension"`.
- **llama-swap `/v1/embeddings`** reachable. The backend's `chat.embed`
  proxies to it; if disabled the endpoints return
  `503 embedding service unavailable`.
- **Backfill complete** for any pre-rc2 rows. Run:
  ```bash
  uv run python scripts/backfill_prompt_embeddings.py
  ```
  Idempotent — re-runs skip rows that already have a `prompt_embedding`.
- **Bearer opted in** to training: `api_key_metadata.training_opt_in =
  1` (default for keys seeded from `.api_keys` on the v1.17.0-rc1 v2→v3
  migration; opt-out for any externally-added bearer until INSERTed).
- **Corpus size**: ≥ ~100 validated rows under the current
  `VALIDATOR_VERSION` for `find_similar_shots` to produce non-noisy
  results; ≥ ~30 rows per LoRA for `recommend_loras` to be meaningful.

A fresh deploy will return empty `results` — that is **not** an error,
it is "no historical signal yet."

---

## 4. Workflow A — shot authoring with retrieval

The canonical loop. Steps 2-4 are the per-shot inner loop; step 5 is
the submit.

### Step 1 — sketch a shot list (prompts only)

Don't bind LoRAs, image_uris, or strengths yet. Let retrieval guide
those.

```python
shots = [
    {"prompt": "masked drummer in smoky red light, slow zoom in"},
    {"prompt": "wide stadium crowd, hands raised, golden hour"},
    {"prompt": "macro detail of guitar strings vibrating"},
]
```

### Step 2 — find similar past shots

For each candidate prompt, look up the top-5 similar past clips at
`min_validator_score=0.6` (the "what worked well" threshold). If
sparse, lower to 0.4 — middling results often expose what to *avoid*.

```python
# MCP tool call (LLM agent flavor)
matches = await find_similar_shots(
    prompt=shots[0]["prompt"],
    k=5,
    min_validator_score=0.6,
)
for r in matches["results"]:
    print(r["validator_score"], r["lora_applied_id"], r["prompt"][:60])
```

Equivalent raw HTTP:

```bash
curl -s -X POST http://localhost:8090/v2/embeddings/search \
  -H "Authorization: Bearer $NF_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"masked drummer in smoky red light, slow zoom in",
       "k":5,"min_validator_score":0.6}'
```

Inspect:

- **`validator_score`** — closer to 1.0 = healthier clip per the
  composite (RAFT + Sapiens + Gemma judge).
- **`in_final_composition`** — `true` means a human (or the LLM) kept
  this clip in the final cut. Strong signal.
- **`lora_applied_id` / `lora_applied_strength`** — the exact LoRA
  binding of the past clip. If the top-1 result scored 0.8+, copy it.
- **`prompt`** — the actual prompt text. Borrow phrasing.

### Step 3 — get a LoRA recommendation

Per shot:

```python
recs = await recommend_loras(
    prompt=shots[0]["prompt"],
    motion_intent="kinetic",     # or "static portrait", "slow tracking", "dynamic action"
    k=3,
)
top = recs["recommendations"]
if top and top[0]["sample_count"] >= 5:
    shots[0]["lora"] = {"id": top[0]["lora_id"], "strength": top[0]["mean_strength"]}
# else: fall through, leave LoRA unset (uses session default or none)
```

The `sample_count >= 5` floor is the **rule of thumb** — anything below
that is a coincidence, not a pattern.

### Step 4 — refine prompts using retrieved language

Read the top-2-3 prompt texts from step 2. Look for:

- Specific lighting words: `"volumetric"`, `"rim-lit"`, `"chiaroscuro"`
- Specific motion verbs: `"orbits"`, `"tracks past"`, `"pushes in"`
- Specific lens framing: `"85mm shallow DOF"`, `"wide-angle distortion"`

Rewrite the candidate prompt to incorporate language that scored well.
The LTX VAE + Gemma judge reward specificity; generic prompts
underperform consistently. (Phase D experiments showed +0.07 mean
composite when the LLM borrowed from validated language vs. authoring
fresh.)

```python
shots[0]["prompt"] = (
    "masked drummer in smoky red volumetric light, "
    "slow 85mm tracking zoom-in, rim-lit silhouette"
)
```

### Step 5 — submit cut_music_video

Pass the refined shot list. Quality validation auto-fires
(`v1.17.0-rc2`+) — set `quality_validation.on_failure="retake"` to let
the orchestrator re-roll any clip whose composite drops below the
retake threshold.

```python
result = await cut_music_video(
    prompt="unified style description (only used when shot omits prompt)",
    music_prompt="upbeat synthwave with driving bass",
    duration_s=180,
    num_clips=len(shots),
    shot_list=shots,
    quality_validation={
        "enabled": True,
        "on_failure": "retake",
        "max_retakes_per_clip": 1,
    },
)
print("composite avg:", result["quality_telemetry"]["composite_score_avg"])
```

A/B yourself: run an authoring session WITHOUT retrieval, then again
WITH retrieval over the same target. Compare
`quality_telemetry.composite_score_avg`.

---

## 5. Workflow B — LoRA selection alone

Use case: the prompt is already written (e.g., copy from a planning
doc) and you just need to pick a LoRA.

```python
recs = await recommend_loras(
    prompt="cinematic wide shot of city skyline at dusk, slow drone push-in",
    motion_intent="slow tracking",
    k=5,
)
```

Decision rule:

| `sample_count` | `mean_validator_score` | Action |
|---|---|---|
| ≥ 10 | ≥ 0.7 | **Use it.** Strong historical signal. |
| ≥ 5 | ≥ 0.65 | Use it; consider one A/B test against no-LoRA. |
| ≥ 5 | < 0.6 | Skip — historically *worse* than baseline. |
| < 5 | any | Don't trust. Use `search_loras(query, ...)` (lexical) instead. |
| empty | — | No historical data. Use default or `list_loras()`. |

`expected_boost` < 0 (LoRA scoring lower than the no-LoRA baseline on
similar prompts) is a hard skip — the corpus is telling you that LoRA
hurts on this prompt class.

---

## 6. Workflow C — validator version migration

When `config.VALIDATOR_VERSION` bumps (e.g., `1.17.0-rc5` →
`1.17.0-rc6` after a Gemma upgrade or a judge prompt rewrite), all
older scores are stale. `find_similar_shots` filters by the current
`VALIDATOR_VERSION` by default, so old rows silently drop out of the
search corpus.

Operator runbook:

```bash
# 1. Dry-run: how many rows need re-scoring?
curl -s -X POST http://localhost:8090/v2/system/bulk-revalidate \
  -H "Authorization: Bearer $NF_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"target_validator_version":"1.17.0-rc6","dry_run":true,"limit":1000}'
# → {"dry_run": true, "would_revalidate": 847, ...}

# 2. Real run, batched.
curl -s -X POST http://localhost:8090/v2/system/bulk-revalidate \
  -H "Authorization: Bearer $NF_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"target_validator_version":"1.17.0-rc6","dry_run":false,"limit":1000}'
```

Or via MCP (note `confirm_admin=true`):

```python
await bulk_revalidate(
    since_validator_version="1.17.0-rc6",
    dry_run=False,
    confirm_admin=True,
)
```

**Cost**: ~3 s per video (RAFT ~150 ms/clip + Gemma judge ~2-3 s).
Sapiens stub-skips (no penalty) when `LOAD_SAPIENS=0`. Spawned
fire-and-forget; the endpoint returns immediately with `{"queued":
N}`. Watch progress in `journalctl -u taco-backend | grep validator`.

**When to run**:

- After a Gemma model upgrade (e.g., `gemma-4-31b-it` →
  `gemma-4-31b-it-pro`) — the judge's scoring distribution shifts.
- After a `JUDGE_PROMPT_V1` → `JUDGE_PROMPT_V2` change — same.
- After widening Tier 1 (e.g., `raft_small` → `raft_large`).
- After backfilling a missing tier (e.g., flipping
  `LOAD_SAPIENS=0 → 1` and wanting old rows re-scored with real Tier 2
  signal).

**When NOT to run**:

- Just because rc5 → rc5.0.1 patch — if the scoring is unchanged,
  bumping `VALIDATOR_VERSION` is the bug, not the cure.
- During peak generation hours — fire-and-forget tasks compete with
  live jobs for cuda:0 RAFT slots and the `chat_manager` Gemma queue.

---

## 7. Limitations + caveats

- **Decoder-only LM embeddings** (Gemma) are weaker than encoder-only
  models on short prompts. Empirically: prompts < 50 tokens cluster
  noisily; prompts > 100 tokens produce strong neighborhoods. If your
  shot list has terse 8-10 word prompts, retrieval similarity scores
  may be unreliable — pad with style/lighting/camera language.
- **Privacy gate is hard**: every query WHEREs by `api_key_hash`.
  Cross-bearer retrieval is not a knob — you only ever see your own
  corpus. Multi-tenant deployments where teammates *want* shared
  retrieval will need a backend change.
- **Rate limit**: 10 req/sec/key on `/v2/embeddings/*` and
  `/v2/system/bulk-revalidate`, token-bucket, burst capacity 10. 429
  with `Retry-After` header on exhaustion. Process-local — restart
  resets buckets.
- **Empty corpus** returns empty `results` (HTTP 200), not 404 or
  error. The MCP wrappers forward verbatim; the LLM should treat
  `results == []` as "no signal, fall back."
- **Validator version drift**: results filtered to current
  `VALIDATOR_VERSION` by default. If you've been running for months
  through several version bumps without `bulk_revalidate`, your
  effective corpus may be small. Pass `validator_version_filter=` to
  inspect older versions explicitly.
- **Corpus age**: don't rely heavily on retrieval until ~4 weeks of
  validated runs. Early days = high variance per query.
- **Genre filter is best-effort**: matches `g.genre` if set. Clips
  generated before `genre` started getting set will have NULL and won't
  match a non-NULL filter.
- **shot_uuid format**: pre-v0.8 sessions wrote 16-hex,
  v0.8+ writes 32-hex. Both are valid as opaque ids — never
  prefix-compare across sessions.

---

## 8. Eval — knowing retrieval is helping

### Online signal

The MCP server logs each `find_similar_shots` / `recommend_loras` call.
Operator can grep by tool name:

```bash
journalctl -u noodlefinger-mcp | grep -E 'find_similar_shots|recommend_loras' | wc -l
```

Backend-side metrics live at `GET /v1/system/metrics`:

```
embeddings_search_total
embeddings_search_success
embeddings_search_failure
embeddings_search_rate_limited
embeddings_search_results_avg     # results_sum / results_count
recommend_loras_total
bulk_revalidate_total
embeddings_search_latency_p50_ms
embeddings_search_latency_p95_ms
```

### Offline acceptance criteria

The retrieval feature graduates from experimental to default-on when:

- **≥ 2% lift** in `quality_telemetry.composite_score_avg` for sessions
  where the LLM called `find_similar_shots` ≥ 1× per shot vs.
  baseline.
- **p95 latency < 800 ms** on `/v2/embeddings/search` (currently
  500-700 ms typical: ~150 ms embed + ~200 ms vec_distance scan + ~200
  ms re-rank/serialize).
- **Hit rate ≥ 0.6** — fraction of `find_similar_shots` calls that
  return ≥ 1 result above `min_validator_score` (proxy for "the corpus
  has enough signal").

Track weekly via the BFF `user_signals` table (every MCP tool call
tees there if `NOODLEFINGER_BFF_URL` is set, see `mcp v0.7.0`).

---

## 9. Forward-looking (Phase B v0.2 and beyond)

Currently retrieval is **manual** — the LLM must call the tools
explicitly. Planned next:

- **Auto-injection in `cut_music_video`**: orchestrator runs
  `find_similar_shots` per shot internally, threads the top match's
  `lora_applied_id` + a ` "borrow:..."` hint into the prompt
  enhancement step. Default-off until eval criteria above are met.
- **Frame-level CLIP visual similarity** (Phase B v0.2): a parallel
  `clip_visual_embeddings` table indexed off the result MP4's
  middle-frame CLIP features. `find_similar_shots(image_uri=...)` for
  visual-style retrieval.
- **Active learning loop on borderline scores** (Phase D): clips with
  composite ∈ [0.45, 0.65] (warn band) get queued for human review;
  reviewer-corrected scores feed a DPO pair-construction pass against
  `parent_clip_id`.
- **Cross-bearer retrieval (opt-in)**: a `retrieval_share=public`
  column on `api_key_metadata` to let teams pool corpora. Privacy-gate
  flips from required to optional, but only over rows with explicit
  share-on.

See [`CAPTURE_VALIDATOR_ROADMAP.md`](CAPTURE_VALIDATOR_ROADMAP.md) for
the full Phase B/C/D plan.
