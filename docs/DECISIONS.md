# Decision Log

Architecture Decision Records (ADRs) for taco-backend and the capture+validator+training pipeline (Phases A→D).

Each ADR captures the **context**, the **decision**, the **consequences** we accepted, and a **revisit-when** trigger. The audience is a future engineer modifying these subsystems who needs to know why we chose X over Y, what trade-offs were considered, and what new evidence would justify revisiting.

ADRs are append-only. If a decision is overturned, add a new ADR that supersedes the old one — don't edit history.

---

## ADR-001: SFT-on-chosen for Phase C v1, Diffusion-DPO deferred to C.1

- **Date**: 2026-04-29
- **Status**: Locked, scoped to Phase C v1
- **Owners**: validator + training pipeline

### Context

Phase C closes the capture→validator→training loop by fine-tuning a LoRA on the operator's preference signal. Two candidate training paradigms:

1. **SFT-on-chosen** — supervised fine-tune on clips the validator (and operator) accepted (`recommendation == pass` and downstream signal: kept in composition, exported, downloaded).
2. **Diffusion-DPO** — direct preference optimization over (chosen, rejected) pairs from `preference_pairs`. Contrastive: pulls toward chosen, pushes away from rejected.

Diffusion-DPO is research-grade. HuggingFace `DPOTrainer` doesn't natively support video diffusion; we'd need a custom training loop wrapping the LTX denoiser, with custom β scheduling and rejected-sample loss masking. The published recipes (Diffusion-DPO paper, DiffuSeq-DPO) are image-only. Adapting to LTX-2.3's distilled multi-stage transformer is a multi-week R&D investment with unknown convergence behavior.

SFT, by contrast, is a solved problem. We already have LTX LoRA training infrastructure in `/mnt/nvme-1/repos/LTX-2/scripts/`. Operator workflow plus validator filtering already produces a clean "chosen" set; the rejected set is correlated noise (failed validators, bad seeds, incoherent prompts).

### Decision

Ship **SFT-on-chosen** as Phase C v1. Train an LTX LoRA on clips where:

- `validator_runs.recommendation == 'pass'`, AND
- the clip appears in at least one exported composition (`composition_clips` join), AND
- the clip's `validator_score >= 0.70` (above the `pass` floor of 0.65 to bias toward strong examples).

Run an A/B against the current baseline (no LoRA) on the next ~30 MV sessions. If the chosen-LoRA wins (operator picks it ≥60% of the time in side-by-side gen on identical prompts), ship it as the default Char-style LoRA. If it plateaus or loses, escalate to Diffusion-DPO (Phase **C.1**).

### Consequences

- Faster ship: existing training scripts reused; operator can produce v1 LoRA in days, not weeks.
- Simpler training math: standard MSE / flow-matching loss; no β tuning, no contrastive pair construction at training time.
- **Ignores the rejected set entirely** — no contrastive learning. Rejected clips still accrue in `preference_pairs` for Phase C.1 if needed; not wasted, just deferred.
- A/B framework is the same regardless of training paradigm, so investment in eval harness carries forward.

### Revisit when

- The first SFT-on-chosen LoRA's A/B completes and the win rate is **<60%** (i.e., SFT plateaued or lost). At that point, build out the Diffusion-DPO training loop using accumulated `preference_pairs` data.
- Or if research lands a working Diffusion-DPO-for-video reference implementation we can fork.

---

## ADR-002: keyframes default chain mode in v0.4.6 (was seamless-segment)

- **Date**: 2026-04-22
- **Status**: Active
- **Supersedes**: v1.12.0 default of `seamless-segment` for `cut_music_video` chained shots

### Context

v1.12.0 shipped seamless-segment chain conditioning (multi-frame `VideoConditionByLatentIndex`, hard-pinning the first 9 target pixel frames to the donor clip's tail) as the **default** chain mode in the mcp orchestrator. The intent: visually frame-perfect continuity between shots.

In production with operator MV sessions, follower clips exhibited a consistent **motion deficit in the first ~3s**. Diagnosis traced to the hard-pin propagating through LTX's temporal attention: the 9 pinned pixel frames (= latent frames 0 and 1) acted as a strong attractor for the surrounding latents, suppressing motion synthesis. Effect was worst on a2v with high BPM where the operator wanted the first beat hit on motion. Soft-guide via `VideoConditionByKeyframeIndex` (the v1.11.5 classical semantic) didn't have this issue — a single still on frame 0 with strength 1.0 only weakly biases neighbors.

### Decision

Flip the orchestrator default to **`chainMode="keyframes"`** in noodlefinger-mcp v0.4.6. `seamless-segment` remains available as an opt-in flag for users explicitly wanting frame-perfect continuity (e.g., split-screen continuity shots).

### Consequences

- Motion restored on the default path; first-beat hits land naturally.
- Continuity across cuts is slightly looser (the head of the follower is a generative interpretation of the donor's last frame, not a literal copy).
- The 9-frame `tailTrimFrames` heuristic on the donor (v0.4.5) is now scoped to the seamless-segment opt-in path; keyframes default donors use `tailTrimFrames=0`. Orchestrator computes this automatically.
- Backend `_build_segment_conditioning_latent` codepath is unchanged — opt-in users still get the same byte-identical behavior.

### Revisit when

- A user reports a motion deficit specifically on the keyframes default (not the seamless-segment opt-in). Likely cause would be a stronger-than-expected frame-0 pin from CRF-33 statistics; would investigate per-keyframe `crf` overrides before considering a wholesale revert.
- Or if seamless-segment's hard-pin issue is fundamentally fixable upstream (e.g., LTX-2.4 ships with attention-strength-aware pin propagation), at which point seamless-segment could become default again.

---

## ADR-003: sqlite-vec for vector store (NOT pgvector / chroma)

- **Date**: 2026-04-15 (Phase B planning)
- **Status**: Locked for Phase B v0.1

### Context

Phase B introduces semantic retrieval over past clips: given the LLM's draft prompt for the next shot, find the K nearest neighbors in our prompt-embedding space and return their generation parameters + recommendation as in-context examples. Storage requirements:

- Embeddings: 768-dim float32 (Gemma final hidden state, mean-pooled), one per clip.
- Scale: 10k-100k clips at single-operator throughput; 1M+ if we ever multi-tenant.
- Query: top-K nearest by cosine, filtered by `validator_version` and (optionally) `api_key_hash`.

Three credible options:

1. **pgvector** — production-grade, HNSW + IVFFlat indexes, mature ecosystem. Requires Postgres to be in the stack.
2. **chroma** — purpose-built embeddings DB. Standalone process, REST API, separate persistence layer.
3. **sqlite-vec** — sqlite extension, KNN + HNSW, embedded.

We are single-operator, single-tenant. We already run sqlite (`history.db`) and have zero ops appetite for a second persistence layer. Adding Postgres or chroma means: another systemd unit, another backup target, another schema migration story, another point of failure during turbo entry/exit storms.

### Decision

Use **sqlite-vec** (~1 MB Rust .so loaded as a sqlite extension). Embed the index inside `history.db` as a virtual table `clip_embeddings` joined by `generation_id`. HNSW index on the 768-dim vector column.

### Consequences

- Zero new processes, zero new deps in the runtime path.
- One backup target (the existing `history.db` WAL).
- Recall at 10k-100k scale is well within HNSW's sweet spot (sub-10ms p99 on a single core).
- **Per-tenant data isolation** is row-level only (filter by `api_key_hash` in the query). If we ever need physical isolation (regulatory, customer-specific encryption), we'd need to migrate.
- sqlite-vec is younger than pgvector; on the off-chance it has correctness regressions, the index can be rebuilt from `prompt_embedding` BLOBs on `generations` (which we keep as ground truth).

### Revisit when

- Multi-tenant load grows beyond ~1M embeddings (HNSW build time and recall both degrade past that without partitioning).
- Per-tenant data-isolation requirements force shard-per-tenant.
- sqlite-vec hits a correctness regression we can't work around (rebuild BLOB → vector index recovery path is the safety net).

---

## ADR-004: Sapiens-2 in tier-2 of validator (despite stub mode)

- **Date**: 2026-04-25
- **Status**: Active in v1.17.0-rc2; real inference lands rc-final

### Context

Three-tier validator design (see ADR-007) needs a tier-2 signal that's orthogonal to RAFT (motion) and Gemma (semantic/identity). Sapiens-2 is Meta's open-vocab pose+segmentation model, well-suited to detecting pose temporal-stability artifacts (e.g., elbow popping, hand identity flicker, foot sliding) that motion-flow can't see.

The license question: Sapiens-2's AUP §1.b.vi.ii prohibits use for **"biometric processing"**. Counsel reviewed; concluded that:

- Per-clip temporal stability scoring on operator-generated content (no human subject identification, no enrolled-user matching) does **not** constitute biometric processing under any common interpretation.
- The scope is internal-use only at this time. Single-tenant deployment; no external bearer is currently active. If we onboard external tenants, that changes the scope and we re-review.

This is documented in `/mnt/nvme-1/servers/sapiens-sidecar/LICENSE_NOTES.md`.

The sidecar is wired and dispatches in v1.17.0-rc2 but ships in **stub mode**: returns `{"stub": true}` for all requests. Real inference (model load, FastAPI handler with cuda:1 placement) lands in the rc-final ship after operator review of the validator pipeline diff.

### Decision

Adopt Sapiens-2 as the tier-2 validator. Ship rc2 with the stub. Land real inference in rc-final.

### Consequences

- Tier-2 stub contributes `0.2 × 1.0 = 0.2` to the composite (treated as "skipped, no penalty"), so composite scores in rc2 are biased ~0.05-0.1 high vs what they'll be once real Sapiens lands. A/B baselines collected during rc2 must be rebuilt against rc-final scores; bumping `VALIDATOR_VERSION` invalidates the cache and forces re-run.
- License risk is bounded by the current scope (internal, non-biometric). Documented and reviewable.
- Tier-2 is the lowest weight (0.2 vs 0.4 for tiers 1 and 3) by design — if Sapiens proves weak signal, the composite is robust.

### Revisit when

- External bearer activates (the scope changes; legal must re-review against AUP).
- Tier-1 + tier-3 alone prove sufficient signal (operator A/B shows tier-2 isn't moving the needle on retake decisions). At that point, evict Sapiens to free cuda:1 budget for other tenants.
- Sapiens-2 license changes (Meta has historically tightened RAIL-style licenses; we should monitor).

---

## ADR-005: manual retrieval (find_similar_shots tool) instead of auto-injection in orchestrator

- **Date**: 2026-04-18 (Phase B planning)
- **Status**: Locked for Phase B v0.1; v0.2 may revisit

### Context

Phase B retrieval can be wired two ways:

- **Option A — manual tool**: expose `find_similar_shots(prompt, k, validator_version)` as an MCP tool. The LLM (Claude inside the noodlefinger-mcp orchestrator) calls it explicitly when it wants in-context examples.
- **Option B — auto-injection**: orchestrator wraps the LLM call; before each shot prompt, transparently retrieves K nearest clips and injects them into context.

Option B is the "do-it-for-them" UX. Pros: zero LLM behavior change required, retrieval always happens. Cons:

- Adds 30-50 ms per shot (sqlite-vec query + serialization + token budget pressure).
- Risks suppressing LLM creativity — every shot's context is now anchored to past clips, which could collapse stylistic variance over a session.
- Fully opaque to the operator; debugging "why did the LLM pick this style" becomes harder.

Option A is the "transparent tool" UX. Pros: LLM exposes its retrieval intent in tool calls (auditable in transcript), latency is paid only when retrieval is actually useful, doesn't fight LLM creativity. Cons: requires a system-prompt nudge so the LLM learns when to call the tool.

### Decision

Ship **Phase B v0.1 with the manual tool only**. Add a one-paragraph system-prompt nudge in `cut_music_video` and `plan_shot_list` flows: *"When you have a draft prompt for a shot and want to ground it in past clips that worked, call `find_similar_shots`."*

Defer Option B (auto-injection) to **Phase B v0.2**, gated on adoption metrics from v0.1.

### Consequences

- Transparency wins: every retrieval call is in the transcript and auditable.
- LLM must learn to call the tool. We accept that v0.1 might see <50% adoption in early sessions.
- We get a clean signal: if adoption is high, Option B is unnecessary; if adoption is low and quality is suffering, that's the trigger to ship Option B.

### Revisit when

- Eval framework shows **<50% session adoption** of `find_similar_shots` after 2 weeks of Phase B v0.1 in production, AND quality regression vs previous baseline. Both conditions: low usage alone isn't a problem if quality holds.
- Operator subjective feedback says retrieval is "always useful, just always have it on." That's a UX signal to flip to auto-injection.

---

## ADR-006: opt-out default for unknown bearers (NOT opt-in)

- **Date**: 2026-04-26 (v1.17.0-rc1 ship)
- **Status**: Active

### Context

The validator dispatch path on completed video jobs writes payloads to `validator_runs` and (in Phase C) feeds `preference_pairs` for training. This is **training data collection**. Two policies for unknown bearers (api_keys not in `api_key_metadata`):

- **Default opt-in**: training data collected unless the bearer explicitly opts out. Friendly for SaaS UX; participants get value (better LoRAs) by default.
- **Default opt-out**: training data collected only if `api_key_metadata.training_opt_in = 1` exists for that key. Safer for unknown bearers; respects "do nothing surprising" principle.

Single-tenant deployment context: the rc1 migration seeded `api_key_metadata` from `.api_keys` with `training_opt_in=1` for every existing entry. So the operator's primary bearer is opted in. The question is what happens if a **new** bearer is added (rotation, dev key, third-party integration test) and no row is INSERTed in `api_key_metadata`.

### Decision

`_is_training_opted_in(api_key)` returns **`False`** for unknown api_key_hashes. Defense-in-depth: a new bearer never participates in training data collection until someone explicitly INSERTs into `api_key_metadata`.

### Consequences

- External bearers (when we activate them) must explicitly opt in. Onboarding flow needs to include the INSERT.
- Single-tenant safe: rotating the primary key requires the operator to re-seed (migration helper or manual SQL). Documented in operator runbook.
- Cannot accidentally collect training data from a key we forgot to register. Failure mode is "no data" not "surprise data."

### Revisit when

- Multi-tenant onboarding flow is built and the SaaS UX requires opt-in-by-default with an opt-out checkbox at signup. At that point, the policy is per-tenant in their settings, not at the validator dispatch layer.
- We add a UI control panel surface for the operator to manage `api_key_metadata` rows; opt-in default per-key becomes a button.

---

## ADR-007: 3-tier validator composite (RAFT + Sapiens + Gemma judge)

- **Date**: 2026-04-25
- **Status**: Active in v1.17.0-rc2

### Context

Single-signal validators are fragile. Examples:

- **RAFT-only**: a static talking-head clip with no motion looks "unhealthy" but is exactly what the prompt asked for. False retake.
- **Pose-only (Sapiens)**: a clean dynamic clip with anatomically-coherent motion but wildly off-prompt content (subject identity wrong, scene wrong) reads "fine." False pass.
- **Gemma-only**: an LLM judge can spot semantic match but is blind to subtle pose flicker or temporal incoherence that breaks the viewer's eye. False pass.

A single weighted composite triangulates all three. The weights encode prior on which tier is most reliable:

- Tier 1 (RAFT, 0.4): rock-solid mechanical signal. Motion-flow magnitude correlates well with operator's "this clip moves right" judgment. High weight.
- Tier 2 (Sapiens, 0.2): orthogonal signal but lower weight because the model is younger in our stack and currently stub-mode. Bumps to 0.3-0.4 if rc-final + tuning prove it strong.
- Tier 3 (Gemma, 0.4): semantic+identity match. High weight because it's the only tier that can see prompt-output mismatch.

The recommendation rule has a **verdict-level override**: even if composite is high, `tier3.verdict == "retake"` forces `recommendation = "retake"`. This catches cases where RAFT and pose look fine but the LLM spots prompt-output mismatch (e.g., "person in red dress" → person in blue dress, motion is great).

### Decision

Composite formula: `score = 0.4·tier1_norm + 0.2·tier2 + 0.4·tier3`.

Recommendation:
- `pass` if `score >= 0.65`
- `warn` if `0.45 <= score < 0.65`
- `retake` if `score < 0.45` OR `tier3.verdict == "retake"`

### Consequences

- Triangulated signal: any single tier failing won't tank a clip; two tiers failing usually will.
- Tier-2 stub contributes `0.2·1.0` (treated as "skipped, no penalty") so the composite is well-defined even before real Sapiens lands.
- `tier3.verdict` override is a safety valve for the LLM's strongest judgments.
- Weights are not tuned against ground truth yet; they're priors. A/B will surface if they're miscalibrated.

### Revisit when

- A/B over the first 30 MV sessions shows tier-3 dominates retake decisions (e.g., >80% of retakes are triggered by the verdict override, ignoring numeric composite). Suggests we should re-weight or change the threshold.
- Tier-2 real inference proves stronger than the 0.2 weight implies (e.g., it catches cases tier-1 and tier-3 miss). Bump to 0.3 or 0.4.
- Or the inverse: a tier proves nearly useless (e.g., tier-2 contributes noise). Drop its weight to 0.1 or remove.

---

## ADR-008: Gemma reuse via llama-swap for embeddings (NOT sentence-transformers)

- **Date**: 2026-04-15 (Phase B planning)
- **Status**: Locked for Phase B v0.1

### Context

Phase B retrieval needs prompt embeddings. Two options:

- **sentence-transformers** (e.g., `all-mpnet-base-v2`, 768-dim, 420 MB): purpose-built for sentence embeddings. Encoder-only, trained with contrastive objectives optimized for clustering and retrieval. Best-in-class on MTEB benchmarks for short-text retrieval.
- **Reuse Gemma 3 12B via llama-swap**: extend llama-swap's API with a `/v1/embeddings` endpoint that returns Gemma's final hidden state mean-pooled. Decoder-LM, autoregressive training objective. Embedding quality on short-text retrieval typically ~10-15% worse on clustering/retrieval benchmarks than dedicated encoder-only models.

We already have Gemma 3 12B loaded for prompt encoding (LTX text-encoder hub) and chat completions (llama-swap). It's hot in VRAM essentially 24/7. Adding sentence-transformers means: +420 MB model file, +1 new pip dep (`sentence-transformers`), +VRAM during inference (cuda:0 contention with LTX) or +CPU latency (slow).

### Decision

Extend llama-swap with `/v1/embeddings` (mean-pool Gemma's final hidden state). Use that for all Phase B retrieval embeddings.

### Consequences

- **Zero new VRAM, zero new pip deps.**
- Embedding quality is materially worse on standard benchmarks: ~10-15% lower recall@K on short-prompt clustering. We accept this for the simplicity win.
- Gemma's embeddings aren't trained with a contrastive objective, so similar-meaning prompts can land far apart in the space. We mitigate by using larger K (10-20 instead of 5) and letting the LLM filter.
- If we ever want best-in-class retrieval (e.g., for a public-facing semantic search product), sentence-transformers fallback path is documented.

### Revisit when

- Retrieval recall@10 drops below an operator-set threshold during Phase B v0.1 eval (TBD: we'll set the threshold from the first 100 retrievals' subjective relevance).
- We decide to ship a public-facing search/retrieval product where embedding quality is competitively differentiating.

---

## ADR-009: BFF as user-signals locus (NOT taco-backend)

- **Date**: 2026-04-12 (Phase A planning)
- **Status**: Locked for BFF v0.2.0+

### Context

User signals (downloads, exports, "kept" / "rejected" actions in the FE) need to be persisted somewhere joinable with `generations.id` for Phase C training data construction. Two options:

- **taco-backend**: add a `user_signals` table to `history.db`. Wire the FE → backend through BFF passthrough.
- **BFF (noodlefinger-bff)**: add `user_signals` table to BFF's DB. BFF already authenticates the actor (knows `actor_email`), already sees all download/export traffic, and is the natural control-plane locus.

taco-backend is intentionally a stateless **compute** layer (modulo `history.db`, which is per-clip generation provenance, not per-user activity). Adding a user-activity surface here muddies the boundary.

BFF v0.2.0 was already adding `actor_email` tracking and download proxying; adding `user_signals` was a small additive scope.

### Decision

User signals live in the **BFF**. BFF v0.2.0 ships `user_signals` table; BFF v0.3.0 adds `api_key_hash` for multi-tenant scoping. taco-backend stays stateless compute; the only join from backend → user signals is via `generation_id` exposed to BFF queries.

### Consequences

- Cleaner separation: backend owns clip provenance (`generations`, `validator_runs`, `composition_clips`); BFF owns user activity (`user_signals`, future `preference_overrides`).
- The Phase C training-data builder runs in BFF (or a sidecar that joins both DBs); not in backend.
- Slight latency for cross-DB joins (two SQLite files), but at 10k-100k scale this is negligible.

### Revisit when

- Performance/latency forces colocation (e.g., we want sub-50 ms training-data construction for online learning). Move the join into a shared layer.
- BFF gets so heavy that splitting user signals into its own service makes sense.

---

## ADR-010: schema-loose checkpoints (additive migrations only)

- **Date**: 2026-04-26 (v1.17.0-rc1)
- **Status**: Project-wide policy, all schema bumps

### Context

Operator MV sessions are long-running (hours to days). A session started under schema v2 must remain resumable under v3 code without forcing a re-export or re-validation. This rules out any DROP COLUMN, RENAME TABLE, or destructive CHECK-tightening migration.

Pre-v3 rows in `generations` need NULL in new columns; pre-v3 sessions in flight need to keep working as the migration runs.

### Decision

**Every schema bump is additive only**:

- ADD COLUMN with NULLable default
- ADD TABLE
- ADD INDEX
- never DROP / RENAME / TIGHTEN

The `_migrate()` ladder in `history_store` runs each version step idempotently on startup. Migration is single-startup and re-run safe.

### Consequences

- Schema grows over time. Deprecated columns persist (e.g., `prompt_embedding BLOB` on `generations` if we ever switch embedding strategies).
- Some columns will become dead-letter (no writer, only kept for backward-compat). We accept this.
- Schema-loose contract makes resume-from-checkpoint robust across versions.
- Cleanup is deferred indefinitely.

### Revisit when

- Schema bloat becomes operationally painful (e.g., >50 columns on `generations`, query plans degrading). At that point, write a one-time consolidation migration with a versioned cut-over and operator runbook.
- A regulatory requirement forces column removal (e.g., GDPR right-to-be-forgotten on a column we'd otherwise keep).

---

## ADR-011: validator_version scoping is the spine (Phase B + C)

- **Date**: 2026-04-25
- **Status**: Active in v1.17.0-rc1+; locked for Phase B + C

### Context

The validator's behavior changes over time: weights tuned, tiers added, prompts revised, models swapped. A clip scored under `validator_version=1.17.0-rc1` and a clip scored under `1.18.0` are not directly comparable — the Tier-3 judge prompt may have changed, the composite formula may have shifted, the recommendation thresholds may have moved.

If Phase B retrieval mixes embeddings from different validator_versions in the same KNN result, the LLM gets inconsistent quality signals. If Phase C training mixes pairs from different versions, the contrastive signal corrupts (apples-to-oranges).

### Decision

**`validator_version` is the scoping spine** for all retrieval and training queries. Every:

- `find_similar_shots(...)` query filters by current `VALIDATOR_VERSION`.
- `preference_pairs` row is tagged with the validator_version it was constructed under.
- Training data builder filters to a single validator_version.

When `VALIDATOR_VERSION` is bumped (e.g., rc-final lands real Sapiens), the cached `validator_runs` are still valid for historical reference but the *meaningful corpus* for Phase B retrieval and Phase C training shrinks to clips scored under the new version.

A **`bulk_revalidate` operator tool** ships alongside this: rerun the new validator over all completed clips since last bump, populating `validator_runs` rows under the new version. Re-runs are O(N · per-clip-cost) and dominated by tier-1 + tier-3 (RAFT + Gemma) latency.

### Consequences

- Bumping `VALIDATOR_VERSION` "resets" the Phase B/C corpus until backfill completes.
- The backfill cost is real (an hour to a day on a large corpus, depending on how many concurrent Gemma calls llama-swap can sustain).
- Cross-version comparison is impossible — if we want it for analytics, we'd need a compat layer (not built).
- Tied to Phase C cycles: if we bump the validator mid-training, the in-flight LoRA's training set is mixed-version. We rule this out by gating validator bumps on Phase C cycle boundaries.

### Revisit when

- Validator changes become so frequent (>1 per week) that they stall Phase C cycles. At that point, we'd add tier-level versioning (`tier3_version`) so a Tier-3 prompt tweak doesn't invalidate Tier-1 + Tier-2 history.
- Or we build a compat layer that maps old→new scores via a learned recalibration.

---

## ADR-012: real-tier integration tests required on every Phase B+C ship

- **Date**: 2026-04-26
- **Status**: Operational discipline, project-wide

### Context

Phase A (capture+validator scaffolding through v1.17.0-rc2) had a recurring failure pattern: **mock-only tests passed, production failed**. Five distinct production bugs traced to mocked dependencies hiding real-world behavior:

1. `chat_manager.generate_chat_completion` mocked in validator tier-3 tests → didn't catch a real schema-validation bug when llama-swap returned trailing whitespace.
2. RAFT `raft_small` mocked in validator tier-1 tests → didn't catch a real torchvision API change in `Raft_Small_Weights.DEFAULT.transforms()`.
3. PyAV decode mocked in `_extract_frames_as_pils` tests → didn't catch a real FFmpeg-bundled-with-conda issue (libx264 missing, swallowed by exception path).
4. sqlite migration mocked at the connection layer → didn't catch a real WAL-mode interaction with the `PRAGMA user_version` ladder.
5. Sapiens client httpx mocked → didn't catch a real `ConnectError` vs `ReadTimeout` exception-class mismatch in the retry path.

Each bug was caught only after deploy, requiring rollback or hotfix. Cumulative cost: many engineering hours, real operator disruption.

### Decision

**Every PR touching the validator pipeline or Phase B retrieval includes at least one integration test that exercises the real components**:

- Tier-1 tests must call real `raft_small` against a real frame pair (small fixture video, lazy weight download cached in CI).
- Tier-3 tests must call real `chat_manager` against a running llama-swap (CI runs llama-swap as a service container; gated test marker `@pytest.mark.requires_llama_swap`).
- Retrieval tests must call real sqlite-vec against a real `history.db` fixture.
- Schema-migration tests must use real sqlite (no fakes), against a real WAL-mode DB.

Mocks remain valid for unit tests of orchestration logic (e.g., "does the dispatcher call tier-2 when LOAD_SAPIENS=0?"), but every PR ships at least one real-tier test for the path it touches.

### Consequences

- Slower CI (real RAFT inference, real Gemma call, real DB I/O). We accept this — historical evidence says the time saved on hotfixes dominates.
- Some environments can't run the full suite (no GPU, no llama-swap). Marker-based skipping makes the local dev loop fast; the gate runs the full suite.
- Test fixtures (small videos, frame pairs, embedding samples) need to be checked in. ~50 MB total, acceptable.

### Revisit when

Never, in spirit. This is operational discipline. The specific marker names or fixture organization may evolve, but the bar — "every Phase B+C PR exercises real components on the path it touches" — does not relax.

---

## ADR-013: 15-week A/B cadence acceptable for single-operator

- **Date**: 2026-04-29 (Phase C planning)
- **Status**: Active for single-operator deployment

### Context

Phase C trains a LoRA, then A/Bs it against baseline. To get statistically meaningful win/loss signal:

- ~30 MVs per arm (A and B), so 60 MVs total.
- Operator throughput is ~4 MVs/week.
- → **15 weeks** to complete one A/B cycle.

This is glacial by industry RLHF standards (where active learning with crowd workers compresses to days). But active learning requires:

- Multiple labelers (we have one operator).
- A reward model (we don't have one yet — Phase D scope at earliest).
- Tooling for online A/B routing with statistical guards (engineering investment).

The alternative to passive 15-week A/B is to **ship blind** (no A/B, just the new LoRA replacing baseline) and judge by gut. We rejected this: Phase A surfaced multiple cases where a "definitely better" change was actually worse on objective metrics.

### Decision

Accept **15-week passive A/B cadence**. The A/B harness routes new sessions to A or B based on a stable hash of `composition_id`, ensuring within-composition consistency. **Auto-rollback**: if mid-A/B (after the first 10 MVs/arm) the experimental arm's average composite score is >2σ below baseline, the harness flips the routing back to 100% baseline and pages the operator.

### Consequences

- Slow signal but high confidence at decision time. We're optimizing for "ship the right LoRA" not "ship a LoRA fast."
- Auto-rollback bounds the downside: a clearly bad LoRA is killed in ~2.5 weeks, not 15.
- Active learning (Phase D) is the path to faster iteration once the base infrastructure is mature.

### Revisit when

- External bearers add volume (e.g., 4× operators concurrent → 4× MVs/week → 15-week cycle becomes 4 weeks). At that scale, full active-learning may not be needed because passive A/B clears fast enough.
- Or Phase D ships the reward model + active-learning framework, at which point we can shrink cycles meaningfully.

---

## ADR-014: dual-format shot_uuid (16-char legacy, 32-char v0.8+)

- **Date**: 2026-04-20 (mcp v0.8.0)
- **Status**: Active; dual-format permanently supported

### Context

`shot_uuid` is the opaque correlation handle the orchestrator emits per-shot, threaded into `generations.shot_uuid` and `composition_clips.shot_uuid` for downstream lineage queries. Original format: 16 hex chars (64-bit), constructed via `hashlib.sha256(prompt + image_uri + position).hexdigest()[:16]`. **Deterministic** — the same shot across resumes hashes identically, which is load-bearing for rc1 lineage tables (resume-safety: re-attempts of the same shot collapse onto one shot_uuid row).

At single-operator scale (10k clips/year), 64-bit space (2^64) is overkill — collision probability negligible. But:

- The format is exposed in MCP tool responses and client-stored state (some users have it pinned in their session state).
- If we ever reach 1B+ clips (multi-tenant 10k operators × 100k clips/year × 10 years), 64-bit collisions become non-negligible (birthday paradox at 2^32).
- More immediately: **we want `shot_uuid` to be opaque and non-guessable** to make tampering harder. 64-bit is guessable in cloud-scale brute-force; 128-bit is not.

### Decision

mcp v0.8.0 emits **32-char hex shot_uuids** (128-bit, `hashlib.sha256(prompt + image_uri + position).hexdigest()[:32]` — `_shot_uuid_for` in `orchestrator.py`). Backend accepts **either** 16-char or 32-char as valid `shot_uuid` (regex `^[0-9a-f]{16}$|^[0-9a-f]{32}$`). The contract stays "opaque hex string, exact length not asserted by clients." The derivation stays **deterministic** so a resume produces the identical shot_uuid for the same shot inputs — the lineage tables (`composition_clips`, retake `parent_clip_id` joins) rely on stability across attempts.

### Consequences

- Backward compat preserved: pre-v0.8 clients keep working.
- Forward-compat preserved: new clients get 128-bit collision space.
- Collision space at 128-bit: ~3.4 × 10^38; safe against any realistic scale.
- Format-detection is regex-based, not version-based; no migration needed.

### Revisit when

- Never expected to. The dual-format is a permanent contract.

---

## ADR-015: multi-agent build sprint pattern as standard practice

- **Date**: 2026-04-29 (retro on Phase A → rc2 sprint)
- **Status**: Project methodology, all multi-week ships

### Context

The capture+validator+training plan (Phase A → rc2) was built using a multi-agent pattern:

1. **Plan agents** identified gaps in the design doc before any code was written (e.g., "schema doesn't cover composition lineage in time for retake DPO pairs").
2. **Build agents** implemented in parallel where possible (rc1 schema vs rc2 validator pipeline were independent enough to overlap).
3. **Reviewer agents** read the implementation diffs and surfaced concerns the build agents missed (license review, threading concerns, error-handling gaps).
4. **Validation agents** ran the test suite and probed real behaviors (real-tier tests per ADR-012).
5. **Doc agents** updated CLAUDE.md, CHANGELOG, MCP.md, and this DECISIONS.md to keep documentation in lockstep with the ship.

Outcomes from the rc1 → rc2 cycle:

- 5 production bugs caught **pre-merge** that would otherwise have shipped (counterfactual measured against Phase A's bug rate).
- 9 ADR-level decisions surfaced that the original plan didn't make explicit (this very document).
- Agent overhead: ~30% more wall-clock time and tokens than a single-agent ship.
- Post-merge bug rate: ~80% lower than Phase A (single-agent baseline).

### Decision

Every multi-week ship runs the **5-stage pattern**: Plan → Build → Reviewer → Validation → Doc. Each stage may use one or many agents in parallel, depending on independence of tasks.

### Consequences

- ~30% more agent overhead per ship.
- ~80% reduction in post-merge bugs (measured on rc2 cycle, n=1, but consistent with operator subjective experience).
- Docs stay in lockstep — `CLAUDE.md`, `CHANGELOG.md`, `MCP.md`, `API.md`, `DECISIONS.md`, `CAPTURE_VALIDATOR.md`, `OPERATOR_QUICKSTART.md` all updated within the same ship cycle, not deferred.
- The pattern requires good task decomposition; ships that aren't decomposable benefit less.

### Revisit when

Never, in spirit. This is project methodology. Specific stage names or agent allocations may evolve; the bar — "no multi-week ship goes Plan → Build → Done; reviewer + validation + doc agents are non-negotiable" — does not relax.

---

## ADR-016: A/B promote/deprecate thresholds (10% / -5% / p<0.05 / 30 MVs)

- **Date**: 2026-04-29 (v1.18.0-rc3 Phase C ship)
- **Status**: Active; thresholds locked for v1 of the auto-A/B harness

### Context

`scripts/ab_decision.py` runs weekly, partitions completed MVs by `_ab_arm` cohort, and computes a paired t-test on per-MV mean validator_score. The harness needs concrete thresholds for the four possible verdicts (`promote` / `deprecate` / `no_action` / `insufficient_samples`) so the operator runbook is unambiguous and the script's behavior is deterministic.

The thresholds are load-bearing across `scripts/ab_decision.py:56-59`, `PHASE_C_TRAINING_RUNBOOK.md:154-162`, `operator-tuning.md:510-514`, and `CLAUDE.md`. They were chosen during the Phase C ship but never written up as an ADR.

### Decision

Canonical thresholds (`scripts/ab_decision.py:56-59`):

- `PROMOTE_DELTA = 0.10` — candidate must beat baseline by ≥ +10% on mean validator_score.
- `DEPRECATE_DELTA = -0.05` — candidate is auto-deprecated when ≤ -5% under baseline.
- `PROMOTE_P_VALUE = 0.05` — paired t-test must reject the null at p < 0.05.
- `MIN_SAMPLES_PER_ARM = 30` — at least 30 MVs per arm; below that → `insufficient_samples` (no decision).

The asymmetric promote/deprecate gap (+10% vs -5%) reflects that shipping a worse LoRA is more expensive than holding a better one back; a smaller deprecate trigger gets bad LoRAs out faster.

### Consequences

- A/B verdicts are deterministic and reproducible from the same `generations` rows.
- Operator runbook (`PHASE_C_TRAINING_RUNBOOK.md`) can quote the numbers as guarantees, not heuristics.
- 30 MVs/arm is ~7-8 weeks at single-operator volume (4 MVs/week, 50/50 split). The first auto-promote can't happen earlier than that, by design.
- A bad LoRA gets ≥30 MV exposure before deprecation — non-zero cost. Acceptable: the operator can still trigger manual rollback via `POST /v1/system/lora/rollback` at any time.

### Revisit when

- Multi-tenant deploy adds 4× MV throughput → 30 MVs/arm hits in ~2 weeks → tighten `MIN_SAMPLES_PER_ARM` upward (30 → 50) to keep statistical power high.
- A LoRA passes `+10% / p<0.05 / 30 MVs` and ships, then post-deploy regresses on a held-out cohort → the +10% bar is too low; raise to +15%.
- Operator subjective experience says auto-promote feels "too eager" or "too cautious" after 3+ A/B cycles → revisit the deltas, not the p-value.

---

## Index of superseded / open-ended ADRs

- ADR-001 supersedes the implicit "we'll figure out training paradigm later" assumption from the original plan.
- ADR-002 supersedes the v1.12.0 default of `seamless-segment` for chained shots.
- All other ADRs are first-mover (no superseded prior decision in this doc).

## Conventions

- ADR titles are short and concrete; the body carries the nuance.
- "Locked" status means the decision will not be revisited within the named scope without new evidence triggering the revisit-when condition.
- "Active" status means the decision is in production and has no time-bounded scope.
- New ADRs append to this file. Don't renumber existing ADRs.
