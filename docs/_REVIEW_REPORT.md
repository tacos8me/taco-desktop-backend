# Cross-doc Consistency Review

**Reviewed:** 2026-04-29 / v1.18.0-rc3
**Scope:** All docs in `/mnt/nvme-1/servers/taco-backend/docs/` + repo-root `CLAUDE.md`, `AGENTS.md`, `CHANGELOG.md`.
**Approach:** Read-only. Source of truth = code (`server.py`, `validator.py`, `config.py`, `history_store.py`, `scripts/*.py`, `configs/sft_quality_lora.yaml`) + repo-root `CLAUDE.md` cheat-sheet.

## Summary

- Total inconsistencies found: 13
- Severity breakdown: P0 (wrong/dangerous): 1, P1 (drift, needs fix): 11, P2 (cosmetic): 1

## P0 — Wrong or dangerous

### TROUBLESHOOTING #7 rollback example uses non-admin bearer

- **Location:** `docs/TROUBLESHOOTING.md:519-524`
- **Disagreement:** The "use the production rollback endpoint (admin-gated)" snippet pulls the bearer from `.api_keys` (`KEY=$(grep -v '^#' .api_keys ...)`) and sends it via `Authorization: Bearer $KEY`. The endpoint is `_require_admin`-gated and rejects any bearer not in `.admin_keys` (or matching `TACO_ADMIN_KEY`).
- **Canonical:** `server.py:2861` calls `_require_admin(request)` which returns `403 admin_required` for non-admin bearers (see `API.md:604`, `operator-tuning.md:528`, `PHASE_C_TRAINING_RUNBOOK.md:186` — all use `$ADMIN_KEY`).
- **Suggested fix:** Change to `ADMIN_KEY=$(grep -v '^#' .admin_keys | grep -v '^$' | head -1)` and `Authorization: Bearer $ADMIN_KEY`. As-written the example will give an on-call operator a 403 in the middle of a quality regression incident.

## P1 — Drift, needs fix

### Stale version headers (4 docs)

- **Locations:**
  - `docs/OPERATOR_QUICKSTART.md:7` — `Server version: v1.17.0-rc2`
  - `docs/ARCHITECTURE.md:4` — `taco-backend v1.17.0-rc2 shipped`
  - `docs/UX_GAPS.md:4` — `Source versions: taco-backend v1.17.0-rc2`
  - `docs/PRIVACY_GOVERNANCE.md:3` — `Versions covered: taco-backend v1.17.0-rc5`
- **Canonical:** `v1.18.0-rc3` per `CLAUDE.md:5` and `INDEX.md:3`.
- **Suggested fix:** Bump headers; `OPERATOR_QUICKSTART.md` and `ARCHITECTURE.md` also need bodies updated for the Phase B + Phase C ships (see next two items). `PRIVACY_GOVERNANCE.md` only needs a header bump unless it adds the Phase B/C governance surface.

### ARCHITECTURE.md describes Phase B + Phase C as "deferred" — both have shipped

- **Locations:** `docs/ARCHITECTURE.md:48` ("Loops 3 and 4 are deferred to v1.18.0; the schema is ready"), `:382-398` (entire "What's deferred" section: "Phase B — Retrieval (taco-backend v1.18.0, MCP v0.8.0)" and "Phase C — SFT LoRA Training (v1.18.0+)" listed as not-yet-shipped, including `lora_applied_id` write fix and `construct_preference_pairs.py` framed as TBD).
- **Canonical:** Phase B shipped in v1.18.0-rc2, Phase C infrastructure shipped in v1.18.0-rc3. `INDEX.md:19-21` reflects this; `CLAUDE.md:9-15` documents both rcs in detail; `server.py:5005,5140,5245` host the live endpoints; `scripts/construct_preference_pairs.py`, `train_dpo_sft.py`, `ab_decision.py` exist on disk.
- **Suggested fix:** Move Phase B + Phase C into a "what's live" section. Phase D is the only remaining deferred item.

### `MAX_BATCH_QUEUE_DEPTH` default is 30, not 5

- **Locations:** `docs/configuration.md:65` (`MAX_BATCH_QUEUE_DEPTH | 5`), `:85` (example `.env`), and `CLAUDE.md:580` ("default 5").
- **Canonical:** `config.py:275` sets default to **30** (`int(os.environ.get("MAX_BATCH_QUEUE_DEPTH", "30"))`); v1.16.4 highlight at `CLAUDE.md:23` confirms `5 → 30` bump and "now env-overridable".
- **Suggested fix:** Update `configuration.md` table + example `.env` block, and the inline comment in `CLAUDE.md:580`. While there, consider adding the v1.16.4 caps (`PER_KEY_QUEUE_CAP=100`, `MAX_QUEUE_DEPTH=200`, `PER_KEY_MUSIC_CAP=20`, `PER_KEY_BATCH_CAP=20`) to `configuration.md` — the file is missing them entirely. `operator-tuning.md:24-28` has them right.

### PRIVACY_GOVERNANCE: `validator_version` example is rc2, not rc5

- **Locations:** `docs/PRIVACY_GOVERNANCE.md:136` (`Pinned config.VALIDATOR_VERSION (e.g. "1.17.0-rc2")`) and `:442` (`config.VALIDATOR_VERSION (currently "1.17.0-rc2")`).
- **Canonical:** `config.py:149` is `VALIDATOR_VERSION = "1.17.0-rc5"`. `GLOSSARY.md:140` and `operator-tuning.md:341` say rc5.
- **Suggested fix:** Two string updates.

### PRIVACY_GOVERNANCE: `shot_uuid` format is wrong on two axes

- **Location:** `docs/PRIVACY_GOVERNANCE.md:147`
- **Disagreement:** Says `shot_uuid` is `TEXT (16-hex)` and is computed as `sha256(prompt ⌷ image_uri ⌷ position)[:16]`.
- **Canonical:**
  - Format is **dual** — 16-hex (legacy mcp ≤ v0.7.x) or 32-hex (mcp v0.8+). Backend accepts both. Source: `DECISIONS.md` ADR-014 (`docs/DECISIONS.md:490-518`); `RETRIEVAL_WORKFLOW.md:402-404` says the same.
  - It's NOT a derived sha256 — it's `secrets.token_hex(8)` legacy / `secrets.token_hex(16)` v0.8+ per `DECISIONS.md:497,507`. The "stable across sessions" idea is what `shot_config_key` does (full-sha256 of generation params), not `shot_uuid`.
- **Suggested fix:** Change column type note to `TEXT (16-hex or 32-hex)` and replace the derivation note with "opaque random token; same shot across sessions hashes identically via `shot_config_key`, not `shot_uuid`."

### `OPERATOR_QUICKSTART` analyze-motion response shape is wrong

- **Location:** `docs/OPERATOR_QUICKSTART.md:421` — `Returns {tier1, tier2, tier3, composite, recommendation}`.
- **Canonical:** `validator.py:658-670` returns `{video_uri, validator_version, tier1, tier2, tier3, composite_score, recommendation, reasoning_summary, ran_at, latency_s, cached}`. Note the field is `composite_score`, not `composite`. `API.md:1602-1623` lists the right shape (modulo the `video_sha256` issue below).
- **Suggested fix:** Rewrite the one-line summary to either link to API.md or list the actual fields.

### `API.md` analyze-motion response shows non-existent `video_sha256` field

- **Location:** `docs/API.md:1605` — example payload includes `"video_sha256": "8c7f...e2"`.
- **Canonical:** `validator.py:658-670` returns `video_uri` (not `video_sha256`). `video_sha256` is the cache-key column on `validator_runs`, but is not surfaced in the API response.
- **Suggested fix:** Replace the `video_sha256` line with `"video_uri": "storage://abc-123"`.

### `TROUBLESHOOTING` rollback response shape is missing `note`

- **Location:** `docs/TROUBLESHOOTING.md:534` — `Returns {rolled_back_from, rolled_back_to, reason, applied_at}`.
- **Canonical:** `server.py:2908-2914` returns 5 fields including `note`. `API.md:617-622`, `PHASE_C_TRAINING_RUNBOOK.md:197`, `operator-tuning.md:540-541` all have 5 fields.
- **Suggested fix:** Add `, note` to the field list.

### `operator-tuning.md` Phase C pre-flight signal-strength threshold disagrees with default config

- **Location:** `docs/operator-tuning.md:427` — `WHERE signal_strength >= 0.7 AND validator_version = ...` cited as the "≥ 1000" pre-flight check.
- **Canonical:** `configs/sft_quality_lora.yaml:15` and `scripts/train_dpo_sft.py:114` set `min_signal_strength=0.5` as the default. At 0.7 you only count `validator_pass` (0.7) + `user_retake` (0.9); the actual training run with default config will draw from `composition_kept` (0.5) too. So the gate the runbook says protects you ("≥ 1000 at 0.7") is stricter than what the trainer ends up consuming. `PHASE_C_TRAINING_RUNBOOK.md:76` correctly cites `min_signal_strength >= 0.5`.
- **Suggested fix:** Lower the operator-tuning threshold to 0.5, or document that `0.7` is intentionally a higher bar than the default trainer floor (and explain why an operator should pre-flight at the stricter bar).

### ADR-007 / DECISIONS thresholds drift from ADR text

- **Locations:** `docs/DECISIONS.md:255-257` agree with canonical (`pass ≥ 0.65`, `warn 0.45-0.65`, `retake < 0.45`). Clean. **However**, DECISIONS.md has no ADR documenting the **Phase C A/B promote thresholds** (`+10%` / `-5%` / `30 MVs/arm` / `p < 0.05`). These thresholds are load-bearing across `PHASE_C_TRAINING_RUNBOOK.md:158-161`, `operator-tuning.md:510-514`, and `scripts/ab_decision.py:57-59`.
- **Canonical:** `scripts/ab_decision.py:57-59,56` (`PROMOTE_DELTA = 0.10`, `DEPRECATE_DELTA = -0.05`, `PROMOTE_P_VALUE = 0.05`, `MIN_SAMPLES_PER_ARM = 30`) — all three docs match canonical, agreement is clean.
- **Suggested fix:** Not strictly an inconsistency — flagging because DECISIONS.md is supposed to be the ADR home. Add ADR-016: "A/B promote thresholds (10%/-5%/p<0.05/30 MVs)" so the `revisit-when` is captured.

### PRIVACY_GOVERNANCE missing Phase B rate-limit governance surface

- **Location:** `docs/PRIVACY_GOVERNANCE.md:66` — "Per-tenant rate limits ✅" lists `PER_KEY_QUEUE_CAP, PER_KEY_MUSIC_CAP, PER_KEY_BATCH_CAP, PER_KEY_LORA_COUNT, PER_KEY_UPLOAD_BYTES_PER_DAY`. Doesn't mention the v1.18.0-rc2 token-bucket rate limit on `/v2/embeddings/*` and `/v2/system/bulk-revalidate` (10 req/sec, burst 10, sha256-keyed).
- **Canonical:** `server.py:4861-4862,4866` + `API.md:140`, `RETRIEVAL_WORKFLOW.md:385-388`. The bucket is keyed by `sha256(api_key)` per `server.py:4883`, which is governance-relevant per the doc's threat model.
- **Suggested fix:** Add a row covering the embeddings rate-limit and reference `API.md` for the canonical defaults.

### `OPERATOR_QUICKSTART` doesn't mention any Phase B / Phase C tooling

- **Location:** Doc-wide. `docs/OPERATOR_QUICKSTART.md` has section 3 (Training opt-in metadata, v1.17.0-rc1) but nothing about `/v2/embeddings/search`, `/v2/embeddings/recommend-loras`, `/v2/system/bulk-revalidate`, `/v1/system/lora/rollback`, the Phase C scripts, or the AB harness — features an operator setting up "the first 30 minutes on the box" today should know exist.
- **Canonical:** `INDEX.md:9-11` and `CLAUDE.md:7-21` document both phases.
- **Suggested fix:** Add a short section pointing at `RETRIEVAL_WORKFLOW.md`, `PHASE_C_TRAINING_RUNBOOK.md`, and the rollback CLI.

## P2 — Cosmetic

### `API.md` search example `shot_id` value uses ULID-like format

- **Location:** `docs/API.md:1704` — `"shot_id": "gen_01HX..."`.
- **Canonical:** `generations.id` is `TEXT (uuid4)` per `PRIVACY_GOVERNANCE.md:104` and `_save` in `history_store.py`. The illustrative ID looks like a ULID prefix; could mislead a client to expect that format. (`RETRIEVAL_WORKFLOW.md:67` uses `12453` which is also wrong shape.)
- **Suggested fix:** Use a uuid4-shaped example (e.g. `"shot_id": "0a1b2c3d-4e5f-6789-abcd-ef0123456789"`) for both docs.

## Clean checks (no issues found)

- **Composite scoring** (`0.4·tier1 + 0.2·tier2 + 0.4·tier3`) — agrees across `validator.py:19,442-451,470-473`, `CAPTURE_VALIDATOR.md`, `DECISIONS.md:252`, `GLOSSARY.md:127-136`, `operator-tuning.md`, `CLAUDE.md`. No drift.
- **Recommendation thresholds** (`pass ≥ 0.65`, `warn 0.45-0.65`, `retake < 0.45`) — agrees in `validator.py:502-508`, `DECISIONS.md:255-257`, `GLOSSARY.md:135-136`. Clean.
- **Search ranking formula** (`0.50·sim + 0.35·v_norm + 0.10·recency + 0.05·comp_kept`) — agrees in `server.py:5102-5107`, `API.md:1684-1693`, `RETRIEVAL_WORKFLOW.md:53-58`, `GLOSSARY.md:324-328`, `CLAUDE.md`. Clean.
- **LoRA recommend ranking** (`0.7·mean + 0.3·max(0,boost)`) — agrees in `server.py:5224`, `API.md:1761-1763`, `RETRIEVAL_WORKFLOW.md:101`, `GLOSSARY.md:344-346`. Clean.
- **Rate limit defaults** (10 req/sec/key, burst 10) — agrees in `server.py:4861-4862`, `API.md:140,1662`, `RETRIEVAL_WORKFLOW.md:385-388`, `CLAUDE.md:368-369`. Clean.
- **Signal strengths** (`user_retake=0.9, validator_pass=0.7, composition_kept=0.5, validator_fail=0.3`) — agrees in `scripts/construct_preference_pairs.py:79-84`, `CLAUDE.md`, `GLOSSARY.md:222-223`, `PHASE_C_TRAINING_RUNBOOK.md` (implicit). Clean.
- **SFT hyperparams** (`rank=64, alpha=64, lr=5e-4, epochs=3, micro_batch=1, ga=4, paged_adamw_32bit`) — agrees between `configs/sft_quality_lora.yaml`, `PHASE_C_TRAINING_RUNBOOK.md`, `operator-tuning.md`, `CLAUDE.md`. Clean.
- **A/B promote thresholds** (`+10% AND p<0.05` / `-5% AND p<0.05` / `<30 MVs/arm`) — agrees between `scripts/ab_decision.py:56-59,210-222`, `PHASE_C_TRAINING_RUNBOOK.md:154-162`, `operator-tuning.md:510-514`, `CLAUDE.md`. Clean.
- **Endpoint inventory** — every endpoint listed in `API.md`, `RETRIEVAL_WORKFLOW.md`, `OPERATOR_QUICKSTART.md`, `TROUBLESHOOTING.md`, `PHASE_C_TRAINING_RUNBOOK.md`, `PRIVACY_GOVERNANCE.md` resolves to a concrete `@app.<verb>` decorator in `server.py`. URLs and methods match.
- **Env vars** (`LOAD_SAPIENS=0`, `LOAD_MADMOM=1` default-on, `SAPIENS_SIDECAR_URL=http://127.0.0.1:8096`, `SAPIENS_TIMEOUT_S=60`, `LTX_REMOTE_SIDECAR_MAX_WORKERS=10`) — agree across `config.py`, `CLAUDE.md`, `operator-tuning.md`, `TROUBLESHOOTING.md`, `OPERATOR_QUICKSTART.md`. Clean.
- **INDEX.md cross-references** — every doc link in `INDEX.md` (32 links) resolves to a file under `docs/` or `CLAUDE.md`/`AGENTS.md`/`CHANGELOG.md`/`README.md` at the repo root. No broken links.
- **`MCP_PRODUCTION_LORA` rollback flow** (`409` on mismatch, atomic `.env` rewrite, restart-required, `note` field in response) — agrees between `server.py:2873-2914`, `API.md:599-650`, `PHASE_C_TRAINING_RUNBOOK.md:179-206`, `operator-tuning.md:521-551`. Clean (modulo the missing `note` field flagged for TROUBLESHOOTING above).
- **DECISIONS ADRs** — sampled ADR-001 (SFT-on-chosen + Diffusion-DPO deferred), ADR-003 (sqlite-vec), ADR-004 (Sapiens stub mode), ADR-006 (opt-out default), ADR-007 (3-tier composite), ADR-011 (validator_version scoping), ADR-014 (dual-format shot_uuid). All match the codebase exactly.
- **GLOSSARY term consistency** — sampled 12 terms (composite score, signal_strength, lora_applied_id, ranking formula, privacy gate, validator_runs cache, schema v3, schema v4, _on_job_complete, MCP_PRODUCTION_LORA, AB_AUTO_PROMOTE, motion_intent). All used consistently across other docs.

## Resolution status

Applied 2026-04-29 by docs-mega-push fix-up agent (task #13). All P0 + 11 P1 fixes applied verbatim per `Suggested fix:` lines; the single P2 was skipped per task scope.

- **P0 — TROUBLESHOOTING #7 rollback admin bearer** — fixed: `KEY=` → `ADMIN_KEY=$(grep -v '^#' .admin_keys ...)` and `Authorization: Bearer $ADMIN_KEY` at `docs/TROUBLESHOOTING.md:519-521`.
- **P1 — Stale version headers (4 docs)** — fixed: `OPERATOR_QUICKSTART.md:7`, `ARCHITECTURE.md:4`, `UX_GAPS.md:4` bumped `v1.17.0-rc2 → v1.18.0-rc3`; `PRIVACY_GOVERNANCE.md:3` bumped `v1.17.0-rc5 → v1.18.0-rc3`.
- **P1 — ARCHITECTURE.md "what's deferred"** — fixed: rewrote `:48` and the entire `## 11. What's deferred` section as `## 11. What's live (and what's still deferred)`. Phase B + Phase C moved to LIVE; Phase D is the only remaining deferred work.
- **P1 — `MAX_BATCH_QUEUE_DEPTH` default 5 → 30** — fixed: `configuration.md` table + example `.env` updated; v1.16.4 caps (`MAX_QUEUE_DEPTH=200`, `PER_KEY_QUEUE_CAP=100`, `PER_KEY_MUSIC_CAP=20`, `PER_KEY_BATCH_CAP=20`) added to both. `CLAUDE.md:580` inline comment updated to `default 30, v1.16.4+`.
- **P1 — PRIVACY_GOVERNANCE rc2 → rc5** — fixed: two string updates at `:136` and `:442`.
- **P1 — PRIVACY_GOVERNANCE shot_uuid format** — fixed: column type `TEXT (16-hex)` → `TEXT (16-hex or 32-hex)`; derivation note replaced with the `secrets.token_hex(8)` / `secrets.token_hex(16)` description per ADR-014.
- **P1 — OPERATOR_QUICKSTART analyze-motion response shape** — fixed: replaced one-liner with full 11-field shape (`{video_uri, validator_version, tier1, tier2, tier3, composite_score, recommendation, reasoning_summary, ran_at, latency_s, cached}`) at `:421`.
- **P1 — API.md analyze-motion `video_sha256`** — fixed: `"video_sha256": "8c7f...e2"` → `"video_uri": "storage://abc-123"` at `:1605`.
- **P1 — TROUBLESHOOTING rollback `note` missing** — fixed: appended `, note` to the field list at `:534`.
- **P1 — operator-tuning Phase C signal-strength threshold** — fixed: `0.7` → `0.5` at `:427`, with note about raising to 0.7 for validator_pass-or-better only.
- **P1 — DECISIONS missing ADR-016 for A/B thresholds** — fixed: appended ADR-016 ("A/B promote/deprecate thresholds (10% / -5% / p<0.05 / 30 MVs)") before the "Index of superseded" section.
- **P1 — PRIVACY_GOVERNANCE missing Phase B rate-limit row** — fixed: added a row covering token-bucket 10 req/sec/key with burst 10 on `/v2/embeddings/*` and `/v2/system/bulk-revalidate` (sha256-keyed) at `:67`.
- **P1 — OPERATOR_QUICKSTART missing Phase B / C tooling** — fixed: added new section `## 3.5 Phase B + Phase C tooling (v1.18.0)` between sections 3 and 4, with three subsections covering retrieval, training, and rollback.

- **P2 — API.md `shot_id` example uses ULID-like format** — SKIPPED per task scope (cosmetic only).
