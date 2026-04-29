# taco-backend

LTX-compatible inference server for noodle-i (image gen) + noodle-v (video gen).

**Version**: v1.18.0-rc4 (2026-04-29).

v1.18.0-rc4 highlights: **embedding-dim fix — schema rebuild at 4096 + model swap to qwen3-embed-8b**. rc1-rc3 referenced `gemma-3-12b-nvfp4` as the embedding upstream and pinned `clip_embeddings` at `FLOAT[3584]` based on a wrong premise (Gemma 3 12B is a chat model with hidden_size=3840, not an embedding model and not 3584-dim). Result: every `chat_manager.embed` call returned 5xx (`unable to start process: upstream command exited prematurely but successfully` — vLLM exits cleanly on `/v1/embeddings` without `--task embedding`); no embedding ever made it into `clip_embeddings`. Fix: (a) `chat_manager.EMBEDDING_MODEL_VERSION` now points at `qwen3-embed-8b` (Qwen3-Embedding-8B-Q8_0.gguf via llama.cpp `--embeddings --pooling mean`, 4096-dim, MTEB-tuned — the upstream was already configured in `/home/ian/config.yml`). (b) New `_migrate()` v4→v5 step drops + recreates `clip_embeddings` at `FLOAT[4096]`; `history_store.CURRENT_SCHEMA_VERSION 4 → 5`. Safe drop+recreate: `clip_embeddings_rowids` was empty in production (rc2 backfill never ran successfully because the upstream was broken). (c) `tests/test_v1_18_phase_b.py` `EMBED_DIM = 4096`. (d) New regression test `test_v1_18_schema.py::test_v5_clip_embeddings_dim` confirms the column type after migration. Operator-tuning section updated with the qwen3-embed-8b config snippet and the model-swap migration recipe (bump `CURRENT_SCHEMA_VERSION` + update `EMBEDDING_MODEL_VERSION` together). No behavioral change for any caller other than embeddings — the v5 migration is additive-safe over the v4 surface; pre-v5 DBs get the rebuild on first boot. Restart taco-backend to migrate.

v1.18.0-rc3 highlights: **Phase C training infrastructure — preference_pairs ETL + train_dpo_sft + A/B decision + lora rollback**. Closes the v1.18.0 sprint. **Infrastructure-only — no first training run.** First invocation waits until corpus crosses ~1000 pairs (~6-8 weeks at single-operator volume); operator-driven, not auto. User-locked: SFT-on-chosen for v1, Diffusion-DPO deferred to Phase C.1. New `scripts/construct_preference_pairs.py` (~360 LOC) runs as weekly cron across 4 sources — `user_retake` (signal_strength 0.9) via `parent_clip_id` lineage, `composition_kept` (0.5) via `composition_clips` ⋈ `shot_config_key`, `validator_pass` (0.7) via shot-cohort threshold, `validator_fail` (0.3) synthetic negatives — every SELECT filters by `validator_version` AND `api_key_metadata.training_opt_in = 1` AND `since_watermark` (read from `.preference_pairs_watermark`); all INSERTs use `INSERT OR IGNORE` against the rc1 `idx_pp_unique_pair_source` UNIQUE index for idempotence. New `scripts/train_dpo_sft.py` (~360 LOC) modeled on ltx-trainer's train.py: defense-in-depth defaults to dry-run (`--execute` required to consume GPU); selects unique chosen_clip_ids meeting `signal_strength >= cfg.min_signal_strength` AND `validator_version` AND `used_in_training_run_id IS NULL`, snapshots dataset to `training_runs/<run_id>/dataset.jsonl`, 90/10 train/eval held-out, loads ltx-2.3 base + PEFT LoraConfig (rank/alpha/target_modules from yaml), trains with bf16 + paged_adamw_32bit + gradient_checkpointing (LoRA-only), saves artifact + persists `training_runs` row with FULL reproducibility metadata (`training_seed` + `hyperparams_json` + `dataset_snapshot_path` + `code_sha` from `git rev-parse HEAD` + `validator_version_at_train` — all rc1 v4 columns), marks consumed pairs as `used_in_training_run_id=run_id`, registers in `lora_registry` as candidate (NOT auto-deployed). Heavy imports (peft/torch/ltx_trainer) gated behind `--execute` so dry-run + tests stay light. New `scripts/ab_decision.py` (~270 LOC) runs weekly: paired t-test on per-MV mean validator_score across `_ab_arm` cohorts (MV grouping via `composition_id`) — promote if delta ≥ +10% AND p<0.05; deprecate if delta ≤ -5% AND p<0.05; insufficient_samples <30 MVs/arm; manual Welch's-t fallback when scipy unavailable. New `configs/sft_quality_lora.yaml` (rank=64, alpha=64, q/k/v/out_proj target modules, 3 epochs, lr=5e-4, paged_adamw_32bit). New admin endpoint `POST /v1/system/lora/rollback` (~85 LOC in server.py): body `{lora_id, reason}`; verifies lora_id matches current `MCP_PRODUCTION_LORA` (409 on mismatch), sets `training_runs.deprecated_at = now()`, finds previous deployed-and-not-deprecated run, atomically rewrites `MCP_PRODUCTION_LORA=<previous>` in `.env`, returns audit shape — applies on next process restart, no hot-swap. **Privacy gate spine**: every source query filters by `training_opt_in = 1`; opt-out bearers' clips never enter training. **Validator-version scoping spine**: cross-version pairs cannot enter training. 15 new tests in `tests/test_v1_18_phase_c.py`; suite now 245 green (was 230). **Live-DB smoke**: all four sources return 0 today against the 4-week corpus — no `parent_clip_id` rows, no `shot_config_key` rows, validator scores all on rc2 (current rc5). Expected: capture-side rc1 wiring shipped but pre-rc1 history rows have NULL in those columns; new corpus accumulates behind rc5 validator. Pair construction starts finding signal once rc5+ corpus crosses ~100 retakes. Operator runbook in `docs/PHASE_C_TRAINING_RUNBOOK.md`. No restart needed for scripts; restart taco-backend to pick up the rollback endpoint.

v1.18.0-rc2 highlights: **Phase B retrieval backend** — sqlite-vec virtual table + 3 new endpoints + per-key rate-limit middleware + `lora_applied_id` persistence + Phase B observability counters. Layered on top of the rc1 schema v4 keystone. New `chat_manager.embed` / `embed_batch` proxy llama-swap `/v1/embeddings` (3584-dim Gemma → float32-LE bytes ready for sqlite-vec). `history_store` loads sqlite-vec extension BEFORE `PRAGMA journal_mode=WAL`; module-level `SQLITE_VEC_AVAILABLE` flag propagates failure gracefully (backend boot never crashes on missing extension); `clip_embeddings(id TEXT PK, embedding FLOAT[3584], embedding_model_version TEXT)` virtual table created when load succeeds. Three new endpoints: `POST /v2/embeddings/search` (privacy-gated semantic search over the caller's own clips, ranking `0.50·sim + 0.35·v_norm + 0.10·recency + 0.05·comp_kept`), `POST /v2/embeddings/recommend-loras` (similarity-then-group LoRA aggregation, ranked by `0.7·mean + 0.3·boost`), `POST /v2/system/bulk-revalidate` (admin-gated re-validator on `validator_version != target` rows; default `dry_run=true`). Token-bucket rate-limit middleware: 10 req/sec/key with burst 10, applied only to `/v2/embeddings/*` and `/v2/system/bulk-revalidate`; existing endpoints unaffected. `lora_applied_id` write fix: every video v2 endpoint (text-to-video / image-to-video / audio-to-video / retake / video-outpaint / video-hdr) now captures `body.lora.id` + `body.lora.strength` via `_lora_applied_pair(body)` and threads them into `job.params` via the existing `_HISTORY_ONLY_PARAMS` strip path → `worker_loop.history.save()` writes `generations.lora_applied_id` end-to-end (was silently NULL on every prior rc; required by `recommend_loras`). `/v1/system/metrics` extended with `embeddings` block (search totals + rolling p50/p95 latency over 1000-call window + recommend_loras/bulk_revalidate counters). New `scripts/backfill_prompt_embeddings.py` (idempotent, resumable; `--dry-run` / `--limit N` / `--rebuild` / `--sleep-ms N`). Adds `sqlite-vec>=0.1.9` dep. 13 new tests in `tests/test_v1_18_phase_b.py`; suite now 230 green (was 217). **Privacy gate** is the spine: every search/recommend query filters by `api_key_hash` of caller — bearer A cannot ever surface bearer B's rows; verified by `test_embeddings_search_privacy_gate`. Restart taco-backend, then optionally run the backfill script. See `docs/operator-tuning.md` "Embeddings + sqlite-vec" for setup.

v1.18.0-rc1 highlights: **schema v4 — keystone migration for Phase B (retrieval) + Phase C (SFT LoRA training)**. `history_store.CURRENT_SCHEMA_VERSION 3 → 4`, additive: 2 new nullable columns on `generations` (`motion_intent`, `embedding_model_version`), 1 new column + 2 new indexes on `preference_pairs` (`validator_version`, `idx_pp_validator_version`, `idx_pp_unique_pair_source` UNIQUE on `(chosen_clip_id, rejected_clip_id, signal_source)` for `INSERT OR IGNORE` idempotence in Phase C multi-source aggregation), 5 new columns on `training_runs` for reproducibility (`training_seed`, `hyperparams_json`, `dataset_snapshot_path`, `code_sha`, `validator_version_at_train`). Single-startup migration via the existing `_migrate()` ladder; pre-v4 rows get NULL in new columns; idempotent re-run safe; v4 DBs opened by pre-v4 code silently ignore the extra columns (additive safety verified by `test_v4_db_loaded_by_v3_code_silently_ignores_extra_columns`). **Dead-letter status** of `generations.prompt_embedding` BLOB column documented in `_migrate()` docstring + this file: at 14KB/row × 1M rows it exceeds SQLite's practical column-storage envelope, so Phase B will land embeddings in a sqlite-vec virtual table (`clip_embeddings`) instead. The BLOB column is retained nullable for backward compat but never written to from v1.18.0-rc1 forward — removal candidate for v1.19+ once zero readers confirmed. **This rc1 ships ONLY the schema** — no feature wiring, no new endpoints, no extension loads; downstream Phase B/C feature ships (sqlite-vec extension load, embeddings endpoint, retrieval MCP tools, pair construction script, training pipeline, A/B framework) land in v1.18.0-rc2+. 10 new tests in `tests/test_v1_18_schema.py`; three v17 tests had hardcoded `user_version == 3` literals updated to `== CURRENT_SCHEMA_VERSION` so the v3-surface intent survives every future ladder bump; suite now 217 green (was 207). Restart taco-backend to migrate on next boot.

v1.17.0-rc5 highlights: **motion_intent wire-through + dispatch metrics + dead-column docs**. Three audit-surfaced fixes folded in. (a) `AnalyzeMotionRequest` now declares `motion_intent: str | None = Field(default=None, max_length=200)` — MCP v0.7.0 was already forwarding it from `quality_validation.motion_intent_map[shot_idx]` but Pydantic's `extra="ignore"` was silently eating it; threaded through `validator.run_all_tiers()` → `_run_tier3_judge(motion_intent=...)`, rendered conditionally into the tier-3 user-message text (line omitted when `None` so rc4 baseline is byte-identical for non-MCP callers). `JUDGE_PROMPT_V1` updated to instruct the judge to reconcile against intent. Passive `_dispatch_validator` doesn't carry intent — only the synchronous `/v2/video/analyze-motion` path does. (b) New `GET /v1/system/metrics` endpoint exposes `_validator_dispatch_counter` — process-local counts of `success` / `failure` / `skipped_not_video` / `skipped_opt_out` / `skipped_validator_disabled` plus computed `failure_rate_pct = 100 * failure / max(1, success+failure)` (skips excluded). Auth-gated when `API_KEYS` is configured. Catches multi-day silent regressions (e.g. Gemma outage NULLing a cohort) that the WARN-only logging in `_dispatch_validator` doesn't surface above the noise floor. (c) `history_store._migrate` docstring annotates v3 columns by writer status: WRITTEN (`parent_clip_id` / `shot_uuid` / `shot_config_key` / `validator_*`) vs DEAD-LETTER (`composition_id` — `composition_clips` is canonical) vs FORWARD-LOOKING (`prompt_embedding` Phase B, `validator_artifact_uri` post-real-Sapiens, `lora_applied_*` post-fusion-hook). `config.VALIDATOR_VERSION` bumped `rc4 → rc5` (JUDGE_PROMPT_V1 changed — cache invalidation). 7 new tests; suite now 207 green (was 200).

v1.17.0-rc4 highlights: **critical validator pipeline bug fixes — every rc2/rc3 dispatch was a silent false-pass**. End-to-end audit found that tier-1 RAFT crashed on non-square inputs (`input image H and W should be divisible by 8` — `h_target=147` for 1920×1080 sources), which left `tier1_summary={}`, which crashed tier-3 Gemma judge format-strings (`None:.3f`), which left tier-2 stub as the lone composite contributor, which `composite()` rescaled to `score=1.0/pass`. Three code fixes: (a) `validator.py:_decode_video_frames_for_flow` snaps both `h_target` and `w_target` to nearest /8 multiple with a min floor of 8; (b) `validator.py:_run_tier3_judge` pre-extracts tier1 values with None checks and falls back to `"Tier1 (optical flow): failed (no optical flow data)"` when tier1 didn't produce real numbers; (c) `validator.py:composite` requires at least one of tier1/tier3 to have produced a score before allowing any positive recommendation — when both are absent, returns `{composite_score: None, recommendation: "error", reasoning_summary: "all_required_tiers_failed: ..."}`. Plus: `server.py:lifespan` now `mkdir -p`'s `VALIDATOR_ARTIFACTS_DIR` and runs a one-time scrub of `validator_runs` + `generations` rows with `validator_version IN ('1.17.0-rc2', '1.17.0-rc3')` AND payload `tier1: null` / `tier3: null` (bounded by validator_version so rc4+ rows never match — safe to leave running). `_stop_cuda1_tenants` / `_restore_cuda1_tenants` now log enter/exit with the gated unit list. `config.VALIDATOR_VERSION` bumped `1.17.0-rc2 → 1.17.0-rc4`. 6 new tests; suite now 200 green. Restart taco-backend after merge to apply the fixes + run the scrub. rc5 wires motion_intent_map through to the Gemma judge and adds dispatch failure metrics.

v1.17.0-rc3 highlights: **rc2 polish** — fixes `_is_training_opted_in` docstring (which contradicted the correct opt-out-by-default implementation) and gates `_stop_cuda1_tenants` on the `LOAD_*` flag tuple component (turbo entry no longer emits a spurious `systemctl stop sapiens-sidecar` when `LOAD_SAPIENS=0`, matching the existing `_restore_cuda1_tenants` semantics). One new regression test; suite now 194 green.

v1.17.0-rc2 highlights: **validator pipeline** — RAFT in-process + Sapiens via sidecar (currently stub) + Gemma judge via existing chat_manager + composite scoring + caching + on-complete dispatch + turbo coordination. Three tiers: tier-1 RAFT-small on cuda:0 (lazy-load + evict; ~150 ms/clip; produces `dynamic_degree`/`flow_windows[4]`/`motion_smoothness`); tier-2 sapiens sidecar at `127.0.0.1:8096` via new `sapiens_client.py` (mirrors `madmom_client.py`; stub-tolerant — `{"stub": true}` is treated as skipped, contributing 0.2·1.0 to composite, not failed); tier-3 Gemma judge via `chat_manager` + `CHAR_VISION_MODEL=gemma-4-31b-it` with strict-JSON schema validation (`JudgeResponseV1`). Composite formula: `0.4·tier1_norm + 0.2·tier2 + 0.4·tier3`; recommendation `pass` ≥ 0.65, `warn` 0.45-0.65, `retake` < 0.45 OR tier3.verdict=="retake". Cached via `validator_runs(video_sha256, validator_version)` UNIQUE index. New endpoint `POST /v2/video/analyze-motion` synchronous + reuses `run_all_tiers()`. New `_on_job_complete` callback chained from `_decr_queue_on_complete` fires `_dispatch_validator(job)` as fire-and-forget task when (a) status=COMPLETED, (b) type ∈ video types, (c) `_is_training_opted_in(api_key)` — looks up `api_key_metadata.training_opt_in` (default opt-out for unknown keys). Turbo-mode coordination: `sapiens-sidecar` added to `_stop_cuda1_tenants` + `_restore_cuda1_tenants` (gated on `LOAD_SAPIENS`). `LOAD_SAPIENS=0` default — operator flips after diff review. New env vars in `config.py`: `SAPIENS_SIDECAR_URL`, `SAPIENS_TIMEOUT_S`, `LOAD_SAPIENS`, `VALIDATOR_VERSION="1.17.0-rc2"`, `VALIDATOR_ARTIFACTS_DIR`, `JUDGE_PROMPT_V1`. 20 new tests in `tests/test_v1_17_validator.py`; suite now 193 green.

v1.17.0-rc1 highlights: **schema v3 + composition lineage + retake provenance + api_key_metadata**. First wave of the capture+validator machine (per `plans/melodic-sniffing-beacon.md`). `history_store.CURRENT_SCHEMA_VERSION 2 → 3`, additive: 11 new nullable columns on `generations` (validator scoring fields + `parent_clip_id` + `shot_uuid` + `shot_config_key` + `composition_id` + `lora_applied_*` + `prompt_embedding`), 3 new indexes (shot_config_key / parent_clip_id / composition_id), 5 new tables (`composition_clips` inverted-index, `validator_runs` cache, `preference_pairs` DPO data, `training_runs` ledger, `api_key_metadata` per-key training_opt_in). Single-startup migration via the existing `_migrate()` ladder; pre-v3 rows get NULL in new columns; idempotent re-run safe. On first v2→v3 migration, `api_key_metadata` is seeded from `.api_keys` with `training_opt_in=1` for every non-comment line (single-tenant deploy default — ON globally, opt-out only). `.api_keys` is only read during seed, never modified. Wiring: `POST /v2/compositions/{id}/export` writes `composition_clips` lineage rows before submitting (best-effort, won't block export); `POST /v2/retake` populates `parent_clip_id` via new `find_id_by_result_uri()` helper for DPO pair construction downstream; `_dispatch_job*` strip v3 history-only fields from `job.params` before splatting into manager kwargs. `worker_loop`'s `on_complete` callback (v1.8.2) verified wired — v1.17.0-rc2 will register validator dispatch there. No breaking changes; no dispatch/sampler/filter-graph changes. Restart taco-backend to migrate on next boot. 12 new tests in `tests/test_v1_17_schema.py`. v1.17.0-rc2 follows with the validator sidecar (RAFT + Sapiens-2 + Gemma judge).

v1.16.4 highlights: **rate-limit caps scaled for heavy-MV operators**. `MAX_QUEUE_DEPTH 30 → 200` (now env-overridable), `PER_KEY_QUEUE_CAP 15 → 100` (half-global ratio preserved), `PER_KEY_MUSIC_CAP 5 → 20`, `PER_KEY_BATCH_CAP 5 → 20`, `MAX_BATCH_QUEUE_DEPTH 5 → 30` (now env-overridable). Sized for 200-clip `cut_music_video` sessions with mcp v0.4.4 parallel clip dispatch — a single bearer can now have ~100 jobs in flight against a 200-deep global queue with headroom for other tenants. All caps remain overridable via `MAX_QUEUE_DEPTH` / `PER_KEY_QUEUE_CAP` / `PER_KEY_MUSIC_CAP` / `PER_KEY_BATCH_CAP` / `MAX_BATCH_QUEUE_DEPTH` env vars. Pure config bump; no code/schema/wire changes. Requires server restart to take effect.

v1.16.3 highlights: **export composition fix — `storage_uri` fallback for clips without `historyId`**. Long-standing contract gap surfaced by users with flash inserts in their shot lists. The MCP orchestrator has always emitted clips with EITHER `historyId` (LTX-generated) OR `storage_uri` (synthetic flash inserts that don't ride history.db at all), per the comment at orchestrator.py:2014-2015. The backend's `export_handler.export_composition` was never updated to honor that contract — line 164 unconditionally accessed `clip["historyId"]` and `KeyError`'d on flash-only or mixed compositions. Live error: `KeyError: 'historyId'` at export_handler.py:164. Fix: added a branch that tries `historyId` first, falls back to `storage_uri` resolved via `UploadStore.resolve(...)`, raises a clear `ValueError` when neither field is present. 4 new regression tests cover storage_uri-only clips, mixed historyId+storage_uri compositions, neither-present rejection, and storage_uri-missing-on-disk handling. No MCP change needed; restart taco-backend to pick up the fix.

v1.16.2 highlights: **composition export quality knobs**. User reported visible blocking on `POST /v2/compositions/{id}/export` output — the hardcoded `-c:v libopenh264` with no quality flags was emitting ~4-8 Mbps default with no CRF/profile control. Switched the default to `libx264 + CRF 18 + preset=medium + profile=high + yuv420p` (visually-transparent quality at sane filesizes); audio default bumped from `192k → 256k`. All knobs overridable per-export via the request body (`output_encoder`, `output_crf`, `output_preset`, `output_profile`, `output_video_bitrate`, `output_audio_bitrate`). Setting `output_video_bitrate` on a CRF encoder switches to 1-pass ABR with `-maxrate`/`-bufsize`. Filter graph is byte-identical to v1.16.1 — encoder-args change only. New helper `_resolve_ffmpeg_binary()` auto-detects the best ffmpeg binary in PATH (prefers `/usr/bin/ffmpeg` over conda's `--disable-gpl` build that lacks libx264); override via `TACO_FFMPEG_BIN`. Backward-compat: existing callers with empty body or just `audio_uri` get higher quality automatically; API shape additive-only. See `docs/operator-tuning.md` "Export quality" section.

v1.16.1 highlights: **rate-limit caps + HTTP layer hardening**. Real-world MV submissions (28 a2v jobs sequentially with 1.5 s pacing) were tripping `per_key_queue_full` 24/28 times. Caps raised: `PER_KEY_QUEUE_CAP 3 → 15`, `MAX_QUEUE_DEPTH 10 → 30`, `PER_KEY_MUSIC_CAP 2 → 5`, `PER_KEY_BATCH_CAP 2 → 5` (all overridable via env). Uvicorn now runs with `--limit-concurrency 200 --backlog 4096` to fix `Connection reset by peer` under 28+ concurrent client polls (kernel SYN backlog overflow). Systemd unit gets `LimitNOFILE=16384` for httpx-pool + WAL + client-socket headroom. Also: pin Gemma IT-NVFP4 snapshot SHA `90152908233cae111ec85f78f3d69bdcbd1c6ffd` so v1.16.0's `GEMMA_VARIANT=gemma-3-12b-it-nvfp4` actually resolves (the HF download landed at the parent of `/hub/`, not under it). See `docs/operator-tuning.md`.

v1.16.0 highlights: two opt-in features. (a) `GEMMA_VARIANT=gemma-3-12b-it-nvfp4` — a 3rd entry in `_GEMMA_VARIANTS`. The IT (instruction-tuned) NVFP4 snapshot follows rewrite instructions cleanly, unlike PT (default) which produces literal continuations. Recommended whenever `enhance_prompt=true`. (b) `analyzer="madmom"` on `POST /v1/music/analyze` — routes to a new CPU-only sidecar at `127.0.0.1:8095` (BSD-licensed, separate venv at `/mnt/nvme-1/servers/madmom-sidecar/` because madmom needs `numpy<1.24`). ~+8% downbeat accuracy on cross-genre pop. Default stays `"librosa"` — byte-identical to v1.15.x for existing callers. `LOAD_MADMOM=1` is the new default; sidecar failures surface as explicit `503` with no silent fallback.

v1.15.3 highlights: **enhance_prompt no longer crashes the run** when the Gemma tokenizer lacks a `chat_template`. Both PT (default) and sikaworld-NVFP4 variants ship without one, so `tokenizer.apply_chat_template` (called by ltx-core's `base_encoder.py:58`) was raising and tearing down the entire generation. Now wrapped in a broad try/except that logs a clear WARN naming the variant, then continues with the raw prompt. Caller-visible behavior: enhancement becomes a no-op until the operator points `GEMMA_VARIANT` at an instruction-tuned snapshot whose tokenizer carries a chat_template (e.g. v1.16.0's `gemma-3-12b-it-nvfp4`).

v1.15.2 highlights: **cleanup_loop sweeps zombie PROCESSING jobs**. `/health` was reporting `queue.processing=2` while `/v1/system/workers` showed only 1 busy worker — a Modal `httpx.ReadError` had failed to propagate, leaving an orphaned PROCESSING job that the cleanup loop never touched (it only swept terminal-status jobs). Added a 30-minute threshold: any PROCESSING job older than that is marked FAILED with code `"zombie"` and surfaces cleanly through the standard error path so `resume_music_video` can distinguish zombies from real failures.

v1.15.1 highlights: **clip.speed silently dropped on single-clip exports**. `export_handler.py` had a `len(clip_paths) == 1 and audio_path is None` short-circuit that returned raw bytes regardless of `clip.speed` or `tailTrimFrames`. `speed=0.5` / `speed=1000` / `speed=-1` produced byte-identical output with no caller-visible signal — silent corruption. Tightened the shortcut to require `speed == 1.0 AND tailTrimFrames == 0`; any transform falls through to the ffmpeg path where `_norm()`'s setpts applies. Two regression tests cover speed and tail-trim singles.

v1.14.1 highlights: **CORS fix**. (a) `allow_origin_regex` in `server.py` widened from `localhost|192.168.X.X` to also include `([a-zA-Z0-9-]+\.)?noodlefinger\.io` — production FE origins (`i.noodlefinger.io`, `v.noodlefinger.io`, `m.noodlefinger.io`) now pass the CORS preflight check. (b) `check_api_key` middleware now short-circuits on `request.method == "OPTIONS"` so unauthenticated browser preflights flow through to `CORSMiddleware` and get a proper response with `access-control-allow-origin` header. Without (b), the auth middleware's 401 ate the preflight before CORS could ever respond, regardless of whether the origin was allowed. Symptom: FEs failed silently to load any cross-origin API call. Root-caused via missing `access-control-allow-origin` in the 400 preflight response.

v1.14.0 highlights: `POST /v2/video-hdr` — IC-LoRA HDR-expansion endpoint for LDR→HDR video transform. Uses `Lightricks/LTX-2.3-22b-IC-LoRA-HDR` (registered as `ic-lora-hdr`, strategy `ic_lora_hdr`). Architecturally piggybacks on `_run_outpaint` with `target == source` and `position="center"` — the actual generation codepath is **unchanged from v1.13.0**, HDR is purely additive scaffolding (new endpoint + new `JobType.VIDEO_HDR` + new sidecar `case` branches). Backend snaps source dims to nearest /64 multiple via `history_store._probe_video_dims`. All three sidecars (local cuda:1, Modal, RunPod) gained a `case "video-hdr":` branch routing to the same `generate_outpaint(...)` they already used for outpaint.

v1.13.0: `GET /v1/system/workers` live-worker introspection endpoint; dashboard "Live Workers" panel; `Job.worker_id` tracking; Modal pool max raised from 4 → 10. Peak concurrent-video capacity is 2 local + 10 Modal + 2 RunPod = **14 concurrent video workers**.

## Quick lookup

| Topic | Go to |
|---|---|
| Adding a new endpoint | [Structure](#structure) + [API contract](#api-contract) + [Conventions](#conventions) |
| Changing GPU behavior (swap, turbo, pool) | [GPU topology](#gpu-topology) + [Turbo mode](#turbo-mode-v12-hardened-v15) + [Remote-sidecar pool](#remote-sidecar-pool-v16) |
| Inspecting live worker state | `GET /v1/system/workers` (v1.13.0) + dashboard Live Workers panel |
| Touching LTX generation (denoising, stages, sampler) | [Conventions](#conventions) + [Critical patterns](#critical-patterns) + `split_model_manager.py` |
| Chain conditioning (v1.12 segment mode) | `split_model_manager.py` entry in [Structure](#structure) + [Conventions](#conventions) + CHANGELOG v1.12.0 |
| Touching Flux generation | [Flux pipeline details](#flux-pipeline-details) + `flux_manager.py` |
| LoRA plumbing | [LTX LoRA](#conventions) / [Flux LoRA](#flux-lora-v11--folder-drop-discovery-adapter-mode) / [IC-LoRA outpaint](#ic-lora-video-outpaint-v170) / [IC-LoRA HDR](#ic-lora-video-hdr-v1140) |
| Async job queue, phases, SSE | [v2 job observability](#v2-job-observability-v116--v117) + `job_queue.py` |
| History + reproducibility | [Generation history](#generation-history-history_storepy) |
| Sidecars (ACE / JoyAI / ERNIE / madmom / sapiens / LTX-remote) | [ACE](#ace-music-sidecar-v12) / [JoyAI](#joyai-image-edit-sidecar-v12-migrated-from-cuda0) / [ERNIE](#ernie-image-sidecar-v13) / [madmom](#madmom-downbeat-sidecar-v1160) / [sapiens](#sapiens-pose-sidecar-v1170-rc2) / [Remote pool](#remote-sidecar-pool-v16) |
| Validator pipeline (RAFT + Sapiens + Gemma judge) | [Validator pipeline (v1.17.0-rc2)](#validator-pipeline-v1170-rc2) — `POST /v2/video/analyze-motion` + `_on_job_complete` dispatch + `validator_runs` cache |
| Phase C training infra (preference_pairs ETL + train_dpo_sft + A/B + rollback) | [Phase C training pipeline (v1.18.0-rc3)](#phase-c-training-pipeline-v1180-rc3) — `scripts/construct_preference_pairs.py` + `scripts/train_dpo_sft.py` + `scripts/ab_decision.py` + `POST /v1/system/lora/rollback` |
| Video outpaint | [IC-LoRA video outpaint](#ic-lora-video-outpaint-v170) |
| MV editing grammar / shot lists / beat-aligned cuts | `docs/MV_EDITING.md` (v1.15.0 — `/v1/music/analyze` + `clip.speed` + `transition.audioLeadFrames`) |
| Music structure analyzer (madmom) | [madmom downbeat sidecar (v1.16.0)](#madmom-downbeat-sidecar-v1160) — `POST /v1/music/analyze` with `analyzer="madmom"` |
| Dashboard + live tuning | [Dashboard](#dashboard-and-gpu-telemetry-v12-advanced-controls-v13) + [Generation config](#generation-config-v13) |
| Client-facing API shape | `docs/API.md` (canonical) |
| LLM-driven workflows | `docs/MCP.md` (noodlefinger-mcp tier-0 + tier-1) |
| Shipped-feature archaeology | `CHANGELOG.md` + `AGENTS.md` (per-version deltas) |
| Rate-limit / concurrency tuning | `docs/operator-tuning.md` (v1.16.1) |
| Per-shot audio slicing in cut_music_video orchestrator | noodlefinger-mcp v0.4.2 (see `flows.json`) |
| Parallel clip dispatch in cut_music_video orchestrator | noodlefinger-mcp v0.4.4 (DAG-aware, auto-scales to live worker count) |
| Donor-aware tailTrimFrames in cut_music_video orchestrator | noodlefinger-mcp v0.4.5 (anchor-mode followers don't consume segment_uri → predecessor gets trim=0; fixes silent 9-frame loss on all-anchor shot lists) |

## Structure
- `server.py` — FastAPI app, all HTTP endpoints, job queue dispatch, history + approved-images APIs, batch scheduler, turbo mode + remote pool, dashboard. ~3.9 k lines; key anchors: `_enter_turbo_mode` (~:1712), `_exit_turbo_mode` (~:1795), `_scale_remote_pool` (~:1604), `_dispatch_job` (~:320), `_dispatch_job_turbo` / `_dispatch_job_turbo_remote` (~:1520/1666), `v2_video_outpaint` (~:2685). v1.10.0 added `POST /v2/video/extract-frames` (PyAV multi-frame → lossless PNGs). v1.13.0 adds **`GET /v1/system/workers`** — live-worker introspection: returns `{turbo_active, workers: [{id, provider, slot, status, current_job}, ...]}` with `id` shaped as `local-0` / `local-1` / `modal-<N>` / `runpod-<N>` and `status` inferred from in-flight `Job.worker_id` in the store (zero external calls). Powers the dashboard Live Workers panel. v1.12.0 adds **`POST /v2/video/extract-segment`** — single-pass PyAV decode + H.264 re-encode of a contiguous frame range as an MP4 upload; body `{video_uri, start_frame, num_frames ∈ {9,17,25,33}}`, returns `{segment_uri, width, height, num_frames, fps}`. Same bearer + capability-URL security + `_FRAME_EXTRACT_SEMAPHORE(2)` + 30 s timeout + `PER_KEY_UPLOAD_BYTES_PER_DAY` quota as extract-frames. v1.12 also adds `AudioToVideoRequest.segment_uri` / `ImageToVideoRequest.segment_uri` (optional str, 3-way mutually exclusive with `image_uri` and `keyframes` via `@model_validator`). **Experimental** — gated behind FE `flags.v112_seamless_segment`; legacy v1.11.5 keyframes path still fully supported.
- `split_model_manager.py` — Single-GPU LTX pipeline: SingleGPUModelBuilder + CachingModelFactory, CFG++ sampler (default), BatchSplitAdapter for multi-pass batching. Houses all `_run_*` methods (t2v / i2v / a2v / retake / outpaint / HQ). ~2.2 k lines. v1.12.0 adds multi-frame video-segment chain conditioning (**experimental**): new module-level `_build_segment_conditioning_latent(segment_path, target_h, target_w, dtype, device, video_encoder)` (mirrors `_build_outpaint_reference_latent` — decodes via `decode_video_from_file`, `video_preprocess`, returns multi-latent-frame tensor). `_image_conds_for_keyframes(..., segment_path=...)` short-circuits to the segment path when set: a single `VideoConditionByLatentIndex(latent=multi_frame_latent, latent_idx=0, strength=1.0)` hard-pins target's head latents [0, 1] (= 9 consecutive target pixel frames via LTX's causal VAE 8k+1 scheme) at every sigma step. `_run_i2v` / `_run_a2v` (and `generate_image_to_video` / `generate_audio_to_video`) accept a `segment_path` kwarg forwarded through threadpool submission. All 3 sidecars (local `ltx-sidecar`, Modal, RunPod) accept `segment_path`; Modal + RunPod also accept `segment_b64` — `_dispatch_job_turbo_remote` base64-encodes and stages to `/tmp/<uuid>.mp4`. When `segment_path` is absent, `_image_conds_for_keyframes` falls through to `combined_image_conditionings` — the v1.11.5 classical LTX semantic (frame 0 hard-pin via `VideoConditionByLatentIndex`; frames 1+ soft-guide via `VideoConditionByKeyframeIndex`) is fully preserved for back-compat. `ImageConditioningInput(...)` uses upstream `DEFAULT_IMAGE_CRF=33` since v1.12.2 (commit `386356a`). v1.11.4 had set `crf=0` to avoid CRF-33 frame-0 blur; that fix was reverted within 25 hours after client reports of motionless a2v + degraded 1440p quality. Root cause: LTX's VAE encoder was trained on CRF-33 statistics, so `crf=0` images are out-of-distribution → "too clean" latent pins too hard via `VideoConditionByLatentIndex` → temporal attention propagates the pin → motion suppressed. Effect scales with resolution (1440p ≈ 1.78× the latent tokens of 1080p, proportionally worse). Tradeoff accepted: frame 0 carries a barely-visible CRF-33 compression artifact (the v1.11.3-and-earlier baseline). Per-keyframe `crf` override remains exposed at the request boundary if a specific case needs it. Dispatch diagnostic `logger.warning("a2v keyframes=... model=... frames=...")` at start of `_run_a2v` / `_run_i2v`. **Known limitation** (deferred to v1.13): LTX's causal VAE replicates a segment's frame 0 for padding, so the encoded 9-frame latents aren't bit-identical to what the prior clip's full-length encoding produced for the same frames — residual RMSE 1-3 on a 0-255 scale, visually imperceptible in most content. v1.11.3 attempted an earlier routing fix through `image_conditionings_by_replacing_latent` to hard-pin multiple frames via 3 PNG keyframes, but `VideoConditionByLatentIndex` operates at latent-frame granularity (a slideshow of held stills in pixel frames 1-16), reverted in v1.11.5; see `docs/debug-v1.11.3-chain-conditioning.md`. v1.12's multi-frame segment encoding is the architecturally correct fix.
- `flux_manager.py` — Flux 2 image generation: per-request LoRA adapter mode on cuda:0, bf16, `enable_model_cpu_offload` on Dev
- `ace_client.py` — ACE music generation sidecar client (httpx → ace-step on cuda:1:8001)
- `chat_manager.py` — Proxies /v1/chat/completions to llama-swap (supports per-request model override for vision ranking)
- `ernie_client.py` — ERNIE-Image sidecar client (httpx → cuda:1:8094), swaps with JoyAI on cuda:1
- `joyai_client.py` — JoyAI-Image-Edit sidecar client (httpx → cuda:1:8092)
- `madmom_client.py` — madmom downbeat-detection sidecar client (httpx → CPU:8095, v1.16.0). Opt-in via `analyzer="madmom"` on `POST /v1/music/analyze`.
- `sapiens_client.py` — sapiens pose-temporal-stability sidecar client (httpx → cuda:1:8096, v1.17.0-rc2). Stub-tolerant: passes `{"stub": true}` payloads through verbatim for `validator.composite()` to treat as "skipped, no penalty". `LOAD_SAPIENS=0` default.
- `validator.py` — v1.17.0-rc2 validator pipeline. Tier 1 (RAFT in-process, raft_small on cuda:0, lazy-load + evict), Tier 2 (sapiens sidecar, stub-tolerant), Tier 3 (Gemma judge via existing `chat_manager` + strict-JSON `JudgeResponseV1`). Composite scoring `0.4·t1 + 0.2·t2 + 0.4·t3`, recommendation pass/warn/retake. `run_all_tiers()` is the entrypoint used by both `POST /v2/video/analyze-motion` and `_on_job_complete` dispatch. Caches via `validator_runs(video_sha256, validator_version)` UNIQUE index from rc1 schema.
- `ltx_sidecar_client.py` — LTX video sidecar client. One primary `ltx_sidecar` (local cuda:1:8093) + optional `ltx_remote_sidecar` (Modal, configured via `LTX_REMOTE_SIDECAR_URL`). `generate()` supports base64 media inlining (`audio_b64` / `image_b64` / `video_b64`) for remote sidecars that can't see the local `uploads/` filesystem, plus outpaint extras (`position`, `conditioning_strength`, `skip_stage_2`).
- `job_queue.py` — Async job queue: submit (202), poll, result, cancel; saves to history on completion. `JobType` enum includes `VIDEO_OUTPAINT` (v1.7.0). v1.13.0 adds `Job.worker_id: str | None` — set by each dispatch path (`local-0`, `local-1`, `modal-<N>`, `runpod-<N>`) and read by `GET /v1/system/workers` to infer per-worker busy/idle state.
- `upload_store.py` — UUID file storage for uploads and job results
- `history_store.py` — SQLite-backed per-API-key generation history with thumbnails; schema v2 (params_json, gen_config_json, seed, enhanced_prompt)
- `lora_registry.py` — Flat-dir LTX LoRA storage with registry.json index. IC-LoRA outpaint LoRA lives here with `strategy="ic_lora_outpaint"`.
- `flux_lora_registry.py` — Flux LoRA folder-drop discovery (filesystem-only, no registry.json)
- `composition_store.py` / `export_handler.py` — Composition export (video concat / transcode). v1.10.0: per-clip `tailTrimFrames: int` (default 0) trims the last N frames of each input via `trim=end_frame=<kept>` prepended before the v1.9.8 `setpts=PTS-STARTPTS,format=yuv420p` normalization. Trimmed durations cascade into the v1.9.6 beat-gap atrim and the v1.9.9 force-IDR seam cumsum so audio slicing and keyframe timestamps stay aligned. Last clip and xfade branch always skip trim; single-clip exports zero it out. v1.11.2: per-clip `audioDurationSec: float | None` (optional, FE-preferred) overrides the beat-gap atrim slice per clip — when numeric + positive it's used verbatim for `atrim duration=`, decoupling the audio side from `effective_duration`. Absent → v1.11.1 clamp behavior (`min(gap, effective_duration)`) preserved exactly. v1.12 FE guidance: `chainMode="seamless-segment"` compositions set `tailTrimFrames=9` on non-final clips **whose follower actually consumes `segment_uri`** (i.e., the next non-skip shot is a primary a2v/i2v with no `image_uri`). Anchor-mode followers (image_uri set) use the 3-way-mutex image_uri path, never the segment chain — their predecessor must use `tailTrimFrames=0` or the predecessor silently loses 9 frames of unique content. The mcp orchestrator computes this automatically as of v0.4.5 (walk-forward through transitive insert/flash/slideshow); raw-API callers building compositions over mixed shot lists must mirror this logic. The 9 pinned head frames of a primary follower re-show the donor's tail exactly once; `export_handler.py` is unchanged — the math is abstract over trim value. No -shortest in the beat_synced branch already, so audio longer than video freezes the final frame for the tail.
- `nvfp4_loader.py` — NVFP4→BF16 dequantizer for Sikaworld Gemma variant
- `dashboard.html` — GPU management dashboard SPA (served at /dashboard). Advanced LTX controls, Flux config, **Remote Pool** button grid (0..MAX), turbo toggle, GPU telemetry. v1.13.0 adds a **Live Workers** panel polling `GET /v1/system/workers` every ~2 s: row per worker (local + per-provider remote), shows `id`, `status` (busy/idle), and truncated `current_job` summary when busy.
- `config.py` — Paths, model mapping, device config, resolution tables, TF32 settings, env-var sidecar toggles (`LOAD_FLUX`, `LOAD_ACE`, `LOAD_JOYAI`, `LOAD_ERNIE`, `LTX_REMOTE_SIDECAR_URL`, `LTX_REMOTE_SIDECAR_MAX_WORKERS`, etc.)
- `scripts/register_outpaint_lora.sh` — idempotent cold-start: `hf download` + symlink + registry.json insert for the IC-LoRA outpaint LoRA (id `ic-lora-outpaint`).
- `scripts/construct_preference_pairs.py` (v1.18.0-rc3) — Phase C weekly cron: ETL across 4 signal sources (`user_retake` / `composition_kept` / `validator_pass` / `validator_fail`), version-scoped, idempotent via `idx_pp_unique_pair_source` UNIQUE. Watermark at `.preference_pairs_watermark`.
- `scripts/train_dpo_sft.py` (v1.18.0-rc3) — Phase C SFT-on-chosen training. Defaults to dry-run; `--execute` for the real run. LoRA-only PEFT, paged_adamw_32bit, gradient_checkpointing. Persists `training_runs` row with full reproducibility metadata.
- `scripts/ab_decision.py` (v1.18.0-rc3) — Phase C weekly cron: paired t-test on per-MV mean validator_score across `_ab_arm` cohorts; promotes/deprecates `training_runs` rows.
- `configs/sft_quality_lora.yaml` (v1.18.0-rc3) — Phase C training config (rank/alpha/target_modules/epochs/lr/seed/hyperparams).
- `run.sh` — entrypoint: sets `LD_LIBRARY_PATH`, `PYTORCH_CUDA_ALLOC_CONF`, env flags, then `uv run python server.py` (port 8090)

## Key commands
- Run: `bash run.sh`
- Test: `uv run pytest tests/ -v`
- Health: `curl http://localhost:8090/health`
- Register outpaint LoRA: `bash scripts/register_outpaint_lora.sh`

## GPU topology
- **cuda:0** → RTX PRO 6000 Blackwell 96GB — **LTX ↔ Flux** (2-tenant swap, mutually exclusive, auto-swapped on dispatch)
- **cuda:1** → RTX PRO 6000 Blackwell 96GB — **ACE xl-base+LM** (~18 GB) + **JoyAI** (~50 GB) OR **ERNIE-Image** (~33 GB), JoyAI and ERNIE swap (mutually exclusive), both coexist with ACE

Verified via `nvidia-smi -L`. No third GPU on this box — any earlier references to `cuda:2`/RTX 4000 are stale.

**DUAL_GPU_LTX mode**: `DUAL_GPU_LTX=1` env flag dedicates both GPUs to LTX video generation. cuda:1 runs an LTX sidecar (`ltx_sidecar_client.py` → `127.0.0.1:8093`) for 2 concurrent video workers. Flux, ACE, and JoyAI are disabled. Unlike turbo mode (runtime toggle), DUAL_GPU_LTX is a boot-time flag requiring restart.

**Why mutually exclusive on cuda:0**: LTX active is ~79 GB (60 GB transformer + 19 GB encoder hub + decoder activations) and Flux active is ~81 GB (60 GB transformer + ~14 GB CPU-offload forward-pass peak via `enable_model_cpu_offload`). Combined ~160 GB > 96 GB physical. They cannot coexist on one GPU during forward pass, so the server evicts the other before running.

**Auto-swap (two tenants on cuda:0)**: `server.py` exposes `_ensure_ltx_resident()` and `_ensure_flux_ready()` helpers, called inside `_inference_lock` by `_dispatch_job()` (v2 async) and every v1 sync handler (text_to_video, image_to_video, audio_to_video, retake, video_outpaint, text_to_image, image_to_image, image_edit) before `torch.cuda.set_device()`. LTX and Flux are mutually exclusive on cuda:0. LTX is **not** auto-reloaded after a Flux request — it stays evicted until the next video request. Long-stretch single-tenant workloads incur zero swap overhead; mixed workloads pay a per-direction-change cost (see swap section below).

### Turbo mode (v1.2, hardened v1.5)

Toggled via `POST /v1/system/turbo` (body: `{"enable": true/false}`). Temporarily claims cuda:1 for LTX, giving **2 concurrent LTX denoiser workers** (one per GPU). With the multi-provider remote-sidecar pool (v1.6 Modal, v1.9 RunPod; v1.13 Modal max raised to 10) turbo supports **up to 14 concurrent video workers total** (2 local + up to 10 Modal + up to 2 RunPod).

- **Entry** (`_enter_turbo_mode`, server.py:~1712): Flux unloaded → `_stop_cuda1_tenants()` runs `systemctl --user stop` on `ace-step`, `joyai-sidecar`, `ernie-image-sidecar`, `ltx-sidecar` → `_wait_cuda1_free(threshold_mib=2000, timeout_s=20)` drains cuda:1 via nvidia-smi polling → on timeout, aborts with `_restore_cuda1_tenants()` rollback → `systemctl start ltx-sidecar` → poll /health → /load → spawn second `worker_loop` → `_scale_remote_pool()` up to `_remote_worker_target`. Entry takes ~20 s.
- **Active**: Flux, ACE, JoyAI, ERNIE, music endpoints all return `503 turbo_mode_active`. Only video generation works.
- **Exit** (`_exit_turbo_mode`, server.py:~1795): scales remote pool to 0 → HTTP /unload local sidecar → `systemctl stop ltx-sidecar` → `_restore_cuda1_tenants()` re-starts configured `LOAD_*=1` services. Exit takes ~15 s.
- **Batch integration**: `_batch_worker` uses `asyncio.gather` to run items concurrently (2 under turbo).
- **Why systemctl-stop, not HTTP /unload**: v1.4 trusted HTTP `/unload` to free cuda:1. Silent unloads that succeeded on the wire while tensors stayed resident caused `CUDA OOM` on the subsequent ltx-sidecar `load`. `systemctl stop` + nvidia-smi drain verification is the hammer.

### Remote-sidecar pool (v1.6 → v1.9.0 multi-provider)

Optional remote workers augment turbo's local 2. v1.9.0: **multi-provider** — Modal and RunPod run side-by-side, each with independent target/active/max counts. Dict-keyed structure in `ltx_sidecar_client.py`: `ltx_remote_sidecars: dict[str, LtxSidecarClient]`.

- **Env (v1.9.0)**: `LTX_MODAL_SIDECAR_URL/TOKEN/MAX_WORKERS` and `LTX_RUNPOD_SIDECAR_URL/TOKEN/MAX_WORKERS`. Legacy `LTX_REMOTE_SIDECAR_*` (singular) still honored and aliased to the modal provider for backwards compat.
- **Client module** (`ltx_sidecar_client.py`): `ltx_remote_sidecars` dict keyed by provider name. Module-level `ltx_remote_sidecar` (singular) is preserved as an alias pointing at the modal entry so v1.6-v1.8 code keeps working.
- **Server state** (`server.py`): `_remote_worker_targets: dict[str, int]` (per-provider targets, persist across turbo toggles) + `_remote_worker_tasks: dict[str, list[asyncio.Task]]`. `_PROVIDERS = ("modal", "runpod")`.
- **Endpoints**:
  - `GET /v1/system/pool` → `{turbo_active, providers: {modal: {...}, runpod: {...}}, remote_*: <legacy aliases>}`.
  - `POST /v1/system/pool/remote-workers` accepts either `{"count": N}` (legacy — scales modal only) or `{"modal": N, "runpod": M}` (per-provider).
  - `POST /v1/system/pool/remote-workers/{provider}` with `{"count": N}` — RESTful per-provider.
- **Dispatch**: `_dispatch_job_turbo_remote(job, *, provider: str)` takes the provider kwarg. `_scale_remote_pool()` uses `functools.partial` to bind each worker task to its provider at spawn time. The LoRA-path rewrite for outpaint uses `config.LTX_PROVIDER_LORAS_MOUNT[provider]` → Modal `/mnt/nvme-1/huggingface/loras/`, RunPod `/runpod-volume/loras/`.
- **Media base64**: same for both providers — `_read_b64()` reads local `audio_path`/`image_path`/`video_path`/keyframe images and ships as `*_b64` kwargs. Neither Modal nor RunPod can see the host's `uploads/`.
- **Transport failure**: neither provider's transport errors auto-exit turbo — remotes are optional extra capacity. Local sidecar failure still triggers `_auto_exit_turbo_on_sidecar_failure`.
- **Dashboard**: two rows — `poolBtnGrid` (Modal) + `poolBtnGridRunpod` (RunPod). Each row renders N+1 buttons 0..MAX, polled every 5 s via `GET /v1/system/pool`. JS walks `data.providers.{modal,runpod}` with fallback to legacy flat fields for safety.
- **RunPod sidecar**: new repo at `/mnt/nvme-1/servers/ltx-sidecar-runpod/`. Load-Balancing Serverless on RTX PRO 6000 Blackwell, FastAPI inside each worker, `/ping` health probe. Dockerfile mirrors Modal's image recipe.

### ACE music sidecar (v1.2)

ACE Step (xl-base + LM) runs on cuda:1 at `127.0.0.1:8001` via a separate systemd service (`ace-step.service`). ~18 GB resident, coexists with JoyAI on cuda:1. `ace_client.py` proxies requests via httpx.

- Endpoints: `POST /v1/music` (sync), `POST /v2/music` (async job)
- Gated by `LOAD_ACE=1` env var. Returns `503` when disabled or during turbo mode.
- Music queue cap: `MAX_MUSIC_PENDING` (default 5), returns `429 music_queue_full` when exceeded.
- Phase for music jobs: `"generating"` (not `"denoising"`).

### JoyAI image-edit sidecar (v1.2, migrated from cuda:0)

Previously a third tenant on cuda:0 (v1.1.8). Now runs on cuda:1 at `127.0.0.1:8092`, coexisting with ACE. Mutually exclusive with ERNIE-Image (combined ~83 GB > 96 GB budget with ACE).

- Activation: `LOAD_JOYAI=1` env var in `.env`.
- Dispatch: `/v1/image-edit` and `/v2/image-edit` with `model="joyai-edit"` route to `joyai_client.edit()`.
- Service: `systemctl --user {start,stop,restart,status} joyai-sidecar`.
- Fallback: `503 joyai_disabled` / `503 sidecar_unreachable` — client should retry with `flux2-klein`.

### ERNIE-Image sidecar (v1.3)

baidu/ERNIE-Image (8B DiT text-to-image, Apache 2.0) runs on cuda:1 at `127.0.0.1:8094`. Swaps with JoyAI (mutually exclusive — combined ~83 GB exceeds 96 GB budget), both coexist with ACE (~18 GB).

- Activation: `LOAD_ERNIE=1` env var in `.env`.
- Dispatch: `/v1/text-to-image` and `/v2/text-to-image` with `model="ernie-image"` route to `ernie_client.generate()`.
- Service: `systemctl --user {start,stop,restart,status} ernie-image-sidecar`.
- VRAM: ~39 GB on disk, ~33 GB active (50 steps), ~18 GB turbo (8 steps).
- Latency: ~11 s at 8 turbo steps (1024x1024).
- Fallback: `503 ernie_disabled` / `503 sidecar_unreachable`.
- Env: `ERNIE_SIDECAR_URL` (default `http://127.0.0.1:8094`).

### madmom downbeat sidecar (v1.16.0)

CPU-only FastAPI service at `127.0.0.1:8095` for higher-accuracy beat + downbeat
detection (~+8% accent-cut accuracy on cross-genre pop vs librosa). BSD-licensed.
Lives in its own venv at `/mnt/nvme-1/servers/madmom-sidecar/` because madmom
needs `numpy<1.24` + `scipy<1.13` (binary-compat ceiling), incompatible with the
main taco-backend venv.

- Activation: `LOAD_MADMOM=1` (default truthy in `config.py`) and the sidecar
  service must be running.
- Dispatch: `POST /v1/music/analyze` with body `{"audio_uri": ..., "analyzer": "madmom"}`
  routes to `madmom_client.analyze()`. Default `analyzer="librosa"` keeps the
  v1.15.x in-process path byte-identical for existing callers.
- Service: `systemctl --user {start,stop,restart,status} madmom-sidecar`.
- Setup: `bash /mnt/nvme-1/servers/madmom-sidecar/setup.sh` — creates the venv,
  installs deps, registers + starts the unit, blocks on `/health`.
- Pre-warming: RNN/DBN processors load once at startup (~20-40 s on CPU); the
  first request after that returns in seconds. `TimeoutStartSec=120` in the
  unit accommodates this.
- Failure semantics: **no silent fallback** — a sidecar timeout / 5xx /
  unreachable returns `503` from `/v1/music/analyze` so callers know they
  didn't get the analyzer they asked for.
- Env: `MADMOM_SIDECAR_URL` (default `http://127.0.0.1:8095`).

### sapiens pose sidecar (v1.17.0-rc2)

Tier-2 of the validator pipeline — pose temporal-stability detection. FastAPI
service at `127.0.0.1:8096`, runs on cuda:1 alongside ACE; gets
`systemctl stop`'d on turbo-mode entry like the other cuda:1 tenants. In
v1.17.0-rc2 the sidecar ships in stub-mode (real inference lands rc2-side);
the client (`sapiens_client.py`) tolerates `{"stub": true}` payloads and
passes them through to `validator.composite()` which treats them as
"skipped, no penalty" (contributes 0.2·1.0 to the composite).

- Activation: `LOAD_SAPIENS=1` (default `0` for the rc2 ship — operator
  flips after diff review).
- Dispatch: in-process via `validator._run_tier2_sapiens(video_path)` →
  `sapiens_client.analyze_pose(video_path)`. Also runs as part of
  `POST /v2/video/analyze-motion` and `_on_job_complete` validator dispatch.
- Service: `systemctl --user {start,stop,restart,status} sapiens-sidecar`.
- Failure semantics: any sidecar error (ConnectError / 5xx / timeout)
  returns `tier2=None` and the composite drops the tier2 weight (graceful
  degrade — validator never blocks on tier-2 failure).
- Env: `SAPIENS_SIDECAR_URL` (default `http://127.0.0.1:8096`),
  `SAPIENS_TIMEOUT_S` (default 60).

## Validator pipeline (v1.17.0-rc2)

Three tiers + composite scoring + `validator_runs` cache + on-complete
dispatch. Lives in `validator.py`. Rooted in the v1.17.0-rc1 schema (the
`validator_runs` table + `validator_score` / `validator_payload_json` /
`validator_version` columns on `generations`).

**Tier 1 — RAFT** (in-process on cuda:0):

- `torchvision.models.optical_flow.raft_small` — chosen over `raft_large`
  because the validator runs on every completed video; the +1 GB transient
  VRAM and ~50 ms/frame difference matter at 28-job concurrency. Switch is
  a one-line change in `_run_tier1_raft` if accuracy regresses.
- Decode pipeline: PyAV at 24fps → downsample to 256-wide → per-pair flow
  magnitude → mean-per-pair time series.
- Outputs: `dynamic_degree` (top-5% percentile), `flow_windows` (4-window
  means), `motion_smoothness` (1 / (1 + variance(diff(flow)))).
- VRAM lifecycle: lazy-load on first call; evict + `empty_cache()` after
  each call so LTX reclaims cuda:0 cleanly. Weights download lazily from
  pytorch.org on first run (~22 MB, cached to `~/.cache/torch/hub/`).

**Tier 2 — Sapiens** (via sidecar at :8096):

- See [sapiens pose sidecar](#sapiens-pose-sidecar-v1170-rc2) above.
- Stub-tolerant: `{"stub": true}` → composite uses `0.2·1.0` in tier2 slot.
- Real-failure (ConnectError / 5xx / timeout) → `tier2=None` → composite
  drops the tier2 weight entirely.

**Tier 3 — Gemma judge** (via existing `chat_manager` + `CHAR_VISION_MODEL`):

- Mirrors `/v2/char/rank` (server.py:4307-4363). Samples 5 keyframes
  (first / quartile / middle / 3-quartile / last) via
  `history_store._extract_frames_as_pils`, base64-encodes as JPEG, builds
  multimodal request with `config.JUDGE_PROMPT_V1` as system prompt.
- Strict-JSON output: `{verdict, score, reasoning, retake_hint}` validated
  against `JudgeResponseV1` Pydantic model (mirrors `CharRankResponse`
  schema-validation pattern from v1.8.2 / SEC P1-5).
- On any failure (LLM unreachable, schema violation, parse error): returns
  fallback `{verdict: "warn", score: 0.5}` so composite still computes.

**Composite scoring**: `0.4·tier1_norm + 0.2·tier2 + 0.4·tier3`, where
`tier1_norm = min(dynamic_degree / 5.0, 1.0)` (empirically healthy clips
sit in 1-10 magnitude at 256-wide). Recommendation:

- `pass` iff composite ≥ 0.65
- `warn` iff 0.45 ≤ composite < 0.65
- `retake` iff composite < 0.45 OR `tier3.verdict == "retake"`
  (verdict-level override beats numeric composite — catches cases where
  RAFT and pose look fine but the LLM spots prompt-output mismatch).

**Caching** via `validator_runs(video_sha256, validator_version)` UNIQUE
index. SHA-256 streamed from disk (1 MiB chunks) so large clips don't
balloon RAM. Bumping `config.VALIDATOR_VERSION` forces re-runs.

**Endpoints**:

- `POST /v2/video/analyze-motion` — synchronous. Body
  `{video_uri, prompt?, shot_uuid?, tiers?, validator_version?}` →
  composite payload. Reuses `run_all_tiers()`. 404 on unknown URI; 401
  on missing bearer (when `API_KEYS` is set); 500 with
  `analyze_motion_failed` envelope on tier orchestration error.

**On-complete dispatch wiring** (`server.py`):

- `_decr_queue_on_complete(job)` — existing v1.8.2 callback — now also
  calls `_on_job_complete(job)` after the per-key counter decrement.
- `_on_job_complete(job)` eligibility:
  - `job.status == COMPLETED`
  - `job.type ∈ _VALIDATOR_VIDEO_TYPES = {TEXT_TO_VIDEO,
    IMAGE_TO_VIDEO, AUDIO_TO_VIDEO, RETAKE, VIDEO_OUTPAINT, VIDEO_HDR}`
  - `_is_training_opted_in(job.api_key)` — looks up
    `api_key_metadata.training_opt_in`. Default opt-out for unknown
    keys (single-tenant deploy seeded `.api_keys` with opt-in=1 on
    rc1's first migration; external bearers stay opt-out until
    explicitly INSERTed).
- `_dispatch_validator(job)` — fire-and-forget task. Resolves the
  result_uri to a path, calls `validator.run_all_tiers`, then UPDATEs
  the matching `generations` row with `validator_score`,
  `validator_payload_json`, `validator_version`. Best-effort: any
  failure logs WARN and returns; the queue worker has already moved on.

**Turbo coordination**: `sapiens-sidecar` added to `_stop_cuda1_tenants`
and `_restore_cuda1_tenants` (gated on `LOAD_SAPIENS`). Mirrors the
ACE / JoyAI / ERNIE pattern verbatim.

## Retrieval pipeline (v1.18.0-rc2)

Phase B retrieval backend. Three new endpoints + sqlite-vec virtual
table + per-key rate-limit middleware + `lora_applied_id`
end-to-end persistence + Phase B observability counters in
`/v1/system/metrics`. Built on top of the rc1 schema v4 keystone.

**`chat_manager.embed` / `embed_batch`** (chat_manager.py): proxies
`/v1/embeddings` to llama-swap. Returns float32-LE-packed bytes
(~14 KB per 3584-dim Gemma embedding) ready to insert directly into
sqlite-vec. `EMBEDDING_MODEL_VERSION` constant tags every row for
migration safety. Sorts by response `index` so server-side parallel
decode reordering doesn't shuffle results.

**sqlite-vec extension** (history_store.py): loaded BEFORE
`PRAGMA journal_mode=WAL` per sqlite-vec docs. Module-level
`SQLITE_VEC_AVAILABLE` + `SQLITE_VEC_LOAD_ERROR` propagate failure
gracefully — backend boot never crashes on missing extension.
Endpoints depending on the virtual table return 503 with a clear
message when the load failed. Virtual table:

```
CREATE VIRTUAL TABLE clip_embeddings USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[3584],
    embedding_model_version TEXT
);
```

**`POST /v2/embeddings/search`**: ranked semantic search over the
caller's own past clips. **Privacy gate**: every query is filtered
by `api_key_hash` of the caller — bearer A cannot surface bearer B's
rows. Ranking formula (50/35/10/5):

```
final = 0.50 * similarity
      + 0.35 * max(0, validator_score - 0.5) / 0.5
      + 0.10 * exp(-age_days / 30)
      + 0.05 * (1.0 if in_final_composition else 0.0)
```

Filter knobs: `k ∈ [1,20]`, `min_validator_score ∈ [0,1]`,
`validator_version_filter` (default = current
`config.VALIDATOR_VERSION`), optional `genre`.

**`POST /v2/embeddings/recommend-loras`**: aggregates LoRA
performance via similarity-then-group. Top-50 similar shots →
group by `lora_applied_id` → ranked by
`0.7·mean_validator_score + 0.3·max(0, expected_boost)` where
`expected_boost = lora_mean - no_lora_mean`. Returns `total_samples`
+ `no_lora_baseline_mean` for debugging. Empty list when no rows
have `lora_applied_id` populated.

**`POST /v2/system/bulk-revalidate`** (admin-gated): re-runs the
validator pipeline on rows where `validator_version != target`.
Default `dry_run=true` returns `{would_revalidate, sample_ids}`
without writing. `dry_run=false` queues fire-and-forget
`_dispatch_validator(synthetic_job)` per row, capped at `limit`.
Each dispatch writes `validator_score / validator_payload_json /
validator_version` via the existing UPDATE path.

**Per-key rate-limit middleware**: token-bucket, in-memory.
10 req/sec/key with burst 10. Applied ONLY to `/v2/embeddings/*` and
`/v2/system/bulk-revalidate` paths — existing endpoints unaffected.
Returns 429 + `Retry-After` header on exhaustion. Bypassed when
`config.API_KEYS` is empty (parity with `check_api_key`). Buckets
keyed by `sha256(api_key)` so raw bearers never land in heap dumps.

**`lora_applied_id` write fix**: every video v2 endpoint
(text-to-video / image-to-video / audio-to-video / retake /
video-outpaint / video-hdr) now captures `body.lora.id` +
`body.lora.strength` via `_lora_applied_pair(body)` and threads them
into `job.params` via the existing `_HISTORY_ONLY_PARAMS` strip path.
`worker_loop.history.save()` already accepted the kwargs since
v1.17.0-rc1 — wiring is now complete end-to-end. Required by
`recommend_loras` aggregation; was silently NULL on every prior rc.

**Observability** (`/v1/system/metrics` extension): new `embeddings`
block exposes `embeddings_search_total/success/failure/rate_limited`,
rolling p50/p95 latency over 1000-call window,
`embeddings_search_results_avg`, `recommend_loras_total`,
`bulk_revalidate_total`. Process-local; resets on restart.

**Backfill script** (`scripts/backfill_prompt_embeddings.py`):
idempotent, resumable. Queries opted-in `generations` rows whose `id`
is not already in `clip_embeddings`, batches at 64 inputs per llama-
swap call. Args: `--dry-run`, `--limit N`, `--rebuild`,
`--sleep-ms N`. Operator runs manually post-deploy; not part of
automated startup.

**Dependency**: `sqlite-vec>=0.1.9` added to `pyproject.toml`. PyPI
package ships the `.so` and a Python loader. No system packages
required.

**Operator tuning**: see `docs/operator-tuning.md` "Embeddings +
sqlite-vec" section for install verification, llama-swap
`/v1/embeddings` config, and troubleshooting.

### Dashboard and GPU telemetry (v1.2, advanced controls v1.3)

- `GET /dashboard` — static HTML SPA for GPU management (served from `dashboard.html`)
- `GET /v1/system/gpu` — nvidia-smi telemetry (2 s cache): per-GPU memory/temp/utilization, turbo state, tenant info
- **Advanced controls**: 14 tunable LTX generation parameters (sampler, fast/pro stage 1 steps, scheduler max/base shift, CFG/STG/rescale/modality scales, stage 2 steps + individual sigma sliders, eta controls, preset dropdowns, reset). All persisted to `.gen_config.json` via `GET/POST /v1/system/config`.
- **Flux config** (v1.4): 2 Flux-turbo tunables (`turbo_steps`, `turbo_guidance`) persisted to `.flux_config.json` via `GET/POST /v1/system/flux-config`.
- **Remote Pool** (v1.6): N+1 button grid 0..`LTX_REMOTE_SIDECAR_MAX_WORKERS` (default 10 since v1.13.0), active highlighted, status line reflects configured / target / active / turbo-pending.
- **Live Workers** (v1.13.0): per-worker status panel (local + modal + runpod) driven by `GET /v1/system/workers`. Each row shows `id`, busy/idle, and `current_job` when busy — useful for debugging which provider picked up a specific job.

## Phase C training pipeline (v1.18.0-rc3)

Infrastructure for SFT-on-chosen LoRA training on the captured preference corpus. **Infrastructure-only — no first training run.** First operator-driven invocation waits until the corpus crosses ~1000 pairs (~6-8 weeks at single-operator volume). User-locked: SFT-on-chosen for v1, Diffusion-DPO deferred to Phase C.1.

**Components**:

- `scripts/construct_preference_pairs.py` — weekly cron ETL across 4 sources, version-scoped, idempotent. Sources: `user_retake` (0.9), `composition_kept` (0.5), `validator_pass` (0.7), `validator_fail` (0.3). Each SELECT filters by `validator_version` AND `api_key_metadata.training_opt_in = 1` AND `since_watermark`. INSERTs use `INSERT OR IGNORE` against the rc1 v4 `idx_pp_unique_pair_source` UNIQUE index. Watermark file at `.preference_pairs_watermark`. CLI: `--since-watermark` (default), `--full-rebuild`, `--validator-version`, `--dry-run`, `--source`.
- `scripts/train_dpo_sft.py` — modeled on `LTX-2/packages/ltx-trainer/scripts/train.py`. **Defaults to dry-run** (defense-in-depth — `--execute` required to consume GPU; ~50-60 GPU-hours per run). LoRA-only via PEFT, paged_adamw_32bit, gradient_checkpointing, bf16. Persists `training_runs` row with FULL reproducibility metadata (`training_seed` + `hyperparams_json` + `dataset_snapshot_path` + `code_sha` + `validator_version_at_train` — all rc1 v4 columns). Marks consumed pairs as `used_in_training_run_id`. Registers in `lora_registry` as candidate (NOT auto-deployed).
- `scripts/ab_decision.py` — weekly cron, paired t-test on per-MV mean validator_score across `_ab_arm` cohorts (MV grouping = `composition_id`). Decision matrix: promote (≥+10% AND p<0.05), deprecate (≤-5% AND p<0.05), insufficient_samples (<30 MVs/arm), else no_action. `AB_AUTO_PROMOTE=1` (default) writes to `training_runs`; `=0` reports without writing. Manual Welch's t-test fallback when scipy unavailable.
- `configs/sft_quality_lora.yaml` — v0.0.1 config (rank=64, alpha=64, q/k/v/out_proj target modules, 3 epochs, lr=5e-4, paged_adamw_32bit). `validator_version: null` resolves to `config.VALIDATOR_VERSION` at runtime.

**Admin endpoint**:

- `POST /v1/system/lora/rollback` — body `{lora_id, reason}`. Verifies `lora_id == current MCP_PRODUCTION_LORA` (409 on mismatch — guards against deprecating non-production candidates), sets `training_runs.deprecated_at = now()`, finds the previous deployed-and-not-deprecated run, atomically rewrites `MCP_PRODUCTION_LORA=<previous>` in `.env`, returns audit shape. Applies on next process restart that re-reads `.env`; no in-process hot-swap. `_require_admin` gating mirrors `/v1/system/turbo` and `/v1/system/pause`.

**Privacy gate spine**: every source query filters by `api_key_metadata.training_opt_in = 1`. Opt-out bearers' clips never enter training. Verified by `test_construct_pairs_privacy_gate`.

**Validator-version scoping spine**: cross-version pairs cannot enter training. The schema-v4 UNIQUE index keys on `(chosen, rejected, signal_source)`; the `validator_version` column is checked in the SELECT before INSERT. Verified by `test_construct_pairs_version_scoping`.

**Operator runbook**: `docs/PHASE_C_TRAINING_RUNBOOK.md` — pre-flight sanity, weekly cron config, first training run, A/B monitoring, rollback CLI, troubleshooting (OOM / overfitting / mid-train crash).

## IC-LoRA video outpaint (v1.7.0)

New async endpoint `POST /v2/video-outpaint`. Expands a source video's canvas to a larger target resolution by letterboxing with pure-black padding, then uses `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint` (registered as LoRA id `ic-lora-outpaint`, strategy `ic_lora_outpaint`) to fill the black regions with temporally coherent content.

- **Request** (`VideoOutpaintRequest`, server.py:~706): `video_uri`, `prompt`, `target_resolution`, `position` (`OutpaintPosition` 9-value Literal: `center` + 4 edges + 4 corners), `duration`, `fps`, `seed`, `enhance_prompt`, `lora` (optional; defaults to `id="ic-lora-outpaint"` if omitted), `conditioning_strength` ∈ [0, 1], `skip_stage_2` escape hatch.
- **Handler** (server.py:~2685): if `lora=None`, substitutes `LoRAInput(id="ic-lora-outpaint", strength=1.0)`. Submits `JobType.VIDEO_OUTPAINT` through the standard `_submit_job` path.
- **Pipeline** (`_run_outpaint` in split_model_manager.py:~1825, `@_with_oom_recovery`): 2-stage distilled, patterned on `_run_t2v` fast branch:
  1. Stage 1 at half target res with the outpaint LoRA fused into the distilled transformer. Letterboxed source is VAE-encoded via `_build_outpaint_reference_latent` and appended to `stage_1_cond` as `VideoConditionByReferenceLatent(latent=ref_latent, downscale_factor=ref_scale, strength=user_lora[1])`. If `conditioning_strength < 1.0`, wrapped in `ConditioningItemAttentionStrengthWrapper(..., attention_mask=conditioning_strength)`.
  2. Stage 2 (if not skipped) upsamples 2x and refines at full target res. **LoRA stays fused across both stages** (accepted deviation from upstream `ltx_pipelines.ic_lora.ICLoraPipeline`, which drops LoRA for stage 2; reloading mid-request would cost ~30 s of fusion work).
- **Helpers** (split_model_manager.py module-level, ~:180): `_read_lora_reference_downscale_factor(lora_path)` reads `reference_downscale_factor` from safetensors metadata (default 1); `_build_outpaint_reference_latent(...)` scales source proportionally to fit, pads remainder with **-1 in normalized pixel space** (= RGB 0,0,0 after VAE decode = the LoRA's training black sentinel). Temporal dim padded with black frames if source is shorter than `num_frames`.
- **Output**: silent MP4 (no audio). Source audio passthrough deferred to v1.7.x.
- **LoRA cache key**: same as every other LTX flow — `(state_name="distilled", user_lora_tuple)`. Fusion is permanent; strength changes require full transformer reload.
- **Turbo parity**: `JobType.VIDEO_OUTPAINT` is in `_VIDEO_JOB_TYPES` (server.py:111), so turbo workers handle it. Local cuda:1 sidecar and Modal both work. `ltx_sidecar_client.py::generate()` carries `position` / `conditioning_strength` / `skip_stage_2` kwargs.
- **Modal staging**: `scripts/register_outpaint_lora.sh` installs the LoRA locally; `modal_app.py::download_weights` stages the same LoRA into the Modal HF volume at `/mnt/nvme-1/huggingface/loras/`.

## IC-LoRA video HDR (v1.14.0)

New async endpoint `POST /v2/video-hdr`. Promotes an LDR source clip to expanded dynamic range using `Lightricks/LTX-2.3-22b-IC-LoRA-HDR` (registered as LoRA id `ic-lora-hdr`, strategy `ic_lora_hdr`). **Architecturally piggybacks on the outpaint pipeline** with `target == source` and `position="center"` — no canvas expansion, no letterboxing of significance. The `_run_outpaint` codepath is reused **unchanged**; HDR is purely additive scaffolding.

- **Request** (`VideoHdrRequest`, server.py:~894): `video_uri`, `prompt`, `duration`, `fps`, `seed`, `enhance_prompt`, `lora` (optional; defaults to `id="ic-lora-hdr"`), `conditioning_strength` ∈ [0, 1], `skip_stage_2`. **No** `target_resolution` or `position` — derived server-side.
- **Handler** (server.py:~3425): if `lora=None`, substitutes `LoRAInput(id="ic-lora-hdr", strength=1.0)`. Probes source video via `history_store._probe_video_dims` (PyAV one-pass), snaps `(src_w, src_h)` to nearest /64 multiple via `_snap64`, builds the same params dict the outpaint endpoint produces but with `target_width=snap_w, target_height=snap_h, position="center"`, then submits `JobType.VIDEO_HDR` through `_submit_job`.
- **Dispatch** (server.py `_dispatch_job` case `VIDEO_HDR`, `_dispatch_job_turbo`, `_dispatch_job_turbo_remote`): all three call `manager.generate_outpaint(...)` — the same method outpaint uses. The only difference vs outpaint dispatch is which LoRA path is in `params["lora_path"]`.
- **Sidecars**: local cuda:1 (`/mnt/nvme-1/servers/ltx-sidecar/sidecar.py`), Modal (`modal_app.py`), and RunPod (`runpod_app.py`) each have a new `case "video-hdr":` branch that mirrors `case "video-outpaint":` and calls `_manager.generate_outpaint(...)`. The Pydantic `GenerateRequest.job_type: Literal[...]` was widened on all three to include `"video-hdr"`.
- **LoRA staging**: `scripts/register_hdr_lora.sh` installs locally; `ltx-sidecar-modal/modal_app.py::download_weights` and `ltx-sidecar-runpod/download_weights.py` stage the HDR LoRA into each provider's volume next to the outpaint LoRA. Symlinks keep `ic-lora-hdr.safetensors` resolvable by the registry id.
- **/64 snap rationale**: LTX requires width/height divisible by 64. Real-world source clips often aren't (e.g. 1920×1080 → 1920 OK, 1080 not). `_snap64(x) = max(64, ((x+32)//64)*64)` rounds to nearest /64 multiple. Diff vs source is ≤32 px per axis — not visually significant for HDR transform.
- **Stage-2 LoRA fusion deviation persists** — same as outpaint, our `_run_outpaint` keeps the IC-LoRA fused through stage 2. Upstream `ICLoraPipeline` runs stage 2 LoRA-free (per `packages/ltx-pipelines/CLAUDE.md`). The HDR LoRA was likely trained against upstream behavior — `skip_stage_2: true` produces the closest-to-upstream output (stage-1-only at half-res).
- **Output**: silent MP4 (no audio).
- **Turbo parity**: `JobType.VIDEO_HDR` ∈ `_VIDEO_JOB_TYPES`, so turbo + remote-pool dispatch automatically work once sidecars are redeployed.

## Flux pipeline details

### Model loading (flux_manager.py) — full bf16, no quantization
1. Build base pipeline in bf16:
   - Dev: `Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, local_files_only=True)` — weights land on CPU by default
   - Klein: `Flux2Transformer2DModel.from_single_file(...)` then `Flux2KleinKVPipeline.from_pretrained(transformer=..., torch_dtype=torch.bfloat16, local_files_only=True)`
2. **If a user LoRA is requested**: `pipe.load_lora_weights(path, adapter_name="user_lora")` — adapter mode, NO fusion. Strength is applied at inference time.
3. Device placement:
   - Dev: `pipe.enable_model_cpu_offload(device="cuda:0")` — text encoder (~45 GB), transformer (~60 GB), and VAE page CPU↔GPU on demand. Dev can NOT be fully resident in bf16 (105.9 GB > 96 GB).
   - Klein: `pipe.to("cuda:0")` — full bf16 resident (~32 GB total, fits comfortably).

**Why no FP8.** FP8 layerwise casting was removed in v1.1.1 after diagnosing screendoor / grid artifacts traced to the `fuse_lora → enable_layerwise_casting(float8_e4m3fn)` interaction. PEFT's input autocast hook forces compute back into FP8, silently defeating `compute_dtype=bfloat16`, and the shifted fused weights sit on non-standard FP8 grid points creating structured dithering (diffusers PR #10685, Flux issue #406). ComfyUI force-casts to bf16 on high-VRAM hardware for the same reason (ComfyUI issue #10087).

**Why `enable_model_cpu_offload` instead of `sequential_cpu_offload`.** Model-level paging moves whole components between CPU and GPU at pipeline call boundaries. Sequential offload pages at the module level and is ~10x slower per step.

**Interaction with single-GPU swap.** Because Flux Dev uses `enable_model_cpu_offload`, its resident GPU footprint between requests is near-zero — the pipeline object exists in Python but weights live on pinned CPU memory. A LTX transformer can remain loaded alongside an idle Flux pipeline in Python, but **not** during a forward pass. `_ensure_flux_ready()` evicts LTX only when Flux is about to run.

### Turbo (Flux image)
- `turbo: bool` field on `TextToImageRequest` / `ImageToImageRequest`
- When `turbo=true`: server overrides to `_flux_config["turbo_steps"]` (default 8), `turbo_guidance` (default 2.5), applies `FLUX_TURBO_SIGMAS` custom sigma schedule
- When `turbo=false`: uses client-provided steps/guidance and default scheduler sigmas
- Turbo and LoRA are **fully composable** — attach the `flux2-turbo` folder-drop LoRA AND set `turbo: true` for both adapter weights and the 8-step sigma schedule.
- `_flux_config` (v1.4) persisted to `.flux_config.json`; tunable via dashboard or `POST /v1/system/flux-config`.

### Klein model
- Loaded via `from_single_file()` (`Flux2KleinKVPipeline`), full bf16 resident
- Klein has its own 4-step distillation; `flux2-turbo` LoRA is Dev-only (client-side `model_compat` filter)
- Klein ignores `guidance_scale` entirely (distilled, no CFG) — the server strips it

## Precision settings (config.py)
- `torch.backends.cuda.matmul.allow_tf32 = False` — full float32 precision for VAE decode
- `torch.backends.cudnn.allow_tf32 = False` — full float32 for VAE convolutions
- TF32 was previously enabled but degraded VAE output quality on Blackwell GPUs (VAE uses `force_upcast=True` expecting real float32)
- `bf16_reduced_precision_reduction`: left at PyTorch default (`True`). LTX-2 was trained with default bf16 accumulation — forcing float32 accumulation creates a training/inference mismatch that compounds across transformer layers × denoising steps.

## API contract
- **`docs/API.md` is the canonical, client-facing, LLM-optimized API spec.** Any commit that adds, removes, or changes an endpoint (URL, method, request/response shape, status codes, auth requirements) MUST update `docs/API.md` in the **same commit**. Do not split "code change" from "doc change" across commits — the doc is the contract the other-side developer reads, and drift breaks them silently.
- When touching `server.py` endpoints, re-read `docs/API.md` before writing code so you know what the external shape is supposed to be, then update both sides together.
- CLAUDE.md (this file) and AGENTS.md describe **how to work in the code**. API.md describes **how clients talk to the service**. Don't duplicate substantive detail — link.

## Conventions
- All generation runs under `@torch.inference_mode()`
- All LTX `_run_*` methods are decorated with `@_with_oom_recovery` — on CUDA OOM the wrapper evicts the transformer + `cleanup_memory()` then re-raises (mirrors Flux pattern)
- Flux output: WEBP quality 95
- LTX output: raw MP4 bytes with `Content-Type: video/mp4`
- LTX: evict transformer before VAE decode (reclaims ~22GB), don't reload after — next request handles its own state
- LTX swap: on dispatch, `_ensure_ltx_resident()` / `_ensure_flux_ready()` (server.py) auto-evicts the other tenant on cuda:0 before each request. Never assume either manager is loaded — call the helper inside `_inference_lock`.
- LTX LoRA: fusion is permanent (no unfuse), different strengths require full transformer reload. Cache key `(state_name, user_lora_tuple)`.
- Flux LoRA: adapter mode (NOT fused) — strength is applied at inference time via `pipe.set_adapters([...], [strength])`. Cache key `(model_name, lora_path)` — strength is NOT in the key, so strength changes are free. Only model or LoRA file changes trigger reload.
- Frame count must be 8k+1; resolution multiples of 64
- Port 8090, auth via `.api_keys` file (disabled when empty)
- CFG++ sampler is default for all LTX video generation (togglable via `/v1/system/config` or `/v1/system/sampler`)
- Generation config persisted to `.gen_config.json` — survives restarts, editable via dashboard or API
- VAE decode uses `TilingConfig.default()` (upstream cosine tiling from ltx-core)
- All transformer calls wrapped in `BatchSplitAdapter(max_batch_size=1)` for correct multi-pass batching
- **Chain conditioning mode selection** (v1.12): when `AudioToVideoRequest.segment_uri` / `ImageToVideoRequest.segment_uri` is set, the request routes through `_build_segment_conditioning_latent` + single multi-frame `VideoConditionByLatentIndex` (9 consecutive target pixel frames hard-pinned). When unset, `_image_conds_for_keyframes` falls through to `combined_image_conditionings` — classical v1.11.5 LTX semantic (frame 0 hard-pin + frames 1+ soft-guide). Pydantic 3-way mutex across `{image_uri, keyframes, segment_uri}` — at most one per request, enforced by `@model_validator(mode="after")` on both request models.
- Cancellation: `DELETE /v2/jobs/{id}` raises `GenerationCancelledError` from `ProgressDenoiser.__call__` when `job.status == CANCELLED`, unwinding the sigma loop naturally (v1.4)
- MP4 encode tmpfile lives on `/dev/shm` via `config.MP4_TMPDIR` (tmpfs; fallback `/tmp`)
- Default negative prompt embeddings are cached per encoder lifecycle (`DEFAULT_NEGATIVE_PROMPT`, nulled in `evict_all`)
- torch.compile available via `TORCH_COMPILE=1` env flag but default OFF (no benefit on Blackwell with cuDNN FA4)

## Flux LoRA (v1.1 — folder-drop discovery, adapter mode)
- Storage: `/mnt/nvme-1/servers/taco-backend/flux_loras/` (filesystem is source of truth, no registry.json)
- ID = slugified filename stem (`MyStyle.safetensors` → `mystyle`)
- Optional sidecar `.json` next to `.safetensors` for name/description/trigger_word/model_compat
- Endpoints: `GET /v1/flux-loras` (list), `POST /v1/flux-loras/rescan` (re-scan folder)
- Request field: `lora: {id, strength}` on `TextToImageRequest` / `ImageToImageRequest` / `ImageEditRequest`
- Reuses the existing `LoRAInput` pydantic model (same `{id, strength}` shape as LTX)
- No upload endpoint by design — files managed via `cp`/`rm`

LoRAs attach as **named adapters**, not fused. At inference time every generate method calls `_apply_lora_strength(lora_path, strength)` → either `pipe.set_adapters(["user_lora"], [strength])` or `pipe.disable_lora()`. Free O(ms) op.

**FluxManager cache key** = `(model_name, lora_path)`. Model change (Dev ↔ Klein) or LoRA file change → full reload (~30–60 s for Dev). Strength change → zero reload.

## GPU swap mode (2-tenant on cuda:0)

LTX and Flux target `cuda:0`. `config.py` sets `LTX_DEVICE = FLUX_DEVICE = "cuda:0"`. cuda:1 runs ACE + JoyAI (or ERNIE) concurrently (no swap needed).

**Auto-swap helpers** (`server.py`):
- `_ensure_ltx_resident()` — no-op if `ltx_manager.is_ready`, else calls `ltx_manager.load_all()` (cold load is 7–30 s depending on OS page cache)
- `_ensure_flux_ready()` — no-op if `not ltx_manager.is_ready`, else calls `ltx_manager.evict_all()` (~3 s)
- Both **must** be called while holding `_inference_lock`. Wired into `_dispatch_job()` (v2 async) and all v1 sync handlers.

**evict_all leak fix** (`split_model_manager.py::evict_all`, v1.1.4): prior to the fix, `DenoiserWorker` held strong refs to source model builders that kept ~22 GB of encoder hub pinned after eviction. Fix: explicitly null reference paths before dropping workers. Verified: cuda:0 drops from 66.9 GB → **683 MiB** after unload. (v1.3 refactored from `ModelLedger` → `SingleGPUModelBuilder` / `CachingModelFactory` but the eviction pattern is the same.)

**Half-load recovery** (v1.4): `SplitModelManager.reset()` nulls workers + encoder_ledger + neg-prompt cache, per-GPU sync + `empty_cache()`. `_load_all_impl` sets `_last_load_failed` on exception; `_ensure_ltx_resident` calls `reset()` before retry when the flag is up.

**Swap + system endpoints** (Bearer auth required):
- `POST /v1/ltx/unload`, `POST /v1/ltx/reload`
- `POST /v1/flux/unload`, `POST /v1/flux/reload`
- `POST /v1/system/pause`, `POST /v1/system/resume` (acquire `_inference_lock`)
- `POST /v1/system/turbo` — toggle turbo mode (see above)
- `GET /v1/system/pool`, `POST /v1/system/pool/remote-workers` — Modal pool control (v1.6)
- `GET /v1/system/workers` — live per-worker state (local + modal + runpod) (v1.13.0)
- `GET /v1/system/sampler`, `POST /v1/system/sampler` — alias for sampler/eta/stage2_sigmas subset
- `GET /v1/system/config`, `POST /v1/system/config`, `POST /v1/system/config/reset` — LTX generation parameters
- `GET /v1/system/flux-config`, `POST /v1/system/flux-config`, `POST /v1/system/flux-config/reset` — Flux-turbo tunables (v1.4)
- `GET /v1/system/gpu` — nvidia-smi telemetry
- `GET /dashboard` — dashboard SPA

**Latency**:
- Within-type (video→video, image→image same LoRA): unchanged, fast
- LTX→Flux: +3 s eviction + Flux forward pass
- Flux→LTX: +7–30 s cold LTX load + video generation
- Turbo entry: ~20 s (evict ACE+JoyAI+Flux, drain cuda:1, load LTX sidecar)
- Turbo exit: ~15 s (scale remote pool to 0, unload sidecar, restart cuda:1 tenants)
- LoRA strength changes (Flux): free runtime op

## Batch scheduler

- `POST /v2/batch` — submit 1-50 items per batch
- `GET /v2/batch/{batch_id}` — poll status + partial results
- `GET /v2/batch/{batch_id}/result/{index}` — download individual item result file
- `DELETE /v2/batch/{batch_id}` — cancel remaining items
- Items are sorted images-first (Klein before Dev) to minimize GPU swaps
- Under turbo, `_batch_worker` uses `asyncio.gather` to process in parallel (2 local + up to MAX remote)
- `MAX_BATCH_QUEUE_DEPTH` (default 30, v1.16.4+), `MAX_BATCH_ITEMS` (default 50)
- Supported item types: `text-to-image`, `image-to-image`, `image-edit`, `text-to-video`, `image-to-video`

## Keyframe symbolic indices
- `KeyframeInput.frame_index` accepts `int | "first" | "middle" | "last"`
- Negative integers supported: -1 = last frame, -12 = 12 frames before end
- Symbolic values resolved in `_resolve_keyframes(body, num_frames)` after num_frames computed
- "first"=0, "middle"=num_frames//2, "last"=num_frames-1
- Duplicate detection on resolved integer values; bounds check `frame_index >= num_frames → 422`
- Recommended strengths: first=1.0, middle=0.5, last=1.0

## Char mode — character consistency ranking
- `POST /v2/char/rank` — `rank_image_uri` + `generated_image_uri` + `prompt` → Gemma 4 31B on llama-swap as multimodal chat completion
- System prompt is `CHAR_RANKING_PROMPT` in server.py — strict JSON with face_match/eyes/proportions/overall_likeness (1-10) + structured `edits: {add, remove, modify}`
- Routing: `chat_manager.generate_chat_completion(..., model=config.CHAR_VISION_MODEL)` (default `gemma-4-31b-it`). Other chat endpoints default to `CHAT_MODEL` (`gemma-3-12b-nvfp4`).
- noodle-i Char tab runs client-side loop: generate → rank → apply edits → regenerate until score ≥ 9 or user hits Stop

## Generation history (history_store.py)
- SQLite DB at `history.db` — **WAL mode** (readers never block behind the single writer)
- Saves every completed v2 job with prompt, model, dimensions, result_uri, thumbnail
- API key hashed with SHA-256 (raw keys never stored)
- Thumbnails: 256px-wide JPEG at `thumbnails/`. Video thumbnails extract the first frame via PyAV.
- Endpoints: `GET /v2/history`, `GET /v2/history/{id}`, `GET /v2/history/{id}/image`, `GET /v2/history/{id}/thumbnail`
- 30-day retention; history manages result-file lifecycle
- `history.save()` runs in `asyncio.to_thread` task fire-and-forgotten from `worker_loop` — queue dequeues next job immediately without stalling on PyAV + SQLite (~300 ms)
- **Schema v2** (v1.4): four columns — `params_json`, `gen_config_json`, `seed`, `enhanced_prompt` — for full reproducibility. Online migration via `PRAGMA user_version`, idempotent, no backfill.
  - `params_json`: raw request body (Pydantic `body.model_dump(mode="json")`) — preserves `storage://` URIs, resolution enum, LoRA `id+strength`, keyframes symbolic indices. Music sanitizes paths back to URIs via `_sanitize_params_for_history`.
  - `gen_config_json`: LTX `_gen_config` snapshot at dispatch time OR `{turbo_steps, turbo_guidance}` for Flux-turbo. NULL for non-turbo Flux, ERNIE, JoyAI.
  - `enhanced_prompt`: LTX-rewritten prompt text when `enhance_prompt=true` (captured via `on_prompt_enhanced` callback). Always NULL for Flux/ERNIE/JoyAI/retake.

## v2 job observability (v1.1.6 / v1.1.7)

- **`Job.phase` field** — coarse post-denoise phase: `"denoising" | "decoding" | "encoding" | "saving" | None`. Denoising callbacks cap at **0.90**; the top 10% of progress is reserved for post-denoise phases emitted by `split_model_manager._run_*` and `flux_manager._generate/_img2img/_edit`.
- **`/v2/jobs/{id}`** status response exposes `phase` when processing.
- **`/v2/jobs/{id}/stream`** SSE endpoint — EventSource-compatible live status stream. Emits on `(status, progress, phase, error_code)` change, closes on terminal state, keepalive comment every 15 s. Accepts bearer header OR `?token=` query param (browsers). Replaces the 240-GET polling loop per video job with one long-lived connection.
- **`/v2/jobs/{id}/preview`** serves the on-disk thumbnail via zero-copy `FileResponse`. Fallback lazy extraction for jobs without api_key is offloaded via `asyncio.to_thread`.
- **`GET /v2/history/{id}`** — full record with parsed params + gen_config.
- **Timing logs** at every post-denoise phase boundary in `split_model_manager` (`vae_decode`, `video_decode+encode`) and `flux_manager` (`flux_webp_encode`), plus `history.save` and `encode_prompts` via `_timed`. Grep production logs: `journalctl -u taco-backend | grep -E "vae_decode|encode|history.save"`.

## Approved images (noodle-i → noodle-v pipeline)
- Manifest: `approved-images/manifest.json`
- Images stored in shared uploads dir (referenced by `storage://{uuid}` URIs)
- Endpoints: `POST /v1/approved-images`, `GET /v1/approved-images`, `GET /v1/approved-images/{id}/file`
- Per-API-key scoped (key hash in manifest entries)
- noodle-i "To Video" button uploads image then POSTs metadata
- noodle-v polls the GET endpoint to display approved feed

## Generation config (v1.3)

All LTX generation parameters stored in `.gen_config.json` (project root). Changes take effect on the next generation request — no restart.

### Parameters (14 tunable via dashboard)
- `sampler`: `"cfg_pp"` (default) or `"euler"` — CFG++ uses alpha=(1-sigma) rescaling for improved motion quality
- `fast_stage1_steps`: 8 (default), range 4-20
- `pro_stage1_steps`: 30 (default), range 10-50
- `scheduler_max_shift`: 2.05, `scheduler_base_shift`: 0.95
- `cfg_scale`: 3.0, `stg_scale`: 1.0, `rescale_scale`: 0.7, `modality_scale`: 3.0
- `stg_blocks`: [28]
- `stage2_sigmas`: [0.85, 0.725, 0.4219, 0.0] — individual sigma sliders in dashboard
- `eta_stage1`: 1.0 (ancestral noise for distilled stage 1), `eta_default`: 0.0 (deterministic for guided/stage 2)

### Endpoints
- `GET /v1/system/config` — returns full config dict
- `POST /v1/system/config` — merge-update (partial body OK, unknown keys ignored)
- `POST /v1/system/config/reset` — restore all defaults
- `GET/POST /v1/system/sampler` — alias for sampler/eta/stage2_sigmas subset

## Critical patterns

- `cleanup_memory()` calls gc.collect + empty_cache + synchronize on **current device only** — don't blindly dedup the 10 open-coded `gc.collect()+synchronize(device)+empty_cache()` triples in `split_model_manager.py`; several need explicit per-device sync for multi-GPU paths (`evict_transformer`, `evict_all`, `reset`, DUAL_GPU_LTX inter-stage). Load-bearing comments mark the 8 that must stay.
- `detect_params(checkpoint)` opens safetensors metadata — cache the result, don't call per-request
- `encode_prompts()` with CachingModelFactory keeps text encoder loaded — the internal `del` only drops local ref
- Retake uses `MultiModalGuider(...)` directly, NOT `create_multimodal_guider_factory().build()` — factory has no `.build()` method
- Audio latent must be trimmed/padded to `AudioLatentShape.from_video_pixel_shape(output_shape).frames`
- A2V uses `GuidedDenoiser` (static) for stage 1, frozen audio with `noise_scale=0.0`
- **IC-LoRA outpaint**: uses `VideoConditionByReferenceLatent` + `ConditioningItemAttentionStrengthWrapper` from `ltx_core.conditioning`. Letterbox uses **-1 fill value** because the VAE expects `[-1, 1]` normalized space — `-1` decodes to RGB 0,0,0, matching the LoRA's training black sentinel. `reference_downscale_factor` comes from the LoRA safetensors metadata via `_read_lora_reference_downscale_factor` (default 1).
- **Outpaint LoRA stays fused through stage 2** — accepted deviation from upstream `ICLoraPipeline`. Reloading would cost ~30 s.
- **Remote dispatch media**: `_dispatch_job_turbo_remote` must base64-encode local files (`audio_path` → `audio_b64`, `segment_path` → `segment_b64`, etc.); Modal and RunPod have no view of the local `uploads/` dir. Base64 expands 4/3; payloads up to ~135 MB for retake video. Segment MP4s are small (~500 KB-1.5 MB for 9 H.264 frames — more efficient than 9 lossless PNGs). Remote sidecars stage `segment_b64` to `/tmp/<uuid>.mp4` before calling the manager.
- **Turbo entry aborts**: if `_wait_cuda1_free` times out, `_restore_cuda1_tenants()` is called before raising so we don't leave cuda:1 empty of services.

## Text encoder variants
- `GEMMA_VARIANT=default` — Google Gemma 3 12B PT (standard, BF16)
- `GEMMA_VARIANT=sikaworld` — Sikaworld abliterated FP4 (uncensored, NVFP4 quantized)
- `GEMMA_VARIANT=gemma-3-12b-it-nvfp4` — IT-tuned NVFP4 (instruction-tuned for prompt enhancement; required if `enhance_prompt=true` should produce useful rewrites). v1.16.0+. The PT (default) variant produces literal continuations of the seed prompt rather than rewritten prompts; the IT variant follows instructions cleanly.
- Set via `.env` or environment variable, requires server restart
- Sikaworld path: `/mnt/nvme-1/huggingface/gemma-3-12b-sikaworld/`
- IT-NVFP4 path: `/mnt/nvme-1/huggingface/hub/models--NeoChen1024--gemma-3-12b-it-NVFP4/snapshots/<commit_sha>` — staged via `hf download NeoChen1024/gemma-3-12b-it-NVFP4`. `config.py` resolves the snapshot via glob; pin to a concrete sha after the download settles.

## Dependencies
- **PyTorch 2.11.0+cu130** — FlexAttention/FA4 on Blackwell sm_120, SDPA auto-selects cuDNN FlashAttention
- **diffusers 0.38.0.dev0** (git main) — required for Flux2KleinKVPipeline
- **ltx-core 1.1.1** (editable install from `/mnt/nvme-1/repos/LTX-2/packages/ltx-core`) — vocoder fp32 fix, cosine tiling, layer streaming, BatchSplitAdapter, IC-LoRA conditioning primitives
- **ltx-pipelines 1.1.1** (editable install from `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines`) — SimpleDenoiser, GuidedDenoiser, FactoryGuidedDenoiser, sampler signatures
- LTX-2 repo: `/mnt/nvme-1/repos/LTX-2` (editable install, `torch~=2.7` pin is PEP 440 compatible with 2.11)
- Checkpoints: `/mnt/nvme-1/huggingface/ltx-2.3-checkpoints/` — v1.1 distilled models + spatial upscaler
- cuDNN >=9.20 (fixes conv3d memory bug) — currently 9.20.0.48
- cuBLAS >=13.2 (BF16/FP8 Blackwell speedup) — currently 13.3.0.5
- nvidia packages revert on `uv sync` — use `--no-sync` for runtime, manual pip for upgrades
- peft (LoRA loading via diffusers)
- comfy-kitchen (NVFP4 dequantization of Sikaworld text encoder)
- IC-LoRA outpaint weights: `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint` (Apache 2.0, ~1.3 GB). Install via `scripts/register_outpaint_lora.sh`.
