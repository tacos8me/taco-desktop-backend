# Glossary

Canonical vocabulary for taco-backend v1.18.0-rc3. Each entry is one
sentence with a link to the doc that owns the term in detail. When a
term lives in CLAUDE.md (the in-repo high-context map), that link is
authoritative.

## Index

- [_dispatch_validator](#_dispatch_validator)
- [_migrate ladder](#_migrate-ladder)
- [_on_job_complete](#_on_job_complete)
- [_stop_cuda1_tenants](#_stop_cuda1_tenants)
- [A/B arm](#ab-arm)
- [AB_AUTO_PROMOTE](#ab_auto_promote)
- [AB_TEST_ACTIVE](#ab_test_active)
- [additive migration](#additive-migration)
- [anchor mode (`image_uri`)](#anchor-mode-image_uri)
- [api_key_hash](#api_key_hash)
- [ARCHITECTURE.md](#architecturemd)
- [AUP override (Sapiens)](#aup-override-sapiens)
- [beat-aligned cut](#beat-aligned-cut)
- [BFF](#bff)
- [bulk_revalidate](#bulk_revalidate)
- [candidate LoRA](#candidate-lora)
- [clip_embeddings](#clip_embeddings)
- [code_sha](#code_sha)
- [composite score](#composite-score)
- [composition_id](#composition_id)
- [CURRENT_SCHEMA_VERSION](#current_schema_version)
- [cuda:0 / cuda:1 topology](#cuda0--cuda1-topology)
- [cut_music_video](#cut_music_video)
- [dataset_snapshot_path](#dataset_snapshot_path)
- [dead-letter column](#dead-letter-column)
- [deprecated_at](#deprecated_at)
- [embedding_model_version](#embedding_model_version)
- [find_similar_shots](#find_similar_shots)
- [flash insert](#flash-insert)
- [Gemma judge](#gemma-judge)
- [hyperparams_json](#hyperparams_json)
- [JUDGE_PROMPT_V1](#judge_prompt_v1)
- [llama-swap](#llama-swap)
- [LOAD_SAPIENS](#load_sapiens)
- [lora_applied_id](#lora_applied_id)
- [lora_applied_strength](#lora_applied_strength)
- [MCP_PRODUCTION_LORA](#mcp_production_lora)
- [Modal](#modal)
- [motion_intent](#motion_intent)
- [parent_clip_id](#parent_clip_id)
- [plan_shot_list](#plan_shot_list)
- [preference_pairs](#preference_pairs)
- [privacy gate](#privacy-gate)
- [production LoRA](#production-lora)
- [prompt_embedding (BLOB)](#prompt_embedding-blob)
- [RAFT](#raft)
- [ranking formula 50/35/10/5](#ranking-formula-5035105)
- [recommend_loras](#recommend_loras)
- [recommendation (pass / warn / retake)](#recommendation-pass--warn--retake)
- [remote pool](#remote-pool)
- [RunPod](#runpod)
- [Sapiens](#sapiens)
- [SAPIENS_SIDECAR_URL](#sapiens_sidecar_url)
- [schema v3](#schema-v3)
- [schema v4](#schema-v4)
- [segment chain (`segment_uri`)](#segment-chain-segment_uri)
- [shot_config_key](#shot_config_key)
- [shot_uuid](#shot_uuid)
- [sidecar](#sidecar)
- [signal_source](#signal_source)
- [signal_strength](#signal_strength)
- [similarity score](#similarity-score)
- [slideshow](#slideshow)
- [tier 1 / 2 / 3 (validator)](#tier-1--2--3-validator)
- [tier-0 / tier-1 (MCP)](#tier-0--tier-1-mcp)
- [training_opt_in](#training_opt_in)
- [training_runs](#training_runs)
- [training_seed](#training_seed)
- [transition.audioLeadFrames](#transitionaudioleadframes)
- [turbo mode](#turbo-mode)
- [used_in_training_run_id](#used_in_training_run_id)
- [user_signals](#user_signals)
- [VALIDATOR_VERSION](#validator_version)
- [validator_runs cache](#validator_runs-cache)
- [validator_version_at_train](#validator_version_at_train)
- [/v1/system/lora/rollback](#v1systemlora-rollback)

---

## Validator pipeline

### tier 1 / 2 / 3 (validator)

Three independently-failing scoring stages run on every completed video
job: tier 1 = RAFT optical flow, tier 2 = Sapiens pose, tier 3 = Gemma
multimodal judge. Link: [CAPTURE_VALIDATOR.md#23-three-tier-validator-taco-backendvalidatorpy](CAPTURE_VALIDATOR.md#22-three-tier-validator-taco-backend-validatorpy)

### RAFT

`torchvision.models.optical_flow.raft_small` — tier-1 motion analyzer
producing `dynamic_degree` / `flow_windows[4]` / `motion_smoothness`;
lazy-loaded on cuda:0 then evicted per call. Link: [CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy](CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy)

### Sapiens

Meta's pose / temporal-stability model running as the tier-2 sidecar at
`127.0.0.1:8096`; rc2 ships in stub mode (`{"stub": true}` → composite
treats it as 0.2·1.0). Link: [ARCHITECTURE.md#5-sidecars](ARCHITECTURE.md#5-sidecars)

### AUP override (Sapiens)

Operator's narrower-internal-use read of Meta Sapiens-2 AUP §1.b.vi.ii
(the `for biometric processing` clause), defended by synthetic-input-only
+ no-real-person-id + no-SaaS + no-external-bearers. Must be re-reviewed
before any scope change. Link: [PRIVACY_GOVERNANCE.md#104-sapiens-aup-read](PRIVACY_GOVERNANCE.md#104-sapiens-aup-read)

### Gemma judge

Tier-3 strict-JSON LLM judge over 5 sampled keyframes via existing
`chat_manager` + `CHAR_VISION_MODEL=gemma-4-31b-it`; output validated
against `JudgeResponseV1`. Link: [CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy](CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy)

### JUDGE_PROMPT_V1

Frozen system prompt for the Gemma judge; bumping the prompt text bumps
`VALIDATOR_VERSION` (cache invalidation tied 1:1). Link: [CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy](CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy)

### composite score

`0.4·tier1_norm + 0.2·tier2 + 0.4·tier3` ∈ [0, 1]; rc4+ requires at
least one of tier1/tier3 to have produced a real score, else returns
`{score: None, recommendation: "error"}`. Link: [CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy](CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy)

### recommendation (pass / warn / retake)

Three-valued composite verdict: `pass` ≥ 0.65, `warn` 0.45-0.65,
`retake` < 0.45 OR tier3 verdict-level retake override. Link: [CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy](CAPTURE_VALIDATOR.md#31-validator-pipeline-taco-backendvalidatorpy)

### VALIDATOR_VERSION

Frozen string in `config.py` (currently `1.17.0-rc5`); part of the
`validator_runs` UNIQUE key — bumping it invalidates every cached score
and forces re-runs. Link: [PRIVACY_GOVERNANCE.md#5-validator-drift--version-scoping](PRIVACY_GOVERNANCE.md#5-validator-drift--version-scoping)

### validator_runs cache

`(video_sha256, validator_version)` UNIQUE-indexed table that memoizes
composite payloads so re-validation is idempotent. Link: [CAPTURE_VALIDATOR.md#32-schema-v3-additions-history_storepy](CAPTURE_VALIDATOR.md#32-schema-v3-additions-history_storepy)

---

## Capture machine

### parent_clip_id

`generations` column populated by `POST /v2/retake` via
`find_id_by_result_uri()` — points the new clip at the clip it was
retaken from for DPO-pair reconstruction. Link: [CAPTURE_VALIDATOR.md#36-shot-lineage-forwarding-orchestratorpy--serverpy](CAPTURE_VALIDATOR.md#36-shot-lineage-forwarding-orchestratorpy--serverpy)

### shot_uuid

Per-clip stable identifier minted by the MCP `cut_music_video`
orchestrator and threaded through to the backend so `(shot_uuid,
shot_config_key)` cohorts survive retakes. Link: [CAPTURE_VALIDATOR.md#36-shot-lineage-forwarding-orchestratorpy--serverpy](CAPTURE_VALIDATOR.md#36-shot-lineage-forwarding-orchestratorpy--serverpy)

### shot_config_key

Hash of (prompt_template, lora_id, motion_intent, keyframes_layout) that
identifies "the same shot config" across retakes — used by Phase C
`composition_kept` pair construction. Link: [CAPTURE_VALIDATOR.md#36-shot-lineage-forwarding-orchestratorpy--serverpy](CAPTURE_VALIDATOR.md#36-shot-lineage-forwarding-orchestratorpy--serverpy)

### composition_id

Forward-looking column on `generations`; the canonical write of
"clip ∈ composition" lives in the v3 `composition_clips` inverted-index
table written by `POST /v2/compositions/{id}/export`. The column itself
is a documented dead-letter. Link: [PRIVACY_GOVERNANCE.md#25-lineage-stamps-rc1](PRIVACY_GOVERNANCE.md#25-lineage-stamps-rc1)

### lora_applied_id

`generations` column populated end-to-end as of v1.18.0-rc2 via
`_lora_applied_pair(body)` on every video v2 endpoint; required by
`recommend_loras` aggregation. Link: [RETRIEVAL_WORKFLOW.md#83-recommend_loras](RETRIEVAL_WORKFLOW.md#3-the-three-tools)

### lora_applied_strength

Companion column to `lora_applied_id`; preserves the per-clip strength
so retrieval can disambiguate same-LoRA different-strength outcomes. Link: [PRIVACY_GOVERNANCE.md#25-lineage-stamps-rc1](PRIVACY_GOVERNANCE.md#25-lineage-stamps-rc1)

### motion_intent

Optional ≤200-char string forwarded by MCP v0.7.0 from
`quality_validation.motion_intent_map[shot_idx]` through the
`POST /v2/video/analyze-motion` request into the tier-3 judge prompt
(rendered conditionally — line omitted when None to preserve rc4
byte-equivalence for non-MCP callers). Link: [CAPTURE_VALIDATOR.md#33-new-endpoint-post-v2videoanalyze-motion](CAPTURE_VALIDATOR.md#33-new-endpoint-post-v2videoanalyze-motion)

### _on_job_complete

Server-side callback chained from `_decr_queue_on_complete` that fires
`_dispatch_validator(job)` when the job is COMPLETED + a video type +
the bearer is `training_opt_in=1`. Link: [CAPTURE_VALIDATOR.md#34-passive-validator-dispatch-serverpy-_on_job_complete](CAPTURE_VALIDATOR.md#34-passive-validator-dispatch-serverpy-_on_job_complete)

### _dispatch_validator

Fire-and-forget asyncio task that runs the 3-tier pipeline and UPDATEs
`generations.{validator_score, validator_payload_json,
validator_version}`; failures log WARN, never raise. Link: [CAPTURE_VALIDATOR.md#34-passive-validator-dispatch-serverpy-_on_job_complete](CAPTURE_VALIDATOR.md#34-passive-validator-dispatch-serverpy-_on_job_complete)

---

## Phase C training

### preference_pairs

Schema-v3 table populated weekly by `scripts/construct_preference_pairs.py`
from 4 signal sources; `idx_pp_unique_pair_source` UNIQUE on
`(chosen, rejected, signal_source)` enables `INSERT OR IGNORE`
idempotence. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-2--pair-construction-weekly-cron](PHASE_C_TRAINING_RUNBOOK.md#section-2--pair-construction-weekly-cron)

### signal_strength

Per-row weight in `preference_pairs` reflecting how confident the
source is: 0.9 (user_retake), 0.7 (validator_pass), 0.5
(composition_kept), 0.3 (validator_fail synthetic). Link: [PHASE_C_TRAINING_RUNBOOK.md#section-2--pair-construction-weekly-cron](PHASE_C_TRAINING_RUNBOOK.md#section-2--pair-construction-weekly-cron)

### signal_source

One of `user_retake` / `composition_kept` / `validator_pass` /
`validator_fail` — names the construction rule that generated the
pair; part of the UNIQUE index so the same clip pair can legitimately
appear twice with different sources. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-2--pair-construction-weekly-cron](PHASE_C_TRAINING_RUNBOOK.md#section-2--pair-construction-weekly-cron)

### training_runs

Schema-v3 ledger table; `train_dpo_sft.py` writes one row per run with
the full reproducibility tuple
(`training_seed` + `hyperparams_json` + `dataset_snapshot_path` +
`code_sha` + `validator_version_at_train`). Link: [PRIVACY_GOVERNANCE.md#6-reproducibility-ledger-training_runs](PRIVACY_GOVERNANCE.md#6-reproducibility-ledger-training_runs)

### training_opt_in

`api_key_metadata` boolean — the privacy spine; **every** capture /
retrieval / training query filters by it. Default opt-out for unknown
keys; seeded opt-in=1 for every `.api_keys` line on first v2→v3
migration only. Link: [PRIVACY_GOVERNANCE.md#3-the-training_opt_in-flag--the-spine](PRIVACY_GOVERNANCE.md#3-the-training_opt_in-flag--the-spine)

### validator_version_at_train

`training_runs` column capturing the `VALIDATOR_VERSION` in effect when
a run was launched; lets the audit story prove "no cross-version data
mixed in this run". Link: [PRIVACY_GOVERNANCE.md#54-phase-c-filter](PRIVACY_GOVERNANCE.md#54-phase-c-filter)

### used_in_training_run_id

`preference_pairs` column written by `train_dpo_sft.py` after a
successful run so the same pair is never consumed by a later run. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-3--first-training-run-operator-gate](PHASE_C_TRAINING_RUNBOOK.md#section-3--first-training-run-operator-gate)

### A/B arm

Cohort label written into `composition_clips._ab_arm` by `cut_music_video`
when `AB_TEST_ACTIVE=1`; `scripts/ab_decision.py` paired-t-tests
per-MV mean validator scores across arms. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring](PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring)

### candidate LoRA

A `training_runs` row registered in `lora_registry` by
`train_dpo_sft.py` but **not** yet promoted to `MCP_PRODUCTION_LORA`;
the A/B harness compares it against production. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring](PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring)

### production LoRA

The LoRA pointed at by the `MCP_PRODUCTION_LORA` env var — what
`cut_music_video` uses by default. Promoting a candidate rewrites
`.env`; takes effect on next process restart, no hot-swap. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring](PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring)

### deprecated_at

`training_runs` column set by `POST /v1/system/lora/rollback`; the
"previous deployed-and-not-deprecated run" lookup uses this to find
the rollback target. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-5--rollback](PHASE_C_TRAINING_RUNBOOK.md#section-5--rollback)

### dataset_snapshot_path

Filesystem path to `training_runs/<run_id>/dataset.jsonl` written
before training starts so a run is reproducible byte-for-byte even if
the live `preference_pairs` table mutates after. Link: [PRIVACY_GOVERNANCE.md#62-reproducibility-from-a-training_runs-row](PRIVACY_GOVERNANCE.md#62-reproducibility-from-a-training_runs-row)

### code_sha

`git rev-parse HEAD` captured at run start so the trainer code can be
reconstructed even if `master` advances. Link: [PRIVACY_GOVERNANCE.md#62-reproducibility-from-a-training_runs-row](PRIVACY_GOVERNANCE.md#62-reproducibility-from-a-training_runs-row)

### hyperparams_json

Verbatim copy of `configs/sft_quality_lora.yaml` rendered as JSON into
the `training_runs` row — no out-of-band config drift. Link: [PRIVACY_GOVERNANCE.md#62-reproducibility-from-a-training_runs-row](PRIVACY_GOVERNANCE.md#62-reproducibility-from-a-training_runs-row)

### training_seed

Random seed pinned per training run for byte-equivalent re-runs. Link: [PRIVACY_GOVERNANCE.md#62-reproducibility-from-a-training_runs-row](PRIVACY_GOVERNANCE.md#62-reproducibility-from-a-training_runs-row)

---

## Phase B retrieval

### clip_embeddings

sqlite-vec virtual table `(id TEXT PK, embedding FLOAT[3584],
embedding_model_version TEXT)` loaded BEFORE
`PRAGMA journal_mode=WAL`; module-level `SQLITE_VEC_AVAILABLE` propagates
load failure as a 503. Link: [RETRIEVAL_WORKFLOW.md#3-prerequisites](RETRIEVAL_WORKFLOW.md#3-prerequisites)

### embedding_model_version

Tag column on every `clip_embeddings` row identifying which
llama-swap embedding model produced the vector — guards against silent
migration breakage. Link: [RETRIEVAL_WORKFLOW.md#3-prerequisites](RETRIEVAL_WORKFLOW.md#3-prerequisites)

### similarity score

Cosine similarity from sqlite-vec between the query embedding and a
clip's stored embedding; one of four ranking inputs. Link: [RETRIEVAL_WORKFLOW.md#find_similar_shots](RETRIEVAL_WORKFLOW.md#find_similar_shots)

### ranking formula 50/35/10/5

`final = 0.50·similarity + 0.35·max(0, validator_score-0.5)/0.5 +
0.10·exp(-age_days/30) + 0.05·in_final_composition`; the
`find_similar_shots` ranker. Link: [RETRIEVAL_WORKFLOW.md#find_similar_shots](RETRIEVAL_WORKFLOW.md#find_similar_shots)

### privacy gate

The invariant that every retrieval / training / signal query filters
by `api_key_hash` (and most also by `training_opt_in=1`); enforced at
the SQL layer, not in application code. Link: [PRIVACY_GOVERNANCE.md#3-the-training_opt_in-flag--the-spine](PRIVACY_GOVERNANCE.md#3-the-training_opt_in-flag--the-spine)

### find_similar_shots

MCP tier-1 tool wrapping `POST /v2/embeddings/search` — returns
top-k past shots ranked by the 50/35/10/5 formula, scoped to the
caller's bearer. Link: [RETRIEVAL_WORKFLOW.md#find_similar_shots](RETRIEVAL_WORKFLOW.md#find_similar_shots)

### recommend_loras

MCP tier-1 tool wrapping `POST /v2/embeddings/recommend-loras` — top-50
similar shots grouped by `lora_applied_id`, ranked by
`0.7·mean_validator + 0.3·max(0, lora_mean - no_lora_mean)`. Link: [RETRIEVAL_WORKFLOW.md#recommend_loras](RETRIEVAL_WORKFLOW.md#recommend_loras)

### bulk_revalidate

Admin-gated MCP tool wrapping `POST /v2/system/bulk-revalidate`;
re-runs the validator pipeline on rows where
`validator_version != target`. Defaults to `dry_run=true`. Link: [RETRIEVAL_WORKFLOW.md#bulk_revalidate-admin](RETRIEVAL_WORKFLOW.md#bulk_revalidate-admin)

---

## Schema

### CURRENT_SCHEMA_VERSION

`history_store.py` constant currently `4`; bumped via the additive
`_migrate()` ladder on first boot. v3 tests check against this constant
(not the literal) so v3-surface intent survives future bumps. Link: [PRIVACY_GOVERNANCE.md#5-validator-drift--version-scoping](PRIVACY_GOVERNANCE.md#5-validator-drift--version-scoping)

### schema v3

The v1.17.0-rc1 keystone: 11 nullable columns on `generations`, 3
indexes, 5 new tables (`composition_clips`, `validator_runs`,
`preference_pairs`, `training_runs`, `api_key_metadata`). Link: [CAPTURE_VALIDATOR.md#32-schema-v3-additions-history_storepy](CAPTURE_VALIDATOR.md#32-schema-v3-additions-history_storepy)

### schema v4

The v1.18.0-rc1 keystone: 2 cols on `generations` (`motion_intent`,
`embedding_model_version`), 3 indexes + 1 col on `preference_pairs`
(`validator_version` + `idx_pp_validator_version` +
`idx_pp_unique_pair_source` UNIQUE), 5 reproducibility cols on
`training_runs`. Link: [PRIVACY_GOVERNANCE.md#54-phase-c-filter](PRIVACY_GOVERNANCE.md#54-phase-c-filter)

### additive migration

Migration discipline: only ADD COLUMN / CREATE TABLE / CREATE INDEX —
never DROP or ALTER existing columns. Lets pre-vN code open a vN+1 DB
and silently ignore extra columns. Link: [DECISIONS.md#adr-010-schema-loose-checkpoints-additive-migrations-only](DECISIONS.md#adr-010-schema-loose-checkpoints-additive-migrations-only)

### _migrate ladder

`history_store._migrate()` chain: every gap from `PRAGMA user_version`
to `CURRENT_SCHEMA_VERSION` runs as a series of idempotent steps;
re-run safe. Link: [CAPTURE_VALIDATOR.md#32-schema-v3-additions-history_storepy](CAPTURE_VALIDATOR.md#32-schema-v3-additions-history_storepy)

### dead-letter column

A column kept nullable for backward compat but never written to from
some version forward — removal candidate. The canonical example is
`generations.prompt_embedding` BLOB (replaced by `clip_embeddings`
virtual table in rc2). Link: [ARCHITECTURE.md#4-data-layer](ARCHITECTURE.md#4-data-layer)

### prompt_embedding (BLOB)

Original schema-v3 BLOB column on `generations`; documented dead-letter
since v1.18.0-rc1 because at 14KB/row × 1M rows it exceeds SQLite's
practical column-storage envelope. Link: [ARCHITECTURE.md#4-data-layer](ARCHITECTURE.md#4-data-layer)

---

## MCP / orchestration

### tier-0 / tier-1 (MCP)

Capability split in noodlefinger-mcp: tier-0 = 6 docs / discovery tools
(no API key required, nothing leaves the box); tier-1 = 23 job-submitting
tools registered only when `NOODLEFINGER_API_KEY` is set. Link: [MCP.md#1-what-it-is](MCP.md#1-what-it-is)

### BFF

The noodlefinger-bff service — owns the `user_signals` table and
auth-mediated portal endpoints. Distinct from taco-backend. Link: [ARCHITECTURE.md#8-the-bff-layer](ARCHITECTURE.md#8-the-bff-layer)

### user_signals

Single BFF table `(ts, actor_email, signal_type, target_id,
metadata_json)` capturing every MCP→BFF tee event + portal user action;
training-data input + product-engagement metric. Link: [ARCHITECTURE.md#8-the-bff-layer](ARCHITECTURE.md#8-the-bff-layer)

### api_key_hash

`sha256(bearer)` — the raw bearer never lands in `history.db` or
`user_signals`; the hash is the per-tenant scoping key for every
privacy-gated query. Link: [PRIVACY_GOVERNANCE.md#12-future-multi-tenant-via-api_key_hash-scoping](PRIVACY_GOVERNANCE.md#12-future-multi-tenant-via-api_key_hash-scoping)

### anchor mode (`image_uri`)

Chain-conditioning mode where each follower clip pins its first frame
to a supplied `image_uri` — independent of the predecessor; in
`cut_music_video`, anchor-mode followers do **not** consume their
predecessor's `segment_uri`. Link: [MV_EDITING.md#51-cut_music_video---extended-shot_list-schema](MV_EDITING.md#51-cut_music_video---extended-shot_list-schema)

### segment chain (`segment_uri`)

v1.12 chain mode where the follower's first 9 latent frames are
hard-pinned to a `segment_uri` MP4 produced by
`POST /v2/video/extract-segment` of the predecessor's tail; default
flipped to `keyframes` in MCP v0.4.6. Link: [MV_EDITING.md#51-cut_music_video---extended-shot_list-schema](MV_EDITING.md#51-cut_music_video---extended-shot_list-schema)

### flash insert

A single-frame or sub-second insert clip in a music video shot list,
typically aligned to a snare or accent — stored as a synthetic clip
with `storage_uri` (no `historyId`). Link: [MV_EDITING.md#3-the-grammar---14-named-techniques](MV_EDITING.md#3-the-grammar---14-named-techniques)

### slideshow

A composition primitive: a stack of stills timed to beats, weaved into
the shot list as a single clip-like unit. Link: [MV_EDITING.md#24-the-composition-language---five-otio-minus-primitives](MV_EDITING.md#24-the-composition-language---five-otio-minus-primitives)

### transition.audioLeadFrames

Per-transition int (default 0) — J/L cut audio offset; positive value
brings next clip's audio in early over the outgoing video. Link: [MV_EDITING.md#51-cut_music_video---extended-shot_list-schema](MV_EDITING.md#51-cut_music_video---extended-shot_list-schema)

### beat-aligned cut

Cut placed deterministically on a beat or downbeat from
`POST /v1/music/analyze`; opt-in via MCP v0.5
`slice_strategy="beats"|"downbeats"`. Link: [MV_EDITING.md#4-the-algorithm---deterministic-cut-placement](MV_EDITING.md#4-the-algorithm---deterministic-cut-placement)

### cut_music_video

Tier-1 macro tool: music → N a2v clips (chained) → composition →
export, in one call; checkpointed and resume-safe. Link: [MCP.md#43-tier-1-macro-cut_music_video-end-to-end](MCP.md#43-tier-1-macro-cut_music_video-end-to-end)

### plan_shot_list

Tier-1 helper that turns `(audio_summary, prompt, genre,
num_beats_per_shot, sections)` into a structured shot list ready for
`cut_music_video`. Link: [MV_EDITING.md#53-plan_shot_listaudio_summary-prompt-genre-num_beats_per_shot8-sections](MV_EDITING.md#53-plan_shot_listaudio_summary-prompt-genre-num_beats_per_shot8-sections)

---

## Infrastructure

### turbo mode

Runtime toggle (`POST /v1/system/turbo`) that evicts the cuda:1
tenants and gives LTX a second worker on cuda:1; combined with the
remote pool it scales up to 14 concurrent video workers. Link: [ARCHITECTURE.md#3-gpu-topology](ARCHITECTURE.md#3-gpu-topology)

### remote pool

Optional Modal + RunPod remote LTX sidecars dispatched via
`_dispatch_job_turbo_remote`; per-provider targets controlled by
`POST /v1/system/pool/remote-workers/{provider}`. Link: [ARCHITECTURE.md#3-gpu-topology](ARCHITECTURE.md#3-gpu-topology)

### Modal

One of the two remote-pool providers; max 10 concurrent workers since
v1.13.0. Media is base64-inlined since Modal can't see local
`uploads/`. Link: [ARCHITECTURE.md#3-gpu-topology](ARCHITECTURE.md#3-gpu-topology)

### RunPod

Second remote-pool provider on Load-Balancing Serverless RTX PRO 6000
Blackwell; mirrors Modal's image recipe + base64 media protocol. Link: [ARCHITECTURE.md#3-gpu-topology](ARCHITECTURE.md#3-gpu-topology)

### sidecar

Out-of-process FastAPI service taco-backend proxies via httpx — `ltx`,
`ace`, `joyai`, `ernie`, `madmom`, `sapiens`, `ltx-modal`,
`ltx-runpod`. Each has its own systemd user unit. Link: [ARCHITECTURE.md#5-sidecars](ARCHITECTURE.md#5-sidecars)

### llama-swap

External chat-completions proxy (Gemma 3/4 family) that taco-backend
hits via `chat_manager`; also serves `/v1/embeddings` for Phase B's
3584-dim Gemma embeddings. Link: [ARCHITECTURE.md#5-sidecars](ARCHITECTURE.md#5-sidecars)

### cuda:0 / cuda:1 topology

Two RTX PRO 6000 Blackwell 96GB GPUs: cuda:0 = LTX↔Flux 2-tenant swap,
cuda:1 = ACE + JoyAI/ERNIE swap (+ sapiens stub at :8096). Link: [ARCHITECTURE.md#3-gpu-topology](ARCHITECTURE.md#3-gpu-topology)

### _stop_cuda1_tenants

Server helper that `systemctl --user stop`s every cuda:1 service
gated by its `LOAD_*` flag (ace-step, joyai-sidecar,
ernie-image-sidecar, sapiens-sidecar, ltx-sidecar) before turbo mode
claims the GPU. Mirrored by `_restore_cuda1_tenants`. Link: [ARCHITECTURE.md#3-gpu-topology](ARCHITECTURE.md#3-gpu-topology)

---

## Endpoints / admin env

### MCP_PRODUCTION_LORA

`.env` var pointing at the LoRA id `cut_music_video` should default to;
rewritten atomically by `POST /v1/system/lora/rollback`; takes effect
on next process restart. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-5--rollback](PHASE_C_TRAINING_RUNBOOK.md#section-5--rollback)

### AB_AUTO_PROMOTE

`.env` toggle (default `1`) that lets `scripts/ab_decision.py` actually
write `deployed_at` / `deprecated_at` to `training_runs`; set `0` for
report-only mode. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring](PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring)

### AB_TEST_ACTIVE

`.env` flag that activates per-MV arm randomization in
`cut_music_video`; when `0`, every shot uses the production LoRA. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring](PHASE_C_TRAINING_RUNBOOK.md#section-4--ab-monitoring)

### LOAD_SAPIENS

Boot-time env var (default `0`) gating both the tier-2 dispatch and
the turbo-mode `_stop_cuda1_tenants` entry for `sapiens-sidecar`. Link: [operator-tuning.md#enabling-the-sapiens-sidecar-tier-2](operator-tuning.md#enabling-the-sapiens-sidecar-tier-2)

### SAPIENS_SIDECAR_URL

Env var (default `http://127.0.0.1:8096`) — where `sapiens_client`
sends pose-analysis requests. Link: [operator-tuning.md#env-vars](operator-tuning.md#env-vars)

### /v1/system/lora/rollback

Admin-gated endpoint (body `{lora_id, reason}`) that verifies the id
matches current `MCP_PRODUCTION_LORA` (409 on mismatch), sets
`training_runs.deprecated_at`, and atomically rewrites `.env`. Applies
on next restart. Link: [PHASE_C_TRAINING_RUNBOOK.md#section-5--rollback](PHASE_C_TRAINING_RUNBOOK.md#section-5--rollback)

---

## Doc ownership map

| Topic | Authoritative doc |
|---|---|
| Validator pipeline internals | [CAPTURE_VALIDATOR.md](CAPTURE_VALIDATOR.md) |
| Validator roadmap | [CAPTURE_VALIDATOR_ROADMAP.md](CAPTURE_VALIDATOR_ROADMAP.md) |
| Phase B retrieval workflow | [RETRIEVAL_WORKFLOW.md](RETRIEVAL_WORKFLOW.md) |
| Phase C training runbook | [PHASE_C_TRAINING_RUNBOOK.md](PHASE_C_TRAINING_RUNBOOK.md) |
| Privacy + opt-in + audit | [PRIVACY_GOVERNANCE.md](PRIVACY_GOVERNANCE.md) |
| Architectural design rationale | [DECISIONS.md](DECISIONS.md) (ADRs) |
| Cross-cutting architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| MV editing grammar | [MV_EDITING.md](MV_EDITING.md) |
| MCP tools + auth model | [MCP.md](MCP.md) |
| Client-facing endpoint shapes | [API.md](API.md) |
| Day-2 operator tasks | [OPERATOR_QUICKSTART.md](OPERATOR_QUICKSTART.md) |
| Tunable knobs (rate limit, ffmpeg, sapiens) | [operator-tuning.md](operator-tuning.md) |
| Symptom → fix lookups | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| UI gap inventory | [UX_GAPS.md](UX_GAPS.md) |
