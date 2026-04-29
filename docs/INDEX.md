# taco-backend — Docs Index

**Version**: v1.18.0-rc3 (2026-04-29).

taco-backend is a FastAPI inference server fronting an LTX-2.3 video stack and Flux 2 image stack on two RTX PRO 6000 Blackwell GPUs, with a constellation of sidecars (ACE music, JoyAI / ERNIE image-edit, madmom downbeats, sapiens pose, plus a remote LTX worker pool on Modal + RunPod). It powers the noodlefinger product family (`noodle-i`, `noodle-v`, `noodle-mv`) via `api.noodlefinger.io`, and is the substrate for the capture+validator+retrieval+training flywheel that turns every shipped MV clip into labeled training data.

## What shipped recently

- **v1.18.0-rc3** (2026-04-29) — Phase C training infrastructure: weekly preference-pair ETL (`scripts/construct_preference_pairs.py`), SFT-on-chosen LoRA trainer (`scripts/train_dpo_sft.py`, dry-run by default), A/B decision cron (`scripts/ab_decision.py`), `POST /v1/system/lora/rollback` admin endpoint. Infrastructure-only; first real training run waits ~6-8 weeks for the corpus to cross ~1000 pairs. Test suite 230 → 245 green.
- **v1.18.0-rc2** (2026-04) — Phase B retrieval: sqlite-vec virtual table, `POST /v2/embeddings/search`, `POST /v2/embeddings/recommend-loras`, `POST /v2/system/bulk-revalidate`, per-key rate-limit middleware on `/v2/embeddings/*`, end-to-end `lora_applied_id` persistence on every video v2 endpoint, embeddings-block telemetry in `/v1/system/metrics`.
- **v1.18.0-rc1** (2026-04) — Schema v4 keystone migration (`history_store.CURRENT_SCHEMA_VERSION 3 → 4`): additive nullable columns on `generations` / `preference_pairs` / `training_runs` for Phase B + Phase C. Schema-only; no feature wiring. Idempotent re-run safe.

Full history → [CHANGELOG.md](../CHANGELOG.md). Per-version code highlights → [../CLAUDE.md](../CLAUDE.md).

## Phase status at a glance

| Phase | Status | Doc |
|---|---|---|
| A — capture + validator | LIVE (v1.17.0-rc1..rc5) | [CAPTURE_VALIDATOR.md](./CAPTURE_VALIDATOR.md) |
| B — retrieval | LIVE (v1.18.0-rc2) | [RETRIEVAL_WORKFLOW.md](./RETRIEVAL_WORKFLOW.md) |
| C — SFT training | INFRA SHIPPED (v1.18.0-rc3); first run pending corpus | [PHASE_C_TRAINING_RUNBOOK.md](./PHASE_C_TRAINING_RUNBOOK.md) |
| D — forward look | NOT STARTED | [CAPTURE_VALIDATOR_ROADMAP.md](./CAPTURE_VALIDATOR_ROADMAP.md) |

## I want to...

| Intent | Doc |
|---|---|
| spin up a fresh box from scratch | [OPERATOR_QUICKSTART.md](./OPERATOR_QUICKSTART.md) |
| ship a frontend / SDK client in 5 minutes | [QUICKSTART.md](./QUICKSTART.md) |
| understand the system as a whole | [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md) |
| understand how the parts fit together | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| debug a specific failure (429s, 503s, NULL validator scores...) | [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) |
| tune validator / retrieval / rate-limit / export-quality knobs | [operator-tuning.md](./operator-tuning.md) |
| look up a term | [GLOSSARY.md](./GLOSSARY.md) |
| integrate as an HTTP client (full endpoint contract) | [API.md](./API.md) |
| understand why we made decision X | [DECISIONS.md](./DECISIONS.md) |
| understand the privacy / multi-tenant story | [PRIVACY_GOVERNANCE.md](./PRIVACY_GOVERNANCE.md) |
| use Phase B retrieval (`find_similar_shots` / `recommend_loras`) | [RETRIEVAL_WORKFLOW.md](./RETRIEVAL_WORKFLOW.md) |
| find UI gaps for the build queue | [UX_GAPS.md](./UX_GAPS.md) |
| run the first SFT training cycle | [PHASE_C_TRAINING_RUNBOOK.md](./PHASE_C_TRAINING_RUNBOOK.md) |
| understand the capture+validator machine | [CAPTURE_VALIDATOR.md](./CAPTURE_VALIDATOR.md) |
| see what's next (Phase D forward-look) | [CAPTURE_VALIDATOR_ROADMAP.md](./CAPTURE_VALIDATOR_ROADMAP.md) |
| build a music video via MCP | [MCP.md](./MCP.md) + [MV_EDITING.md](./MV_EDITING.md) |
| change GPU topology / turbo / remote pool behavior | [gpu-architecture.md](./gpu-architecture.md) |
| add or list a model | [models.md](./models.md) |
| set environment variables | [configuration.md](./configuration.md) |
| build a gallery / review webapp | [gallery_integrate.md](./gallery_integrate.md) |
| wire `/v2/video-outpaint` into a frontend | [outpaint-frontend-guide.md](./outpaint-frontend-guide.md) |
| wire `/v2/video-hdr` into a frontend | [hdr-frontend-guide.md](./hdr-frontend-guide.md) |
| wire `/v2/retake` into a frontend | [retake-frontend-guide.md](./retake-frontend-guide.md) |
| wire v1.12 chain conditioning into a frontend | [handover-frontend-v1.10-chain.md](./handover-frontend-v1.10-chain.md) |
| work in the code (anchors, conventions, recent highlights) | [../CLAUDE.md](../CLAUDE.md) |

## Reading order by role

If you're new to the project, start at the top of your column and walk down. Most docs make sense in isolation; these are just the orderings that minimize backtracking.

| Operator (on-call, deploys, tunes) | Developer (reads + writes code) | Client (writes a UI / SDK) | Agent (LLM authoring + tools) |
|---|---|---|---|
| [OPERATOR_QUICKSTART.md](./OPERATOR_QUICKSTART.md) | [../CLAUDE.md](../CLAUDE.md) | [QUICKSTART.md](./QUICKSTART.md) | [MCP.md](./MCP.md) |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | [ARCHITECTURE.md](./ARCHITECTURE.md) | [API.md](./API.md) | [MV_EDITING.md](./MV_EDITING.md) |
| [operator-tuning.md](./operator-tuning.md) | [API.md](./API.md) | endpoint guides (outpaint / hdr / retake / chain) | [RETRIEVAL_WORKFLOW.md](./RETRIEVAL_WORKFLOW.md) |
| [PRIVACY_GOVERNANCE.md](./PRIVACY_GOVERNANCE.md) | [DECISIONS.md](./DECISIONS.md) | [gallery_integrate.md](./gallery_integrate.md) | [GLOSSARY.md](./GLOSSARY.md) |
| [PHASE_C_TRAINING_RUNBOOK.md](./PHASE_C_TRAINING_RUNBOOK.md) | [CAPTURE_VALIDATOR.md](./CAPTURE_VALIDATOR.md) | | |

## Doc-by-doc inventory

Alphabetized. Each entry: filename — one-sentence description — *audience*.

- [API.md](./API.md) — Canonical client-facing HTTP contract for every taco-backend endpoint; must update in the same commit as any endpoint change. *client / developer*
- [ARCHITECTURE.md](./ARCHITECTURE.md) — 10-minute topology overview: the capture→validate→retrieve→train flywheel and how each subsystem fits. *developer*
- [audit-2026-04-06-comfyui-comparison.md](./audit-2026-04-06-comfyui-comparison.md) — Historical 5-agent audit comparing our LTX/Flux/VAE pipelines to ComfyUI reference; preserved for archaeology. *developer*
- [batch-turbo-spec.md](./batch-turbo-spec.md) — Original spec for the batch scheduler + turbo-mode features (Phase 3); now-shipped, kept for design context. *developer*
- [CAPTURE_VALIDATOR.md](./CAPTURE_VALIDATOR.md) — Canonical reference for the capture+validator subsystem (RAFT + Sapiens + Gemma judge): flow, knobs, privacy, extension surface. *operator / developer*
- [CAPTURE_VALIDATOR_ROADMAP.md](./CAPTURE_VALIDATOR_ROADMAP.md) — Forward-look for the validator machine: triggers, blockers, priorities for what comes next (Phase D and beyond). *operator / developer*
- [configuration.md](./configuration.md) — All `.env` environment variables and what they toggle. *operator*
- [debug-v1.11.3-chain-conditioning.md](./debug-v1.11.3-chain-conditioning.md) — Debugging record for the seam-2+ subject-drift bug that traced through v1.10.0 → v1.11.5 chain conditioning; preserved for archaeology. *developer*
- [DECISIONS.md](./DECISIONS.md) — Append-only ADRs (context, decision, consequences, revisit-when) for taco-backend and the capture/validator/training pipeline. *developer*
- [frontend-api-changes.md](./frontend-api-changes.md) — Superseded — see [API.md](./API.md) and [QUICKSTART.md](./QUICKSTART.md) instead. Kept as version history. *client*
- [frontend-lora-integration.md](./frontend-lora-integration.md) — Superseded — see [API.md § LoRAs](./API.md). Kept as version history. *client*
- [gallery_integrate.md](./gallery_integrate.md) — One-stop reference for browsing, displaying, and curating outputs across every pipeline; for review / gallery webapps. *client*
- [gpu-architecture.md](./gpu-architecture.md) — Dual-GPU layout, LTX↔Flux swap, turbo mode, remote-sidecar pool, latency profile. *operator / developer*
- [handover-frontend-v1.10-chain.md](./handover-frontend-v1.10-chain.md) — Frontend handover for v1.12 chain conditioning (`extract-segment` + `segment_uri`); supersedes the v1.11.5 3-PNG-keyframes flow. *client*
- [hdr-frontend-guide.md](./hdr-frontend-guide.md) — How to wire `/v2/video-hdr` (LDR→HDR via IC-LoRA-HDR) into a frontend. *client*
- [lora_api_design.md](./lora_api_design.md) — Original REST-API design for LoRA management; now-shipped, kept as reference. *developer*
- [lora_storage_design.md](./lora_storage_design.md) — Storage + discovery design for the LTX flat-dir LoRA registry. *developer*
- [MCP.md](./MCP.md) — `noodlefinger-mcp` LLM-tool reference: tier-0 docs lookup + tier-1 authenticated jobs, `cut_music_video` macro, LoRA browse. *agent / client*
- [models.md](./models.md) — LTX + Flux + sidecar model inventory and per-model knobs. *developer / operator*
- [MV_EDITING.md](./MV_EDITING.md) — Music-video editing theory + composition grammar + `cut_music_video` tool reference for LLM authors. *agent / developer*
- [OPERATOR_QUICKSTART.md](./OPERATOR_QUICKSTART.md) — First-30-minutes-on-the-box doc: health checks, dashboard, generating a test clip, who to call. *operator*
- [operator-tuning.md](./operator-tuning.md) — Runtime-tunable knobs: rate-limit caps, uvicorn concurrency, fd ceilings, export quality, embeddings + sqlite-vec setup. *operator*
- [outpaint-frontend-guide.md](./outpaint-frontend-guide.md) — How to wire `/v2/video-outpaint` (canvas-expansion via IC-LoRA-Outpaint) into a frontend. *client*
- [PHASE_C_TRAINING_RUNBOOK.md](./PHASE_C_TRAINING_RUNBOOK.md) — Operator playbook for the SFT-on-chosen LoRA trainer: pre-flight, weekly cron, first run, A/B monitoring, rollback, troubleshooting. *operator*
- [PRIVACY_GOVERNANCE.md](./PRIVACY_GOVERNANCE.md) — What the system captures per generation, who can see it, what enters training, how opt-out works — the data-governance surface. *operator / auditor*
- [QUICKSTART.md](./QUICKSTART.md) — Frontend / SDK 5-minute onboarding; minimal payloads to hit each endpoint. *client*
- [retake-frontend-guide.md](./retake-frontend-guide.md) — How to wire `/v2/retake` (regenerate a time window of an existing video) into a frontend. *client*
- [RETRIEVAL_WORKFLOW.md](./RETRIEVAL_WORKFLOW.md) — Workflow guide for Phase B retrieval tools (`find_similar_shots`, `recommend_loras`, `bulk_revalidate`) during MV authoring. *agent / operator*
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) — Symptom → check → fix triage layer for on-call: HTTP errors, validator NULLs, sidecar crashes, capacity FAQ. *operator*
- [UX_GAPS.md](./UX_GAPS.md) — Inventory of UI surfaces today, mapping of every Phase-B/C capability to the surface it should live on, P0/P1/P2 build queue. *developer / product*
- [../AGENTS.md](../AGENTS.md) — Performance-optimization backlog by tier; agent-facing scratchpad of identified-but-not-shipped optimizations. *agent / developer*
- [../CHANGELOG.md](../CHANGELOG.md) — Per-version delta log — every shipped feature, every fix, every rc. *developer*
- [../CLAUDE.md](../CLAUDE.md) — Code cheat-sheet: file anchors, conventions, version highlights, GPU topology, sidecars, validator wiring. Read this when working in the code. *agent / developer*
- [../README.md](../README.md) — Repo landing page: features, base URLs, links to docs/. *client / developer*

## Conventions

A few load-bearing rules for these docs:

- **CLAUDE.md** is the cheat-sheet for working in the code (file anchors, recent version highlights, conventions). Update it when behavior worth remembering changes.
- **API.md is the contract**. Any commit that adds, removes, or changes an endpoint (URL, method, request/response shape, status codes, auth) MUST update API.md in the same commit. Don't split "code change" from "doc change" across commits — drift breaks clients silently.
- **DECISIONS.md is append-only**. ADRs capture context, decision, consequences, revisit-when triggers. If a decision is overturned, add a new ADR that supersedes the old one — don't edit history.
- `*-frontend-guide.md` docs are integration walkthroughs for one endpoint each; they may go stale faster than API.md. When they conflict, API.md wins.
- Several docs are explicitly **superseded** (see `frontend-api-changes.md`, `frontend-lora-integration.md`) — preserved for version history but not authoritative; use the doc they point at.
- If a `_REVIEW_REPORT.md` is present in `docs/`, read it first — it flags cross-doc inconsistencies that the index assumes are agreed upon.
