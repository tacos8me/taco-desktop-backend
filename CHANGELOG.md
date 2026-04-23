# Changelog

All notable changes to taco-backend. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## v1.12.3 — 2026-04-23

### Fix: LoRA path rewriting applies to all remote video jobs, not just outpaint

Swarm audit of the Modal/RunPod dispatch path found that `_dispatch_job_turbo_remote` (server.py) only rewrote the `lora_path` prefix from `LORAS_DIR` to the provider's network-volume mount **when `job.type == JobType.VIDEO_OUTPAINT`**. Any other video job (t2v, i2v, a2v) with a custom user LoRA routed to Modal or RunPod sent the LOCAL filesystem path to the remote worker, which then failed at LoRA file open.

Local turbo (cuda:1 sidecar) didn't trip this because the filesystem is shared. Remote-only silent failure path — hadn't triggered in production because most LoRA traffic was outpaint (IC-LoRA, correctly rewritten).

Fix: drop the `VIDEO_OUTPAINT` gate. Rewrite for every video job type with a `lora_path`. One-line change.

### Not changed

- Remote dispatch field coverage: audit confirmed 100% parity between `ltx_sidecar_client.generate()`, Modal's `GenerateRequest`, RunPod's `GenerateRequest`, and the manager call dispatch. segment_b64 staging works correctly on both remote sidecars.
- 8-concurrency backend soundness: verified via lock inventory. No backend serialization on the dispatch hot path. `_inference_lock` bypassed by all 7 turbo workers; `asyncio.Queue` multi-consumer safe; ThreadPoolExecutor has ample headroom at 8 concurrent jobs.

### Ops

Service restart to load v1.12.3 code. Modal scaling up to 4 workers immediately after.

## v1.12.2 — 2026-04-21

### Fix: revert v1.11.4's `crf=0` on conditioning images — restores motion + 1440p quality

Client reports since v1.11.4 deploy: motionless a2v outputs (prompts explicitly ask for motion; output freezes around reference), degraded 1440p quality, "hard time with a2v generation". One client was observed dialing `image_strength` down to 0.80-0.95 trying to reclaim motion — the right instinct, wrong knob.

Root cause: v1.11.4 set `ImageConditioningInput(..., crf=0)` on every conditioning image across `_run_a2v` and `_run_i2v`. LTX was trained with `DEFAULT_IMAGE_CRF=33` (H.264 preprocessing of every conditioning image) so the VAE encoder expects compressed-video-like input statistics. Feeding uncompressed `crf=0` images is out-of-distribution; the resulting latent is "too clean" and at `strength=1.0` pins too hard — temporal attention propagates the pin across all frames, suppressing motion generation. Effect scales with resolution: at 1440p the reference image produces ~1.78× the latent tokens of 1080p, so proportionally more of the video is "frozen" around the reference.

v1.11.4 was intended to fix a different complaint ("reference vaguely resembles my input" — CRF 33 compression artifacts were visible at frame 0). We traded subject-pin fidelity for motion dynamics and didn't realize it.

Fix: remove `crf=0` from both `ImageConditioningInput` constructions in `split_model_manager.py`. Falls back to `DEFAULT_IMAGE_CRF=33` which matches LTX's training distribution. Restores pre-v1.11.4 quality — motion-rich outputs, stable across resolutions.

Trade-off accepted: frame 0's output has a slight CRF-33 compression artifact (barely-visible blur vs the input image). That's the v1.11.3-and-earlier baseline that shipped without complaint for months.

Scope: 2 lines removed (the `, crf=0` parameter on two call sites). No Pydantic changes, no API shape changes, no composition changes. Byte-identical v1.12.0 behavior for everything except the reference-image preprocessing step.

### Not changed

- v1.12.0 segment-mode chain conditioning (experimental): unaffected — segment VAE encoding is a separate path. `_build_segment_conditioning_latent` doesn't construct `ImageConditioningInput`.
- v1.10.1 `image_strength` default of 1.0: unchanged (matches pre-v1.10.0 effective behavior).
- All other v1.11.x and v1.12.0 work (composition audio timing, chain debug, experimental segment endpoint, dispatch diagnostics) remains.

### Known outstanding

Turbo-mode deadlock reported mid-day 2026-04-21: a2v at 2560×1440 under turbo hung 4+ min with both GPUs idle, threads sleeping, progress=0. Hard restart cleared it. Root cause traced (3-agent investigation): sidecar `_lock` held during torch.cuda.synchronize() that never returns + HTTP client 600 s timeout = 10-min visible hang. Full task list and observability + timeout fixes scoped for v1.12.3+. Until then, **leave turbo OFF for stable service**.

### Ops

Service restarted on v1.12.2 code; turbo remains off.

## v1.12.1 — 2026-04-21

Pure docs-patch release. No code changes. Comprehensive v1.12 documentation push across `docs/API.md`, `CLAUDE.md`, `README.md`, `docs/QUICKSTART.md`, and supporting docs, so client Claude sessions reading the repo get consistent v1.12-current context from every surface. GitHub release v1.12.0 body also rewritten for clarity + working curl/JSON example. See commit `95317e3`.

## v1.12.0 — 2026-04-20

### Feat: multi-frame video-segment chain conditioning (the proper fix) — EXPERIMENTAL / opt-in

> **Experimental**: backend-complete and smoke-tested, but FE migration and visual regression testing have not yet completed. Gate FE behind `flags.v112_seamless_segment`; legacy v1.11.5 `keyframes` path remains fully supported as the default. Do not flip on for prod users until the `flags.v112_seamless_segment` rollout is verified.

v1.10.0 through v1.11.5 tried to chain-condition clips by sending 3 extracted PNG frames as keyframes at `frame_index=[0, 1, 2]`, strength 1.0. Empirical testing at v1.11.5 proved this architecture can only hard-pin pixel frame 0 (via `VideoConditionByLatentIndex`); frames 1-2 are soft-guided via `VideoConditionByKeyframeIndex` (appended context tokens, main video latent still free-denoised). Subject identity drifted cliff-wise at seam 2+ because each chain hop's anchor was itself a free-generated frame from the prior clip's tail. v1.11.3 attempted to route those frames through `image_conditionings_by_replacing_latent` to hard-pin all three, but the replace path operates at latent-frame granularity (1 latent ≈ 8 pixel frames under LTX's causal 8k+1 VAE), which produced a "slideshow" of held stills in pixel frames 1–16 — reverted in v1.11.5. v1.12 is the architecturally correct fix. See commit `c45449f`.

v1.12 fixes this architecturally. FE extracts a contiguous 9-pixel-frame MP4 segment from the prior clip's tail via new `/v2/video/extract-segment` endpoint. Backend VAE-encodes the segment as a 2-latent-frame tensor and pins target's latents [0, 1] via a single `VideoConditionByLatentIndex(latent=multi_frame_latent, latent_idx=0, strength=1.0)`. LTX's causal VAE maps latent 0 → pixel 0, latent 1 → pixel frames 1-8 — so 9 consecutive target pixel frames are hard-pinned at every sigma step, with real continuous motion content from the prior clip (not held stills as in the v1.11.3 slideshow regression).

### New API surface

- **`POST /v2/video/extract-segment`** — body `{video_uri: "storage://...", start_frame: int, num_frames: int}`, where `num_frames ∈ {9, 17, 25, 33}` (8k+1 for k∈{1..4}); v1.12 ships with 9 as the default. Response `{segment_uri: "storage://...", width, height, num_frames, fps}`. Same capability-URL + bearer + `PER_KEY_UPLOAD_BYTES_PER_DAY` quota pattern as `/v2/video/extract-frames`. Segment MP4s are ~500 KB–1.5 MB for 9 H.264 frames — more efficient than 9 lossless PNGs.

- **`AudioToVideoRequest.segment_uri` and `ImageToVideoRequest.segment_uri`** — new optional string field. Mutually exclusive with `image_uri` and `keyframes` (3-way exclusion enforced by Pydantic validator). When set, backend decodes the segment via PyAV, normalizes pixels, VAE-encodes, and pins target's head latents.

### Internals

- `history_store._extract_segment_as_mp4(video_bytes, start_frame, num_frames)` — single-pass PyAV decode of a contiguous frame range, re-encoded as H.264 MP4 (video-only, no audio track). Shares `_FRAME_EXTRACT_SEMAPHORE` with extract-frames.
- `split_model_manager._build_segment_conditioning_latent(segment_path, target_h, target_w, dtype, device, video_encoder)` — mirrors `_build_outpaint_reference_latent`: decodes via `decode_video_from_file`, preprocesses via `video_preprocess`, returns a multi-latent-frame tensor.
- `_image_conds_for_keyframes(..., segment_path=...)` — when `segment_path` is set, routes to segment path (single multi-frame `VideoConditionByLatentIndex`); otherwise falls through to the classical `combined_image_conditionings` (v1.11.5 single-pixel-frame-0 pin + soft-guide for [1, 2]).
- `_run_i2v` and `_run_a2v` accept a new `segment_path` kwarg; `SplitModelManager.generate_image_to_video` and `generate_audio_to_video` forward it to the threadpool submission.
- `_dispatch_job_turbo_remote` base64-encodes the segment MP4 as `segment_b64` for remote workers (Modal + RunPod). `ltx_sidecar_client.generate()` gains `segment_path` / `segment_b64` kwargs.
- Modal + RunPod sidecars: added `segment_b64` to `GenerateRequest`; staging logic writes to `/tmp/<uuid>.mp4` before calling the manager. i2v endpoint now accepts `keyframes` OR `segment_path` instead of requiring `keyframes`.
- Local cuda:1 LTX sidecar (`ltx-sidecar`) also accepts `segment_path` directly (shared filesystem with taco-backend, no b64 staging needed).

### Known limitation (deferred to v1.13)

LTX's causal VAE assumes frame 0 of an encoded segment is the "start" of a video — it replicates that frame for causal padding. Encoding a 9-frame segment from the middle of a clip gives internally-consistent latents for those 9 frames but NOT bit-identical to what the prior clip's full-length encoding produced for the same frames. Residual RMSE expected: 1-3 on a 0-255 scale. Measurable but visually imperceptible in most content. v1.13 candidate: save prior clip's final 2 latent frames to history and reuse them directly (no VAE re-encode, eliminates causal mismatch entirely).

### Back-compat

- Clients not sending `segment_uri`: unchanged. v1.11.5 keyframes path routes through `combined_image_conditionings` as before.
- Single-keyframe `image_uri` requests: unchanged. Frame 0 hard-pin at strength=1.0 via `VideoConditionByLatentIndex`.
- Legacy compositions with v1.11.2+ `keyframes` / `audioDurationSec` / `tailTrimFrames`: byte-identical export.
- `/v2/video/extract-frames` endpoint: unchanged. FE can use either endpoint depending on conditioning mode.
- Composition export (`export_handler.py`): zero changes. `tailTrimFrames` math is abstract over trim value; FE sets `tailTrimFrames=9` on clips whose successors use segment mode.

### FE contract

FE flow: generate clip 0 → on completion, `POST /v2/video/extract-segment {video_uri, start_frame: num_frames - 9, num_frames: 9}` → use returned `segment_uri` on next clip's a2v/i2v request. On composition save, set `tailTrimFrames=9` on non-final clips. Gate UI behind `flags.v112_seamless_segment`; default off until backend verified in prod. See `docs/handover-frontend-v1.10-chain.md` v1.12 section.

### Verification

- 120/120 pytest green
- `/v2/video/extract-segment` smoke-tested: correctly rejects single-frame inputs as `segment_out_of_range`, emits valid MP4 uploads on valid ranges.
- Service restarted on v1.12.0 code.
- End-to-end visual regression of a seamless-segment composition against the v1.11.5 baseline is **pending** — blocker for flipping `flags.v112_seamless_segment` on.

## v1.11.5 — 2026-04-20

### Fix: revert v1.11.3's wrong routing + promote diagnostic log level

Post-deploy empirical test of v1.11.3 + v1.11.4 showed the chain keyframe fix was architecturally wrong. Pinning `VideoConditionByLatentIndex(latent_idx=[0,1,2])` pins LATENT-frame indices, not pixel-frame indices. LTX's causal VAE maps latent 0 → pixel 0, latent 1 → pixel frames 1..8, latent 2 → pixel frames 9..16. Pinning `latent_idx=1` to the user's kf1 held pixel frames 1..8 ALL to kf1's image; `latent_idx=2` held pixel frames 9..16 all to kf2. The clip head visibly "slideshowed" through 17 pixel frames of three held keyframe stills before the model was free to denoise — worse visually than the soft-guide drift v1.11.3 tried to fix.

Verified against real output: pixel frame 2 had RMSE 4.47 to kf1 (nearly identical, held) and RMSE 21.07 to kf0 (completely different). Pixel frame 15 had RMSE 8.98 to kf2 (held). See `docs/debug-v1.11.3-chain-conditioning.md` addendum for the full RMSE matrix.

Revert: `_image_conds_for_keyframes` now unconditionally delegates to `combined_image_conditionings` (classical LTX semantic). Pixel frame 0 gets clean hard-pin via `VideoConditionByLatentIndex`; frames 1, 2 soft-guide via `VideoConditionByKeyframeIndex`. That's the best LTX can do with image-shaped keyframes; true multi-pixel-frame hard-pin requires encoding a **multi-frame video segment** (a v1.12 proper fix, scoped in the debug doc).

Also in v1.11.5:
- Promote the v1.11.4 `logger.info("a2v keyframes=...")` diagnostic to `logger.warning` so it actually emits. Default Python logging threshold is WARNING; info-level lines were silently dropped. Confirmed by inspecting journalctl — no `a2v keyframes` line for a real job that DID fire the dispatch code. Now emits.

Unchanged from v1.11.4:
- `ImageConditioningInput(..., crf=0)` — reference images not re-compressed at CRF 33 before VAE encode.

### Why the hard problem deferred

Hard-pinning consecutive pixel frames requires a multi-frame video-segment encoding as the conditioning source, not three single-image VAE encodings. Options:
- Extend `KeyframeInput` to accept `video_uri + start_frame + end_frame` for segment conditioning.
- New `/v2/video/extract-segment` endpoint returning a storage URI for a short MP4 clip.
- FE sends a segment reference instead of 3 PNG frames.

Backend changes are ~40 LOC but require FE coordination on the contract. Deferred to v1.12.

## v1.11.4 — 2026-04-20

### Fix: reference images no longer silently H.264-CRF33-compressed before VAE encode

User report post-v1.11.3: a2v with a reference image at `image_strength=1.0` still doesn't reproduce the provided reference — "vaguely resembles" the input even though LTX should be hard-pinning frame 0 at that strength. Root cause: `ltx_pipelines.utils.args.ImageConditioningInput` has a `crf: int = 33` field (`DEFAULT_IMAGE_CRF`) that runs every conditioning image through H.264 encode+decode at CRF 33 inside `load_image_and_preprocess` before VAE encoding. CRF 33 is heavy compression — fine detail, color fidelity, and subject identity all visibly degrade. At strength=1.0 the "pinned" frame 0 becomes `VAE_decode(VAE_encode(CRF33(input)))` instead of what the user sent.

LTX uses CRF 33 during training to match compressed-video statistics, and that's correct for video-frame keyframes (LTX-generated tail frames already went through the MP4 encode-decode pipeline so CRF on top of them doesn't hurt much). For a **user-provided reference image at strength=1.0**, CRF 33 actively fights the "this frame should BE my image" intent.

Fix: construct `ImageConditioningInput(..., crf=0)` everywhere taco-backend builds conditioning from keyframes (i2v + a2v, covering both the initial reference-image path and the chain-conditioning path). PNGs from `/v2/video/extract-frames` are lossless so chain frames also benefit — no roundtrip compression.

Also adds a diagnostic log line at a2v / i2v dispatch: `a2v keyframes=[(0, 1.0)] model=ltx-2-3-pro frames=49`. Lets us (and FE) verify actual strengths hitting the pipeline without having to dump Pydantic bodies. Grep with `journalctl --user -u taco-backend | grep "a2v keyframes"`.

Scope: 2 lines (crf=0 on both ImageConditioningInput constructions) + 2 log lines. No API shape change, no Pydantic field added. Safe, additive, and reversible.

### Ops note for operators

v1.11.x changes only take effect after `systemctl --user restart taco-backend` — Python imports happen at process start, and this chain of releases (v1.10.1 → v1.11.4) requires a restart to pick up. If `/v1/system/gpu` reports uptime older than the v1.11.x commit timestamps (visible in `git log`), the process is running stale code.

## v1.11.3 — 2026-04-20

### Fix: chain conditioning actually pins all 3 head keyframes (root cause across v1.10.0 → v1.11.2)

User report after v1.11.2: first seam (clip 0→1) excellent, second seam (clip 1→2) good, third seam (clip 2→3) step-changes to an entirely different scene. Initial keyframe "vaguely resembles" the output by clip 3. Every fix from v1.9.6 through v1.11.2 addressed a real adjacent problem but left the deeper cause untouched.

Root cause found in `ltx_pipelines.utils.helpers.combined_image_conditionings` (the function taco-backend has been calling for all multi-keyframe a2v/i2v paths since v1.10.0): it **branches on `frame_idx == 0`** and uses two semantically different conditioning mechanisms.

- **Frame 0** → `VideoConditionByLatentIndex`: directly replaces the main video's latent tokens at position 0 and sets denoise_mask = 1 − strength. At strength=1.0, the denoise mask is 0 and the output at frame 0 IS the input image, pinned across every sigma step.
- **Frames 1+** → `VideoConditionByKeyframeIndex`: appends keyframe tokens as **auxiliary context** (torch.cat to the latent state) with positional encoding at `frame_idx / fps`. The main video's latent at frame positions 1, 2 is **still noise** that gets denoised from scratch; the keyframe tokens only provide soft attention guidance. At strength=1.0 the keyframe tokens themselves stay clean, but the output frames are free-denoised.

This is the classical LTX sparse-keyframes semantic (first/middle/last at [0, 24, 48]) — correct for that use case. For our **chain conditioning** use case, where the 3 head keyframes at [0, 1, 2] are meant to be exact reproductions of the prior clip's tail, only frame 0 actually pinned. Frames 1, 2 drifted softly. That drift — compounded across three chain hops — is exactly the user's observed cliff at seam 2 and the "initial keyframe vaguely resembles" subject drift.

Fix: `ltx_pipelines/helpers.py` already ships a correct helper — `image_conditionings_by_replacing_latent` (lines 164–191) uses `VideoConditionByLatentIndex` for **every** frame with `latent_idx = img.frame_idx`. All keyframes become hard pins at every sigma step.

`split_model_manager.py` gains a module-level `_image_conds_for_keyframes` helper that auto-detects chain pattern (consecutive `frame_idx` starting at 0) and routes to the right LTX helper:
- Chain pattern (`[0, 1, 2]`, `[0, 1]`, `[0]`) → `image_conditionings_by_replacing_latent` (hard pin on all).
- Sparse pattern (`[0, 24, 48]` first/middle/last) → `combined_image_conditionings` (classical LTX semantic preserved).

Scope: `_run_a2v` and `_run_i2v`, stage 1 + stage 2 (4 call sites). t2v (`images=[]`) and outpaint (empty images + VideoConditionByReferenceLatent) paths are unchanged.

Detection is strict — `[img.frame_idx for img in images] == list(range(len(images)))` — so i2v first/middle/last users keep classical behavior.

### Back-compat
- Single-keyframe requests (frame 0 only): both helpers produce identical output (LatentIndex at 0). Zero behavioral change.
- i2v sparse first/middle/last (e.g., [0, 24, 48]): pattern not chain → combined_image_conditionings → classical LTX semantic preserved.
- Chain-mode a2v/i2v (e.g., [0, 1, 2]): all frames now hard-pinned. Subject should hold across seams. Long-chain grounding (passing the original user image periodically) remains a frontend concern.

See `docs/debug-v1.11.3-chain-conditioning.md` for the full investigation trail: v1.9.6 → v1.9.9 encoder-layer fixes, v1.10.0 content-discontinuity hypothesis, v1.10.1 strength regression, v1.11.0 / v1.11.1 tailTrimFrames churn, v1.11.2 audioDurationSec decoupling, and the actual root cause.

## v1.11.2 — 2026-04-20

### Feat: per-clip `audioDurationSec` decouples audio atrim from video trim

FE audit of the v1.11.0→v1.11.1 churn identified the real fix: the audio-side atrim slice has been clamped to `min(beat_gap, effective_duration)` since v1.10.0, so any video trim silently dropped song material at every seam. v1.11.0's push to `tailTrimFrames=6` made this audible (208 ms per seam); v1.11.1 reverted to `tailTrimFrames=3` as a 83/83 ms split across audio and video (both sub-threshold). Both clamp-based configs gave up song continuity to stay monotonic with the video.

FE now sends an explicit `audioDurationSec: float | None` per clip. When present (> 0, not bool), the exporter uses it verbatim for the `atrim duration=` — audio and video become independent concerns. Absent → v1.11.1 clamp behavior is preserved exactly (backward-compatible for every saved comp on disk today).

FE-preferred path with v1.11.2:
- `tailTrimFrames=6` (drops both safe and unsafe tails; chain regens them at the follower's head → 0 ms visual seam)
- `audioDurationSec = next.beatTime - this.beatTime` for non-final clips (= the full beat gap, typically 2.0 s), `= clip.duration` for the final clip
- Result: song plays with 0 ms dropout across every seam; video cuts progressively earlier than the beats (~208 ms per seam at 49 fr / 24 fps / 2.0 s beats, accumulating to ~832 ms over 5 clips). The final video frame freezes for the accumulated audio-lead while the song completes. Backend's `beat_synced` branch already omits `-shortest`, so freeze-frame tail works without code changes.

Change surface:
- `export_handler.py` — slice_duration loop prefers `clips[i]["audioDurationSec"]` when numeric + positive; falls back to the v1.11.1 clamp otherwise.
- `docs/API.md` — clip schema documents `audioDurationSec` and the two recommended configs (v1.11.2 decoupled vs v1.11.1 clamp).
- `docs/handover-frontend-v1.10-chain.md` — rewritten to describe both paths; v1.11.2 is FE-preferred.
- No Pydantic model changes (composition data is stored as raw JSON via `composition_store`; unknown clip fields already pass through).
- No endpoint shape changes. Field is purely additive.

Backward compat: saved comps without `audioDurationSec` → byte-identical to v1.11.1. Saved v1.11.0 comps (`tailTrimFrames=6`, no `audioDurationSec`) still export but retain the audio-dropout symptom; FE ships a one-shot migration to attach `audioDurationSec = beat_gap` on load.

## v1.11.1 — 2026-04-20

### Fix: revert v1.11.0's `tailTrimFrames=6` recommendation — audio was audibly clipping at every seam

v1.11.0 bumped the FE spec from `tailTrimFrames=3` → `6` to eliminate a ~83 ms visual stutter at every seam. User testing immediately surfaced a worse symptom: audible audio dropout at every seam (~208 ms of song silently skipped per cut). Ultrathink trade-off analysis:

| `tailTrimFrames` | effective video/clip (49 fr @ 24 fps) | audio dropout per seam (2.0 s beat gap) | visible video seam | verdict |
|---|---|---|---|---|
| 0 (legacy) | 2.04 s | 0 ms | hard cut (full content discontinuity) | not chain-compatible |
| **3** (v1.10.0, v1.11.1) | 1.92 s | 83 ms (sub-threshold) | 2-frame backwards jump (83 ms, sub-threshold) | **minimum total perception** |
| 6 (v1.11.0 briefly) | 1.79 s | **208 ms (audible)** | 0 ms | trades one sub-threshold artifact for one audible one |
| clamp removed | 1.79 s | 0 ms | 0 ms | visual cut drifts earlier than the beat — 0.83 s off by seam 4, far worse |

Root cause of the v1.11.0 audio dropout: the v1.9.6 beat-gap atrim slice is `min(gap, effective_duration)`. With `tailTrimFrames=6` the effective duration drops to 1.7917 s (below the 2.0 s gap), and the 208 ms of song between the two clamps to `effective` instead of `gap`, never reaches the timeline at every seam. The clamp is load-bearing (removing it drifts the visual cut progressively earlier than the beat — by seam 4 the cut is 0.83 s BEFORE the beat, much worse than 208 ms of silence). The fundamental math at 49 frames / 24 fps / 2.0 s beats cannot produce a perfectly seamless result on both axes simultaneously — one axis takes the hit; `tailTrimFrames=3` splits it 83/83 ms across both, below perceptual threshold on each.

Fix is spec-revert only, zero code change:
- `docs/handover-frontend-v1.10-chain.md` reverts `tailTrimFrames=3` throughout, adds the trade-off table + "why 6 sounded clipped" + "why 3 is the sweet spot" sections, migration note (6 → 3).
- `docs/API.md` clip schema example `tailTrimFrames: 6 → 3`; prose updated to document the audio-clipping trap for anyone who re-discovers this trade-off.
- `CLAUDE.md` / `README.md` version bump to v1.11.1.
- Pre-existing saved compositions with `tailTrimFrames=6` will still export (just with the 208 ms audio dropout symptom users reported); re-save to apply `3`, or add a FE one-shot migration bumping 6 → 3 on load.

Backward compat: all prior behavior preserved. Byte-identical export for legacy compositions (no `tailTrimFrames`). No API shape changes in v1.11.1.

## v1.11.0 — 2026-04-20

### Stabilization — tailTrimFrames spec correction + v1.10.x rollup

v1.10.1 restored a2v keyframe strength and fixed an obvious regression, but users still reported a "barely noticeable stutter" at every seam in seamless-chain compositions. Ultrathink review found the cause was a spec bug in the v1.10.0 frontend handover doc:

- The chain sends clip N's frames `[N-6, N-5, N-4]` as keyframes at clip N+1's indices `[0, 1, 2]`.
- v1.10.0 spec told the frontend to set `tailTrimFrames=3` — drops ONLY the unsafe Stage-2 artifact tail (`[N-3, N-2, N-1]`).
- Result: clip N still shows `[43, 44, 45]`, then clip N+1 starts with the regenerated versions of the same 3 frames → **2-frame backwards jump + 3-frame content repeat** at every seam.
- Correct value is `tailTrimFrames = safe_tail_count + unsafe_tail_count = 6`. Viewer then sees a monotonic timeline `… 41, 42 | 43', 44', 45', 46', …` — no repeat, no jump.

Fix is spec-clarification only, zero code change:
- `docs/handover-frontend-v1.10-chain.md` bumped to v1.11.0, `tailTrimFrames=3 → 6` throughout, added an ASCII timeline diagram, migration note for existing FE implementations.
- `docs/API.md` clip schema expanded with explicit `tailTrimFrames` math and the bug's history ("using 3 causes a barely noticeable stutter"). `fps` per-clip override documented.
- Version lock at v1.11.0 to signify v1.10 stabilization.

### v1.10.x rollup (recap of what's on master)

- **v1.10.0** — multi-frame chain conditioning: `POST /v2/video/extract-frames`, `AudioToVideoRequest.keyframes`, `tailTrimFrames` composition field. Root-cause fix for seam glitches.
- **v1.10.1** — `AudioToVideoRequest.image_strength` default `0.85 → 1.0` (v1.10.0 silently regressed a2v image conditioning by 15% per sigma step).
- **v1.11.0** — this release. Spec clarification only.

Backward compat: all prior behavior preserved. Legacy compositions (no `tailTrimFrames`) export byte-identical to v1.9.9. Pre-v1.10.0 a2v clients that relied on the default image_strength behavior get the original effective strength=1.0 restored by v1.10.1. No API shape changes in v1.11.0.

## v1.10.1 — 2026-04-20

### Fix: a2v keyframe regression — `image_strength` default 0.85 → 1.0

User report after v1.10.0: a2v keyframes feel weaker ("acts almost like t2v"). Traced — effective image-conditioning strength on the default a2v path:

| Version | effective strength |
|---|---|
| pre-v1.9.5 | `1.0` — hardcoded in `_run_a2v`, Pydantic field didn't exist |
| v1.9.5 – v1.9.9 | `1.0` — Pydantic field accepted but `_submit_job` popped/discarded; `_run_a2v` still hardcoded `1.0` |
| **v1.10.0** | **`0.85`** — Unit A's refactor routed `body.image_strength` through `_resolve_keyframes` to `_run_a2v` for the first time, at the Pydantic default of 0.85 — silently weakening every default-path a2v call |

LTX's conditioning blend `output = denoised * (1 - strength) + clean * strength` is applied at **every sigma step**. A 15 %-per-step leak compounds over 8–30 steps: the image latent dilutes, the audio+prompt take over, the image anchor fades mid-clip. Matches the reported symptom exactly.

Fix (minimal):

- `AudioToVideoRequest.image_strength` Pydantic default `0.85 → 1.0`. Default-path clients get pre-v1.10.0 effective behavior (keyframe pinned at every sigma step). Clients who explicitly set `image_strength` to a lower value now get what they asked for — new capability first working in v1.10.1 (the field was silently ignored in every prior version).
- `model_validator` switched from literal-default comparison to `self.model_fields_set`, so "keyframes + image_strength" conflict detection is robust to default changes and correctly identifies client-explicit values.
- No change to i2v (its 0.85 default has been end-to-end effective since v1.1 — image is i2v's primary anchor). No change to multi-keyframe a2v (keyframe strengths come from the per-keyframe dict).

## v1.10.0 — 2026-04-20

### Feat: multi-frame chain conditioning for seamless MusicVideo export

Root-cause fix for the seam glitches that survived every encoder-layer patch from v1.9.6 through v1.9.9. The v1.9.6 (beat-gap audio slicing), v1.9.7 (HTTP cache + preview hygiene), v1.9.8 (per-input `setpts` + `format=yuv420p`), and v1.9.9 (force-IDR at seam timestamps) fixes cleaned up the encoder stack — PTS is monotonic, keyframes land at every cut, no libopenh264 cross-clip prediction. But frame-by-frame inspection still showed real pixel jumps (seam RMSE 51–65 vs within-clip ~22) and luminance deltas up to ±18 of 255. The upstream cause: every LTX a2v clip independently decides its own exposure, tone, and motion trajectory. Concatenation of independently-generated clips is content-level discontinuity no encoder trick can hide. v1.10.0 attacks the cause: condition each chain follower on the **last 3 safe frames** of its predecessor via multi-frame keyframe conditioning, so the follower's motion trajectory and visual state start from real established continuity. The viewer sees `[clip 0 | clip 1's regenerated tail of clip 0 | clip 1 body | clip 2's regenerated tail of clip 1 | ...]` — same 3 frames are never visible twice because clip N is export-time trimmed by 3.

**Unit A — `export_handler.py` + `AudioToVideoRequest.keyframes`.** New per-clip field `tailTrimFrames: int` (default 0) flows verbatim through `composition_store.py`'s v1.9.5 passthrough. The per-input filter chain now prepends `[i:v]trim=end_frame=<kept>` *before* the v1.9.8 `setpts=PTS-STARTPTS,format=yuv420p` normalization (`trim` preserves original PTS; `setpts` then resets to zero — order matters). Trimmed durations cascade into `effective_durations[i] = duration - tailTrimFrames/fps`, which replaces `clip_durations` in exactly two callsites: the v1.9.6 beat-gap atrim (non-last slice duration clamps to `min(gap, effective_durations[i])`, preventing atrim past EOF) and the v1.9.9 force-IDR seam cumsum (seams computed from `effective_durations[:-1]` so every IDR still lands exactly on a cut). `AudioToVideoRequest` gains `keyframes: list[KeyframeInput] | None` to match `ImageToVideoRequest`; `generate_audio_to_video` and `_run_a2v` are extended to accept the resolved keyframes list via `ImageConditioningInput`. The legacy single-keyframe path through `image_uri` + `image_strength` is unchanged.

**Unit B — `POST /v2/video/extract-frames`.** New endpoint for frontend chain orchestration: body `{video_uri: "storage://...", frame_indices: [int]}` (Pydantic validates 1–16 non-negative indices, server-side dedupe + sort), response `{frames: [{frame_index, storage_uri, width, height}, ...]}` in sorted-index order. Server-side PyAV single-pass iteration (stops at `max(indices)`), lossless PNG output (feeds straight into LTX VAE encoding — JPEG artifacts would propagate forward). Bearer + capability-URL security matching the v1.9.1 `/uploads/get/{id}` pattern. Bounded concurrency via a dedicated `_FRAME_EXTRACT_SEMAPHORE(2)` (separate from the v1.9.7 `_PREVIEW_EXTRACT_SEMAPHORE` — long extracts can't starve preview polling) with a 30 s timeout. Output bytes counted against `PER_KEY_UPLOAD_BYTES_PER_DAY`. Error taxonomy: `404 video_not_found`, `422 frame_index_out_of_range`, `504 pyav_timeout`, `429 upload_quota_exceeded`, `500 extract_failed`.

**LTX strength semantics (confirmed).** Strength is clamped `[0.0, 1.0]` by `KeyframeInput` (Pydantic `ge=0.0, le=1.0`). Blending formula `output = denoised * (1 - strength) + clean_latent * strength` is applied **at every sigma step** (not init-only); `strength=1.0` pins the conditioned latent across the entire denoise loop. Consecutive `frame_index=0,1,2` is fully supported by `VideoConditionByLatentIndex` (frame 0) + `VideoConditionByKeyframeIndex` (frames 1, 2) — the architecture has no sparse-keyframe assumption. Default chain strength is `[1.0, 1.0, 1.0]` across the 3 head frames; a hidden taper to `[1.0, 0.95, 0.9]` is available if drift appears.

**Backward compatibility.** Clips without `tailTrimFrames` (all current production compositions) default to 0 = v1.9.9 behavior exact, byte-identical export. Single-clip exports zero out `tailTrimFrames` regardless of input. The xfade branch silently skips trim (xfade already overlaps — can't stack them). `AudioToVideoRequest` without the `keyframes` field falls through to the unchanged single-keyframe path. The extract-frames endpoint is purely additive; no existing endpoint changes.

**Frontend handover.** Frontend spec for the noodle-v team in `docs/handover-frontend-v1.10-chain.md`.

## v1.9.9 — 2026-04-20

### Fix: REAL cause of video seam glitches — force IDR at seams

The v1.9.8 fix (`setpts=PTS-STARTPTS,format=yuv420p`) addressed PTS monotonicity but didn't solve the perceived seam problem. Frame-by-frame inspection of the v1.9.8 output revealed:
- **1 IDR frame** for the entire 533-frame output (just at t=0).
- libopenh264 was doing cross-clip P-frame prediction at every seam — the first frame of each downstream clip was motion-vector-interpolated from the previous clip's last frame.
- Measured RMSE at seam 249→250: **12.60** (LOWER than the ~15 within-clip delta). The encoder was averaging/smoothing between two unrelated clips, producing a single "bled" frame at every cut. That was the visible glitch.
- Ironically, v1.9.8's `setpts` reset made this WORSE by erasing the only scene-change hint libopenh264 was getting (PTS gaps).

Fix: explicit `-force_key_frames "<t1>,<t2>,..."` with seam timestamps computed from `clip_durations` cumulative sum. libopenh264 inserts a fresh IDR at each seam, so every downstream clip starts from a clean intra frame with zero cross-clip prediction.

After: 6 IDRs in the same output (1 start + 1 per seam), seam RMSE = **65.19** (clean hard cut, 5× higher than within-clip motion). Output bytes grew ~7 % from the extra IDRs — cheap price.

Kept v1.9.8's `setpts=PTS-STARTPTS,format=yuv420p` normalization (still correct for timing + pixel-format safety). This release stacks on top of it.

## v1.9.8 — 2026-04-19

### Fix: video seam glitches on composition export

The concat filter in `export_handler.py` passed `[0:v][1:v]...concat=n=N:v=1:a=0[vout]` with no per-input normalization. Two problems showed up at every seam:

- **PTS discontinuity** — each clip's re-encoded intermediate had its own PTS timeline (not guaranteed to start at 0). concat stitches PTS through, so a clip that started at PTS=0.05 would cause a visible single-frame freeze or jump at the join.
- **Pixel-format drift** — if two adjacent clips had different pixel formats (yuv420p vs yuv444p, subtly different after different encode paths), concat would stall at the change.

Fix: prepend each input with `setpts=PTS-STARTPTS,format=yuv420p` before the concat/xfade chain. The video side now has an equivalent to the audio-side `asetpts=N/SR/TB` reset that v1.9.3 shipped.

Verified: synthetic 3-clip export and real-world MusicVideo export (3 × LTX video + 15-second song, beat-gap audio slicing) both show strictly monotonic PTS at exactly 1/24 s across all seams. Before the fix, frame deltas at clip boundaries could be anywhere from 0 ms (duplicate) to 80 ms (gap).

Paths normalized: simple concat, xfade chain, and single-clip-with-audio. Single-clip-no-audio still short-circuits to raw bytes (no ffmpeg invocation). No API shape change.

## v1.9.7 — 2026-04-19

### Performance pass — HTTP cache headers, conditional GETs, gzip, preview hygiene

Bundled perf wins from a 5-agent audit. All additive, no API shape change, zero new infra (explicitly not Redis — see plan notes).

- **`Cache-Control: public, max-age=<long>, immutable` + manual 304 Not Modified** on `/v2/history/{id}/thumbnail`, `/v2/history/{id}/image`, and `/v1/approved-images/{id}/file`. FastAPI's `FileResponse` sets `ETag` + `Last-Modified` but does NOT honor `If-None-Match` / `If-Modified-Since` — always returns 200 with full body. New `_serve_with_http_cache(path, media_type, request, max_age, immutable=)` helper mirrors Starlette's ETag formula so clients that cached an ETag from a prior FileResponse get 304 hits. Thumbnails → 1-year `immutable` (content-addressed by `thumb_id`); result files + approved images → 30-day `immutable` (30-day retention window). Paired with Cloudflare's default behavior: `public` unlocks edge caching for authenticated requests (CF refuses by default on `Authorization:` headers without `public`).
- **`GZipMiddleware(minimum_size=1024)`**. `GET /v2/history?limit=50` is 26,243 B JSON; gzipped 4,969 B (5.3× reduction). Zero effect on image/video/audio responses (middleware respects mime type). One middleware line.
- **Lazy PyAV preview extract — timeout + semaphore**. `GET /v2/jobs/{id}/preview`'s fallback path for missing video thumbnails used to run PyAV decode on a 100+ MB video inside the default thread pool with no timeout. A malformed MP4 could tie up a slot indefinitely. Now: `asyncio.wait_for(..., timeout=8.0)` + process-wide `asyncio.Semaphore(2)` cap on concurrent lazy extracts. Timeout → `204` (same "keep polling" signal the endpoint already uses).
- **In-memory `approved-images/manifest.json` cache**. `_load_approved_manifest()` caches the parsed list keyed by `(mtime_ns, size)` — any write (including our own `manifest.write_text(...)`) naturally invalidates. Drops a `read_text()` + `json.loads()` off every `GET /v1/approved-images`, `GET /v1/approved-images/{id}/file`, and `POST /v1/approved-images`.
- **Uvicorn access-log off**. `run.sh` now passes `--no-access-log` — our per-route logger covers the interesting cases; uvicorn's default line-per-request was pure journalctl noise. ~30% log-line reduction during normal workload.

Deferred (documented in `/home/ian/.claude/plans/melodic-sniffing-beacon.md`):
- **Capability URL `/v2/thumb/{thumb_id}`** (unauth'd, CF Cache Rule) — phase 2, gated on v1.9.7 measurement.
- **WebP thumbnails** — bundle with phase 2 when we need magic-byte Content-Type sniffing anyway.
- **Redis** — not for thumbnails. Targeted shared-state move when we go multi-instance (rate-limit counters, SSE tokens, pool-scaling targets). See plan for the precise inventory.

## v1.9.6 — 2026-04-19

### Audio-seam fix + turbo dispatch fix

Two fixes surfaced by the v1.9.5 end-to-end validation run.

**Beat-gap atrim (frontend report):** v1.9.3 sliced the song via `atrim duration=clip_durations[i]`, but LTX quantizes to 8k+1 frames (e.g. 2.04 s for 49 frames @ 24 fps) while the frontend's beat grid is typically clean 2.0 s. Result: adjacent atrim slices overlapped by ~40 ms of song content at every seam — audible repeat on each beat transition. Fix: slice duration is now the beat gap `audioStart[i+1] - audioStart[i]` for every clip except the last, and `clip_durations[-1]` for the last (no next beat). Non-monotonic audioStart falls back to `clip_durations[i]` per clip defensively. Output audio length is now `sum(beat_gaps) + clip_duration[last]` — within one LTX-quantization step of video length, no content overlap.

**Turbo worker refuses non-video jobs (my validation found):** turbo and remote workers' dispatch functions hard-crash on non-video job types (`Turbo worker cannot handle export-composition — only video jobs supported`). Pre-v1.9.6, if a turbo worker dequeued an `EXPORT_COMPOSITION` job before the main worker, the job failed immediately. Export under turbo was effectively broken. Fix: new `accept_check: Callable[[Job], bool] | None` param on `job_queue.worker_loop` — when it returns False, the worker re-queues the job and yields (50 ms sleep to avoid hot-spin). Wired on both the local turbo worker and every remote-provider worker with `accept_check=lambda job: job.type in _VIDEO_JOB_TYPES`. Main worker takes everything as before.

Live verified: beat-synced export under turbo completes with audio duration 4.040 s for `audioStart=[0.0, 2.0]` + `duration=2.04` clips — exactly `2.0 + 2.04 = 4.04`. Prior v1.9.5 behavior would have been 4.08 s (2.04 × 2) with 40 ms of song content repeated at the seam.

## v1.9.5 — 2026-04-19

### MusicVideo contract fixes — three bugs from frontend's 5-agent audit

All three are one-way contract breaks (frontend was doing the right thing; backend silently dropped fields on persist). Additive, no breaking changes.

- **`audio_uri` dropped on composition save/update** (`server.py:4488`, `:4527`). The persisted `data` dict was hardcoded `{"clips": ..., "transitions": ...}`, silently stripping every other top-level body field. Result: MusicVideo compositions persisted without audio metadata → reload → re-export → silent MP4. Fix: new `_composition_data_from_body(body)` helper copies the full body minus `name`, defaults `clips`/`transitions` to empty lists. Future frontend additions no longer need server changes.
- **Per-clip `audioStart` round-trip fragile** (same root cause as #1). Since clips is a flat list passthrough it was already surviving the JSON round-trip, but the pattern of rebuilding `data` manually would break any future per-composition field. Now fixed by the passthrough.
- **`AudioToVideoRequest` missing `image_strength`** (`server.py:728`). Pydantic's default `extra="ignore"` silently stripped the field. `_submit_job` has always accepted `image_strength` (pops it at line 1031 with default 0.85), so a2v runs always used 0.85 regardless of client request. Added `image_strength: float = Field(default=0.85, ge=0.0, le=1.0)` to the Pydantic model. Matches `ImageToVideoRequest` exactly.
- **Export route precedence** — `POST /v2/compositions/{id}/export` now falls back to `comp["data"]["audio_uri"]` when the request body omits `audio_uri`. Request body still wins for ad-hoc overrides. Combined with the save-preserve fix, MusicVideo's save → reload → export path produces the same MP4 as the original export.

Verified live: `POST /v2/compositions` with `audio_uri` + `audioStart` on clips → `GET` returns them intact; `PUT` update preserves them too; `POST /export` with empty body queues a job with the stored `audio_uri`; `image_strength=0.42` passes Pydantic validation on a2v.

## v1.9.4 — 2026-04-19

### Fix: real root cause of `avcodec_open2(aac)` EINVAL was 192 kHz input

v1.9.2's thread-lock was defensive hygiene but not the actual bug. Post-deploy logs showed the error still firing on a2v jobs after the lock was in place. Investigation: recent uploads were PCM WAV at **192,000 Hz** sample rate. AAC supports only {96, 88.2, 64, 48, 44.1, 32, 24, 22.05, 16, 12, 11.025, 8, 7.35} kHz — 192 kHz is **not on that list**. The pipeline passed `audio_sample_rate=192000` straight into `_prepare_audio_stream("aac", rate=192000)`, and ffmpeg's AAC encoder init returned EINVAL.

Fix in `split_model_manager.py`:
- `_AAC_SAMPLE_RATES` constant + `_normalize_audio_for_aac(audio)` helper that resamples to 48 kHz via `torchaudio.functional.resample` when the source rate isn't in the supported set. No-op on already-compatible rates.
- Called inside `_video_to_bytes` before handing audio to `encode_video`, so all video types with passthrough audio (primarily a2v) get the guard.

Verified with the actual production upload `b1378db45...` (stereo PCM at 192 kHz, 5 s): source `(1, 2, 960000)` → resampled `(2, 240000)` at 48 kHz → encodes to 104,804-byte MP4 cleanly. Previous error was `av.error.ValueError: [Errno 22] Invalid argument: 'avcodec_open2(aac)'` at `container.mux()`.

v1.9.2's `_ENCODE_LOCK` stays — concurrent-encode hygiene is still worth having even though it wasn't this bug's cause.

## v1.9.3 — 2026-04-19

### Composition export: per-clip audio segmentation + AAC encoder flag

Two changes in `export_handler.py`, both additive — no breaking changes.

- **Per-clip audio segmentation** (MusicVideo mode). When every clip carries a numeric `audioStart` field (seconds into the source song where that clip's audio window begins), the song is sliced per clip via `atrim=start=X:duration=Y,asetpts=N/SR/TB` and concatenated 1:1 with the video concat (`concat=n=N:v=0:a=1[aout]`). Audio and video stay exactly aligned even across LTX's `8k+1` frame-count quantization — atrim `duration=` uses the same `clip_durations[i]` the video concat uses. Drops `-shortest` on this path (audio length equals video length by construction).
- **Fallback to legacy full-song overlay** when any clip lacks `audioStart` (timeline-mode compositions pre-dating the field), when xfade transitions are present (audio-crossfade alignment is a separate design), or when no audio is attached (video-only export, unchanged since v1.7). Bit-identical output on these paths vs v1.9.2 — verified via md5sum.
- **`-strict -2` on the AAC encoder flag**. Enables the native AAC encoder on ffmpeg builds where it's flagged experimental (symptom: `avcodec_open2(aac) EINVAL` from the subprocess, distinct from the v1.9.2 PyAV race). No-op on builds where AAC is already stable (e.g. this box's ffmpeg 6.1.1).
- Defensive guards: `audioStart` must be `int|float` and NOT `bool` (Python's `isinstance(True, int)` gotcha); empty `clips` list can't trigger beat-synced path; composition JSON round-trip through `composition_store` preserves `audioStart` verbatim (no Pydantic coercion).

## v1.9.2 — 2026-04-19

### Fix: concurrent PyAV encode race on `avcodec_open2(aac)`

Under turbo mode (2 local workers), two a2v jobs hitting `_video_to_bytes` around the same time intermittently failed with `av.error.ValueError: [Errno 22] Invalid argument: 'avcodec_open2(aac)'` during `container.mux()` / `start_encoding()`. FFmpeg's muxer initialization (opening the libx264 + AAC encoders) is not fully thread-safe across concurrent output containers in the same process. Single-threaded repro of the exact failing file works byte-for-byte.

Fix: module-level `_ENCODE_LOCK = threading.Lock()` in `split_model_manager.py` wraps every `encode_video(...)` call site (the single funnel `_video_to_bytes`). Denoising still runs in parallel under turbo (the expensive 10-60s part); only the final ~1-2s MP4 encode tail serializes. Throughput impact ≤ 5% on typical 5-30s videos. Applies to every video job type (t2v, i2v, a2v, retake, outpaint).

## v1.9.1 — 2026-04-19

### `GET /uploads/get/{upload_id}` — serve uploads back

Routing gap fix. The upload store has always been disk-backed + persistent, but there was no GET route to read files back. Frontend consumers (noodle-m MusicVideo tab reloading an uploaded song, composition-export audio preview) hit 404s.

- **New**: `GET /uploads/get/{upload_id}` returns the raw bytes with an inferred `Content-Type` (`image/jpeg`, `image/png`, `image/webp`, `audio/wav`, `audio/mpeg`, `audio/flac`, `audio/ogg`, `video/mp4`; fallback `application/octet-stream`). Auth via global bearer middleware — the 128-bit `uuid4` hex ID is the capability.
- **Errors**: `400 invalid_upload_id` (malformed), `404 upload_not_found` (valid ID, no file), `500 upload_read_failed` (disk I/O).
- **Deferred by design**: no TTL, no per-key ownership enforcement, no signed URLs, no HTTP Range — add selectively if needed.
- New helper `_infer_media_type_from_magic(head) -> str` alongside the existing `_content_type_matches_magic` (v1.8.2).

## v1.9.0 — 2026-04-19

### Composition export: optional audio overlay

`POST /v2/compositions/{comp_id}/export` now accepts an optional body `{"audio_uri": "storage://<id>"}`. When set, ffmpeg muxes the referenced audio file onto the stitched video output (AAC @ 192kbps, `-shortest`). No body / empty body preserves pre-v1.9 video-only behavior. Single-clip + audio works via an inserted `[0:v]null[vout]` pass-through filter (prior single-clip short-circuit that returned raw bytes is bypassed when audio is present).

### RunPod as a second remote-sidecar provider alongside Modal

v1.6's single remote-sidecar pool becomes a **multi-provider** pool. Modal and RunPod run side-by-side, each with independent target/active/max worker counts. Operators can burst to both simultaneously (up to `2 local + 4 modal + 2 runpod = 8` concurrent video workers) or pick whichever is cheapest / has availability. No Modal retirement — existing deployments keep working unchanged.

**Why**: RunPod RTX PRO 6000 Blackwell serverless is cheaper than Modal (~$2.66/hr active vs ~$3.03/hr), the RunPod account has free credits, and two providers = redundancy against single-vendor outages.

#### Backwards compatibility

All pre-v1.9 deployments keep working. `LTX_REMOTE_SIDECAR_URL` / `LTX_REMOTE_SIDECAR_TOKEN` / `LTX_REMOTE_SIDECAR_MAX_WORKERS` are still honored and transparently aliased to the `modal` provider. The legacy `POST /v1/system/pool/remote-workers {"count": N}` body shape still scales modal only. Legacy flat fields (`remote_sidecar_configured`, `remote_worker_target/active/max`, `remote_sidecar_url`) stay in the `GET /v1/system/pool` response as aliases to the modal provider.

#### API — breaking additions (response shape expanded, not removed)

- **`GET /v1/system/pool`** adds `providers: {modal: {configured, url, target, active, max}, runpod: {...}}`. Legacy flat fields preserved.
- **`POST /v1/system/pool/remote-workers`** now accepts `{"modal": N, "runpod": M}` alongside the legacy `{"count": N}`. Response returns the same shape as `GET /v1/system/pool` plus `applied_now`.
- **`POST /v1/system/pool/remote-workers/{provider}`** (new) — cleanest RESTful per-provider scale, `{provider}` ∈ `{"modal", "runpod"}`.

#### Config

- New env vars: `LTX_MODAL_SIDECAR_URL/TOKEN/MAX_WORKERS`, `LTX_RUNPOD_SIDECAR_URL/TOKEN/MAX_WORKERS`. Modal vars fall back to `LTX_REMOTE_SIDECAR_*` when unset.
- New `config.LTX_PROVIDER_LORAS_MOUNT` maps provider → LoRA mount path for outpaint LoRA rewrites (Modal `/mnt/nvme-1/huggingface/loras/`, RunPod `/runpod-volume/loras/`).
- `LTX_RUNPOD_MAX_WORKERS` defaults to 2 (matches the endpoint's `workers.max` in `endpoint.yaml`).

#### Internal refactors

- `ltx_sidecar_client.ltx_remote_sidecars: dict[str, LtxSidecarClient]` replaces the single `ltx_remote_sidecar` module-level. The singular name is kept as a backwards-compat alias pointing at the modal entry.
- `server.py` pool state becomes per-provider dicts: `_remote_worker_targets` + `_remote_worker_tasks` keyed by provider. `_PROVIDERS = ("modal", "runpod")`.
- `_dispatch_job_turbo_remote(job, *, provider: str)` gains a provider kwarg. `_scale_remote_pool()` uses `functools.partial` to bind each worker task to its provider at spawn time.
- Dashboard grows a second row ("RunPod Pool") mirroring the Modal row. JS walks `data.providers` with fallback to legacy flat fields.

#### New repo: `/mnt/nvme-1/servers/ltx-sidecar-runpod/`

- `runpod_app.py` — FastAPI app mirroring `modal_app.py::fastapi_app`. `/ping` health probe, `/health`, `/load`, `/unload`, `/generate` all share the Modal client contract.
- `Dockerfile` — `runpod/pytorch:2.11.0-py3.12-cuda13.0-ubuntu24.04` base + LTX-2 editable install + minimal taco-inference-deps.
- `download_weights.py` — one-shot script to populate the RunPod Network Volume (~80 GB).
- `endpoint.yaml` — RunPod Load-Balancing Serverless config (GPU `RTX_PRO_6000`, `min_workers=1`, `max_workers=2`).

#### Migration

- No action required for existing single-provider Modal users — old env vars still work.
- To add RunPod: build + push the image at `/mnt/nvme-1/servers/ltx-sidecar-runpod/`, create the Serverless endpoint + Network Volume, populate weights, add `LTX_RUNPOD_SIDECAR_URL` + `LTX_RUNPOD_SIDECAR_TOKEN` to `.env`, restart backend.
- **Security note**: rotate any RunPod API keys that were shared in chat/transcripts during planning. The `SIDECAR_AUTH_TOKEN` secret on the RunPod endpoint is independent of the user-account API key.

## v1.8.2 — 2026-04-18

### server.py security sweep — admin gate, quotas, validation, timing hardening

Eight SEC findings from the v1.8.1 audit, all landing in `server.py`. No new dependencies. No endpoint URLs change. The one behavioural change that matters to clients is the admin gate on 12 mutation endpoints — see migration note below.

- **SEC P0-2 — Admin gate on 12 mutation endpoints.** `POST /v1/system/{pause,resume,turbo,config,config/reset,flux-config,flux-config/reset,sampler,pool/remote-workers}` and `POST /v1/{flux,ltx}/{unload,reload}` now require the caller's bearer to appear in a new `.admin_keys` file (or `TACO_ADMIN_KEY` env var). On mismatch: `403 admin_required`. Read endpoints (`GET /v1/system/{pool,config,flux-config,sampler}`) stay user-level. **Backwards-compat bridge**: when `.admin_keys` is empty, every entry in `.api_keys` is treated as admin (preserves pre-v1.8.2 behaviour), and a WARN is logged at boot so ops notices the degraded posture. When `.api_keys` is also empty, auth is globally off and the gate is a no-op.
- **SEC P1-3 — Per-API-key queue caps.** New `PER_KEY_QUEUE_CAP` (default 3), `PER_KEY_MUSIC_CAP` (2), `PER_KEY_BATCH_CAP` (2). Enforced BEFORE the global `MAX_QUEUE_DEPTH` / `MAX_MUSIC_PENDING` / `MAX_BATCH_QUEUE_DEPTH` caps, so one bearer can't claim the whole queue. Breach returns `429 per_key_queue_full` + `Retry-After: 30`. Counters keyed by `sha256(api_key)` — raw bearers never land in the map. Decremented from `worker_loop`'s `finally` (via new `on_complete` callback on `job_queue.py::worker_loop`), from `_run_music_job`'s `finally`, and from `batch_worker` completion.
- **SEC P1-5 — CharRankResponse validation.** `/v2/char/rank` previously parsed whatever JSON the vision model emitted and echoed it to the client. Now validates against a new `CharRankResponse` Pydantic model (`score` 0–10, `analysis.{face_match,eyes,proportions,overall_likeness}` 1–10, `edits.{add,remove,modify}`). Failures return `502 char_rank_schema_violation` with the Pydantic detail truncated to ≤500 chars.
- **SEC P2-1 — Constant-time bearer compare.** Middleware `any(compare_digest(...) for key in API_KEYS)` short-circuited on first match, leaking set membership via wall-clock timing. Replaced with full-iteration compare at both the middleware and inside `_require_admin`.
- **SEC P2-3 — Per-key upload byte quota.** New `PER_KEY_UPLOAD_BYTES_PER_DAY` (default 10 GiB). Rolling 24h window keyed by `sha256(api_key)`. Applied in `PUT /uploads/put/{id}` (early peek via `Content-Length`, final check after body read) and `POST /v1/loras`. Breach returns `429 per_key_upload_quota_exceeded` + `Retry-After: 3600`.
- **SEC P2-4 — Per-key active-LoRA count.** New `PER_KEY_LORA_COUNT` (default 20). Breach returns `429 per_key_lora_count_exceeded`. Decremented on `DELETE /v1/loras/{id}`.
- **SEC P2-7 — Magic-byte upload Content-Type check.** `PUT /uploads/put/{id}` now peeks the first 16 bytes of the body and rejects with `422 content_type_mismatch` when the declared `Content-Type` doesn't match the file's magic (JPEG `FF D8`, PNG `89 50 4E 47`, WebP `RIFF..WEBP`, MP4 `ftyp` at offset 4, MP3 `ID3` / `FF FB`, WAV `RIFF..WAVE`, FLAC `fLaC`, Ogg `OggS`). Lenient on `application/octet-stream` and unrecognized/missing content-types — those pass through unchanged.
- **SEC P2-8 — Dedup auto-exit-turbo.** When the Modal sidecar flaps, every failed remote-turbo job previously spawned its own `_auto_exit_turbo_on_sidecar_failure` task. Added a module-level `_exit_turbo_scheduled` flag + `_schedule_auto_exit_turbo()` wrapper: one exit task max in flight, subsequent failures log a WARN and return.
- **SEC P2-10 — Manifest type guard.** All three `approved-images/manifest.json` load paths now verify `isinstance(manifest, list)` after parsing and reset to `[]` on mismatch (with a WARN).

### Migration notes for clients

- **Admin gate** is the one user-visible change. If you've been using a single bearer for both generation AND system operations, either:
  1. Create `/mnt/nvme-1/servers/taco-backend/.admin_keys` with the operator bearer(s), one per line — recommended.
  2. Do nothing. The backwards-compat bridge keeps every `API_KEYS` entry admin. A `logger.warning` at boot (`admin auth disabled: .admin_keys is empty`) tells you the gate is degraded.
- **Per-key quota 429s** are new error codes. Clients that already handle `queue_full` can treat `per_key_queue_full` identically (same `Retry-After: 30`). Bulk-upload clients should expect `per_key_upload_quota_exceeded` with `Retry-After: 3600` once a bearer crosses 10 GiB in a 24h window.
- **`422 content_type_mismatch`** on uploads means the declared `Content-Type` header doesn't match the file's magic bytes. Either correct the header or send `application/octet-stream` (explicitly exempt).
- **`502 char_rank_schema_violation`** replaces `500 "Failed to parse vision model response"` when the vision model emits malformed JSON.

### Non-server hardening

- **SEC P2-2 — /dev/shm size guard on MP4 tmpfile** (`split_model_manager.py`). Concurrent turbo encodes (2 local + up to 4 Modal workers) could each land several hundred MB of intermediate MP4 on `/dev/shm`, and when the tmpfs ceiling was hit we saw the ltx-sidecar freeze on `kmalloc`. Added `_pick_tmp_dir(estimated_bytes)` which queries `shutil.disk_usage(config.MP4_TMPDIR)` and falls back to `/tmp` (NVMe) with a WARN log when free bytes drop below `max(estimated * 3, 2 GB)`. Every `_run_*` call site now passes an estimate derived from `num_frames × width × height × 3 × 1.2`. `_video_to_bytes`'s legacy signature still works — `estimated_bytes` is an optional kwarg defaulting to a conservative 500 MB.
- **SEC P2-5 + P2-6 — History blob caps + WAL checkpoint cadence** (`history_store.py`). `params_json` is now capped at 100 KB and `gen_config_json` at 50 KB; over-limit blobs are replaced with a `{"__truncated__": true, "original_bytes": N, "preview": "..."}` sentinel (first 4 KB of the original). Prevents a single rogue request from inflating the history row to multi-MB. Added a write counter + automatic `PRAGMA wal_checkpoint(TRUNCATE)` every 500 rows via new `checkpoint_wal(mode="TRUNCATE")` method; logs at INFO if the WAL file was >1 GB immediately before the checkpoint.
- **SEC P2-11 — Bounded retry on `IdentityFeatureTransfer` blend failures** (`flux_identity.py`). The blend-exception path silently swallowed every failure and returned the raw attention output unmodified. A shape-mismatch regression could silently produce an identity-free image across all 6 hooks × N steps, leaving the client no signal. Added `self._consec_failures` on `IdentityFeatureTransfer`: first failure logs WARN with shape info, 5 consecutive failures re-raises so `identity_session`'s `try/finally` tears down the forward hooks and the job aborts cleanly.

### Files changed

| File | Change |
|------|--------|
| `server.py` | 11 new helpers (`_require_admin`, `_constant_time_match`, `_sha256_key`, per-key counter helpers, upload-window helpers, `_content_type_matches_magic`, `_schedule_auto_exit_turbo`, `_decr_queue_on_complete`); 12 admin-gated handlers gain `request: Request` + gate check; middleware compare fixed; `CharRankResponse` / `CharRankAnalysis` / `CharRankEdits` Pydantic models + `/v2/char/rank` validation; three manifest type guards; per-key increment+decrement wired in `_submit_job` / `v2_music` / `v2_batch_submit` / `_run_music_job` / `batch_worker`; startup-time admin-posture log |
| `config.py` | `ADMIN_KEYS` loader (from `.admin_keys` / `TACO_ADMIN_KEY`); `PER_KEY_QUEUE_CAP` / `PER_KEY_MUSIC_CAP` / `PER_KEY_BATCH_CAP` / `PER_KEY_UPLOAD_BYTES_PER_DAY` / `PER_KEY_LORA_COUNT` with env-var overrides |
| `job_queue.py` | `worker_loop(..., on_complete=None)` — optional terminal-state callback invoked from the `finally` block; exceptions inside the callback logged but don't crash the worker |
| `split_model_manager.py` | Added `_pick_tmp_dir()` + `_estimate_mp4_bytes()` module-level helpers, `_SHM_MIN_FREE_BYTES` + `_DEFAULT_ENCODE_ESTIMATE_BYTES` constants, and optional `estimated_bytes` kwarg on `_video_to_bytes`. All 7 call sites (`_run_t2v`, `_run_t2v_hq`, `_run_i2v`, `_run_a2v`, `_run_retake`, `_run_outpaint` ×2) pass an estimate |
| `history_store.py` | New `_truncate_json_blob()` helper + `_HISTORY_PARAMS_MAX_BYTES` / `_HISTORY_GEN_CONFIG_MAX_BYTES` / `_HISTORY_TRUNCATED_PREVIEW_BYTES` / `_HISTORY_WAL_CHECKPOINT_EVERY` / `_HISTORY_WAL_WARN_BYTES` constants; `HistoryStore.save()` truncates before INSERT and bumps `_write_count`; new `checkpoint_wal(mode)` method |
| `flux_identity.py` | `_MAX_BLEND_FAILURES` constant; `IdentityFeatureTransfer._consec_failures` counter; `_hook_fn` logs WARN on first failure, re-raises after 5 consecutive |
| `docs/API.md` | Admin gate noted under affected endpoints; Error taxonomy additions (`admin_required`, `per_key_queue_full`, `per_key_upload_quota_exceeded`, `per_key_lora_count_exceeded`, `content_type_mismatch`, `char_rank_schema_violation`) |

---

## v1.8.1 — 2026-04-18

### Security hardening + canonical public URL + frontend service persistence

**Public base URL is now `https://api.noodlefinger.io`.** `https://taco.noodlefinger.io` was retired at the same time — its DNS record was removed, so it no longer resolves. Hard cutover, not an alias overlap. If you see DNS failures on clients pointing at `taco.` that's the reason; repoint them at `api.` and they'll work unchanged (same Cloudflare Tunnel, same origin, same auth, same request shape).

**SEC P0-1 — IDOR ownership gate on `/v2/jobs/*` and `/v2/batch/*`** (`server.py`). Before this release, any authenticated bearer could fetch / cancel any other tenant's job or batch by guessing the 128-bit ID (or via any ID leak through logs, SSE `?token=` query params, screenshots, etc.). Added a `_require_owner(owner_key, request, *, sse_token=None)` helper and injected it into 8 handlers: `GET /v2/jobs/{id}`, `/preview`, `/result`, `/stream`, `DELETE /v2/jobs/{id}`, `GET /v2/batch/{id}`, `/result/{index}`, `DELETE /v2/batch/{id}`. Cross-tenant requests now return `404 Not found` — same shape as an unknown ID (no existence oracle). Constant-time compare via `hmac.compare_digest`. Legacy jobs/batches with empty `api_key` remain accessible (backwards-compat); history + approved-images endpoints were already SQL-scoped by `api_key_hash` so they were unaffected.

**SEC P1-1+P1-2 — Dashboard + GPU telemetry moved to a LAN-only admin companion** (`dashboard_server.py`, `taco-dashboard.service`). The previous `GET /dashboard` and `GET /v1/system/gpu` were in the public server's no-auth whitelist, exposing the ops SPA and live GPU state (model, memory, temperature, utilization, tenant info, gen_config) to the internet via `api.noodlefinger.io`. Both routes are now removed from the whitelist and 401 on the public host. A tiny FastAPI companion on `192.168.1.80:8099` (LAN-bound, not routed through Cloudflare Tunnel) serves `dashboard.html` and transparently proxies every other path to `127.0.0.1:8090` with the caller's `Authorization` header. SSE streams are passed through. Access from off-LAN requires an SSH tunnel (`ssh -L 8099:192.168.1.80:8099 ...`). `taco-dashboard.service` is systemd-user-managed, enabled for boot.

**Ops: noodle-i / noodle-v / noodle-mv frontend services persisted via systemd.** After the box crash earlier today, three Vite/Express frontends (`i.noodlefinger.io`, `v.noodlefinger.io`, `mv.noodlefinger.io`) didn't auto-restart because they were running from manual `pnpm dev` invocations. Created `noodle-i.service`, `noodle-v.service`, `noodle-mv.service` with `Type=simple`, `KillMode=control-group`, `Restart=on-failure`, enabled for boot. The Cloudflare Tunnel ingress map is unchanged (`i → :5173`, `v → :5174`, `t → :5175`, `mv → :5176`, `taco → :8090`). `run-dashboard.sh` / `dashboard_server.py` sit on the new LAN-only port 8099.

### Files changed

| File | Change |
|------|--------|
| `server.py` | New `_require_owner()` helper; 8 job/batch handlers gain `request: Request` + ownership check; middleware whitelist trimmed to `/health` + `/v1/approved-images/events` only; `/dashboard` route now 404 stub |
| `dashboard_server.py` *(new, ~125 LOC)* | FastAPI on `192.168.1.80:8099`: serves `dashboard.html` + transparent proxy of `/v1`, `/v2`, `/health` etc. to `127.0.0.1:8090` with `Authorization` forwarded. SSE passthrough. Not in OpenAPI |
| `README.md`, `docs/API.md`, `docs/QUICKSTART.md` | Base URL set to `api.noodlefinger.io` + retirement notice for `taco.noodlefinger.io` |
| `~/.config/systemd/user/{noodle-i,noodle-v,noodle-mv,taco-dashboard}.service` *(new)* | systemd user units for the three frontend apps and the new admin dashboard. All `Type=simple` + `KillMode=control-group`, all enabled for boot |

### Migration notes for clients

- **Required**: repoint from `taco.noodlefinger.io` to `api.noodlefinger.io`. The old DNS is gone — clients still pointed at `taco.` get NXDOMAIN and fail immediately.
- If you hit 404 on a job that used to work cross-key — that's the IDOR fix. Use the same bearer that submitted the job.
- If you hit 401 on `GET /dashboard` or `GET /v1/system/gpu` — intentional; use the LAN admin server on 8099 (SSH tunnel off-LAN).

---

## v1.8.0 — 2026-04-18

### Flux 2 Klein identity preservation on `/v2/image-edit`

Adds three optional fields to `ImageEditRequest` for subject-identity-preserving edits on Klein, ported from [`capitan01R/ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer#identity-preservation-nodes) (MIT). Fully additive — default behaviour is unchanged.

- **`preserve_identity: bool = false`** — master switch. `false` is a zero-cost no-op path. `true` is rejected with `422 preserve_identity_klein_only` for any `model` other than `flux2-klein` (hooks target the Klein KV transformer specifically).
- **`identity_strength: float = 0.5`** — overall dial ∈ [0, 1]. Scales both internal hooks proportionally. `0.5` reproduces the upstream plugin's recommended defaults; `1.0` is maximum. `0.0` is treated as a no-op even if `preserve_identity=true`.
- **`identity_mode: "balanced" | "faithful" | "loose"` = "balanced"`** — three curated presets. Each pairs one `IdentityGuidance` mode (latent-space pull) with one `IdentityFeatureTransfer` mode (attention-output steering):
  - `balanced` = `adaptive` + `cosine_pull` (plugin default)
  - `faithful` = `direct` + `topk_replace` (stronger lock)
  - `loose` = `channel_match` + `mean_transfer` (palette/lighting fidelity, flexible geometry)

Under the hood:
- **IdentityGuidance** runs in `callback_on_step_end`, pulls the denoised latent toward the VAE-encoded first reference inside the sampling window `[0, 0.5]`. Three modes implemented: `direct`, `adaptive` (cosine-weighted), `channel_match`.
- **IdentityFeatureTransfer** uses `torch.nn.Module.register_forward_hook` on `Flux2Attention` within the middle-plus 25–88% of the 8 double-stream blocks (`transformer_blocks[2..6]` on the current Klein 9B). Three modes implemented: `cosine_pull`, `topk_replace`, `mean_transfer`. Because Klein KV caches reference K/V after step 0 (ref tokens not in subsequent attention sequences), the hook is self-gating: it observes `T_img > expected_gen_tokens` before blending.
- Both hooks share the same per-request `reference_latent` derived from resizing `image_uris[0]` to target `(width, height)` then VAE-encoding once.
- Hook install + teardown lives in `flux_identity.identity_session()` — a strict `contextmanager` with `try/finally` hook removal, important because `FluxManager._pipe` is long-lived across requests; any leaked state would corrupt subsequent non-identity edits.

### Files changed

| File | Change |
|------|--------|
| `flux_identity.py` *(new, ~340 LOC)* | `IdentityGuidance` + `IdentityFeatureTransfer` + `_resolve_identity_preset` + `identity_session` context manager |
| `server.py` | `ImageEditRequest` gains 3 fields; `image_edit` (v1) + `v2_image_edit` (v2) validate Klein-only + forward new params through the dispatch params dict |
| `flux_manager.py` | `_edit()` accepts + forwards the 3 kwargs; when active, prepares reference latent via `pipe.vae.encode(resized first image)` and wraps the pipeline call in `identity_session` |
| `docs/API.md` | New "Identity preservation" subsection under `POST /v1/image-edit` documenting preset table, known limits, and timing delta |

### Client guidance

- Existing clients are unaffected — all three fields default to zero-cost off.
- For best results: portrait-style first reference, `balanced` preset, `identity_strength=0.5–0.7`. Bump to `faithful` when the edit prompt is radically different from the reference (e.g., "now as a statue"). Use `loose` for pose / scene changes where strict pixel lock would fight the prompt.
- No new endpoint — this continues to ship through `POST /v2/image-edit` (async) and `POST /v1/image-edit` (sync).

---

## v1.7.0 — 2026-04-17

### IC-LoRA video outpaint — new `/v2/video-outpaint` endpoint

Adds a new async endpoint that expands a source video's canvas to a larger target resolution by letterboxing with pure-black padding, then uses an IC-LoRA to fill the black regions with temporally coherent generated content. Backed by [`oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint`](https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint) (Apache 2.0).

Fully additive — no existing endpoints, request shapes, headers, or response semantics changed.

- **Endpoint**: `POST /v2/video-outpaint`. Returns 202 + submission envelope, same pattern as other v2 endpoints.
- **Request**: `VideoOutpaintRequest` with `video_uri`, `prompt`, `target_resolution` (reuses existing `Resolution` literal union), `position` (9-value enum: center + 4 edges + 4 corners), `duration`, `fps`, `seed`, `enhance_prompt`, `lora` (optional override; defaults to `id="ic-lora-outpaint"`), `conditioning_strength` ∈ [0, 1], `skip_stage_2` escape hatch.
- **Pipeline**: 2-stage distilled, patterned on `_run_t2v` fast branch with an IC-LoRA `VideoConditionByReferenceLatent` appended to stage 1 conditionings:
  - Stage 1 at half target res with the outpaint LoRA fused into the distilled transformer; letterboxed source is VAE-encoded and passed as `VideoConditionByReferenceLatent` (optionally wrapped in `ConditioningItemAttentionStrengthWrapper` when `conditioning_strength < 1.0`).
  - Stage 2 (if not skipped) upsamples 2x and refines at full target res. LoRA stays fused across both stages (accepted deviation from upstream `ltx_pipelines.ic_lora.ICLoraPipeline`, which drops LoRA for stage 2; reloading mid-request would cost ~30 s of fusion work — see plan notes for the tradeoff).
- **Letterbox**: scale source proportionally to fit target, pad remainder with -1 in normalized pixel space (= RGB 0,0,0 after VAE decode = the LoRA's training black sentinel). Temporal dim padded with black frames if source is shorter than `num_frames`. `reference_downscale_factor` read from LoRA safetensors metadata (default 1).
- **Output**: silent MP4 (no audio). Source audio passthrough deferred to v1.7.x.
- **Turbo + Modal parity**: outpaint works under turbo with local cuda:1 sidecar and via the Modal pool. Modal container has the outpaint LoRA pre-staged on the HF volume at `/mnt/nvme-1/huggingface/loras/ic-lora-outpaint.safetensors` (populated by `modal run modal_app.py::download_weights`); `_dispatch_job_turbo_remote` rewrites the local LoRA path to that volume path before calling remote. Custom IC-LoRAs over remote fall back to single-machine dispatch for v1.7.0.
- **LoRA registered** under id `ic-lora-outpaint` (strategy `ic_lora_outpaint`). Download + registration script: `scripts/register_outpaint_lora.sh` (idempotent).

### Files changed

| File | Change |
|------|--------|
| `server.py` | New `OutpaintPosition` + `VideoOutpaintRequest`; new `v2_video_outpaint` handler with default-LoRA substitution; `_dispatch_job` branch for `JobType.VIDEO_OUTPAINT`; `_dispatch_job_turbo` + `_dispatch_job_turbo_remote` pass outpaint extras; `_VIDEO_JOB_TYPES` includes new type |
| `job_queue.py` | `JobType.VIDEO_OUTPAINT` enum value + `_MEDIA_TYPES` mapping (`video/mp4`) |
| `split_model_manager.py` | New module-level helpers `_read_lora_reference_downscale_factor` + `_build_outpaint_reference_latent`; new `_run_outpaint` method (2-stage, IC-LoRA conditioning); new `generate_outpaint` async wrapper; added `VideoConditionByReferenceLatent` + `ConditioningItemAttentionStrengthWrapper` + `decode_video_by_frame` imports |
| `ltx_sidecar_client.py` | `generate()` accepts `position`, `conditioning_strength`, `skip_stage_2` kwargs; payload includes them when set |
| `loras/registry.json` | New entry: `id=ic-lora-outpaint`, name `IC-LoRA Outpaint`, strategy `ic_lora_outpaint` |
| `loras/ic-lora-outpaint.safetensors` | Symlink to downloaded `ltx-2.3-22b-ic-lora-outpaint.safetensors` (1.3 GB, 960 tensors, metadata `reference_downscale_factor=1`) |
| `docs/API.md` | New `POST /v2/video-outpaint` section with request shape, position values, known limitations (dark content + silent output) |
| `scripts/register_outpaint_lora.sh` (new) | Idempotent LoRA fetch + registry insert for cold-start installs |

Companion (ops trees, not in this repo):
- `ltx-sidecar/sidecar.py`: `GenerateRequest` gains `position` / `conditioning_strength` / `skip_stage_2` + "video-outpaint" in the `job_type` Literal; new match case routes to `manager.generate_outpaint(...)`.
- `ltx-sidecar-modal/modal_app.py`: same GenerateRequest + match case additions; `download_weights()` extended to fetch the outpaint LoRA into the HF volume at `/mnt/nvme-1/huggingface/loras/` + symlink to the canonical `ic-lora-outpaint.safetensors` ID.

---

## v1.6.1 — 2026-04-17

### Hot-fix: remote sidecar can't see taco-backend's `uploads/` filesystem

Reported symptom (live): `generate_failed: [Errno 2] No such file or directory: '/mnt/nvme-1/servers/taco-backend/uploads/<uuid>'` on every `a2v` / `i2v` / `retake` dispatched to the Modal remote pool. Text-to-video worked because it has no source-media path fields.

Root cause: `_dispatch_job_turbo_remote` was passing taco-backend's local absolute paths (`audio_path`, `image_path`, `video_path`, `keyframes[].image_path`) straight through to Modal's `/generate`. Modal's container has no mount of the local `uploads/` directory, so `av.open(path)` / `Path(p).read_bytes()` fail with `FileNotFoundError`.

Fix: inline the media as base64 in the request body.

- `ltx_sidecar_client.LtxSidecarClient.generate()` gained `audio_b64`, `image_b64`, `video_b64` kwargs. When set, they go into the JSON payload alongside (or instead of) the corresponding `*_path` fields.
- `_dispatch_job_turbo_remote` in `server.py`: before calling the remote, reads each local media file (`Path(p).read_bytes()`), base64-encodes, and passes as `*_b64` with the path field set to `None`. Keyframe images get the same treatment per-entry. Raises `ValueError("remote_dispatch: media file not found: ...")` if a path doesn't exist (fail fast, not mid-call).
- Modal `/generate` (in `/mnt/nvme-1/servers/ltx-sidecar-modal/modal_app.py`): `GenerateRequest` gains the three `*_b64` fields. On arrival, any present b64 is written to `tempfile.mkstemp(prefix="modal-sidecar-", suffix=".wav|.png|.mp4")`, and the resulting path is passed downstream to the pipeline. Staged files are removed in a `finally` block regardless of outcome.
- Local sidecar (`_dispatch_job_turbo`) path is unchanged — it has direct filesystem access to `uploads/` so it keeps using the `*_path` fields directly.

Payload size impact: base64 expands 4/3. Typical audio (3–10 s): 30–100 KB → 40–135 KB. Reference image: 500 KB–2 MB → 670 KB–2.7 MB. Retake source video (5–30 s at 1080p): 10–100 MB → 13–135 MB. All within reasonable HTTP body limits.

### Files changed

| File | Change |
|------|--------|
| `ltx_sidecar_client.py` | `generate()` accepts `audio_b64` / `image_b64` / `video_b64`; payload includes them when set |
| `server.py` | `_dispatch_job_turbo_remote` reads local media files and converts to base64 before calling remote |

Companion (ops tree, not in this repo): `modal_app.py::GenerateRequest` gained the `*_b64` fields; `/generate` materializes b64 → `/tmp` and cleans up in `finally`.

---

## v1.6 — 2026-04-17

### Remote-sidecar pool with dashboard controls (up to 4 Modal workers)

Evolution of v1.5's single-remote-sidecar addition. The pool now scales 0..N on demand with a dashboard slider, giving a total of **up to 6 concurrent video workers** (2 local — main cuda:0 in-process + local cuda:1 sidecar — plus up to 4 remote Modal containers).

- `config.LTX_REMOTE_SIDECAR_MAX_WORKERS` (default 4) caps the pool. Must not exceed Modal's `max_containers` or requests queue forever.
- Modal app (`/mnt/nvme-1/servers/ltx-sidecar-modal/modal_app.py`) bumped to `max_containers=4` to match.
- `server.py` now manages `_remote_worker_tasks: list[asyncio.Task]` + `_remote_worker_target: int` (persists across turbo toggles). `_scale_remote_pool()` reconciles the live workers to match target IF turbo is active (the pool is turbo-scoped because non-video jobs submitted while turbo is off would otherwise be stolen by remote workers that can only handle video).
- New endpoints:
  - `GET /v1/system/pool` — returns `{turbo_active, remote_sidecar_configured, remote_sidecar_url, remote_worker_target, remote_worker_active, remote_worker_max}`.
  - `POST /v1/system/pool/remote-workers {"count": N}` — sets target. Scales live if turbo is on; else just stores target for next turbo-on.
- Dashboard: new "Remote Pool" row under Controls, rendered with N+1 buttons (0..MAX). Active button is highlighted; status line reflects configured / target / active / turbo-pending states.
- Backward compat: v1.5 clients that relied on "turbo-on auto-spawns 1 remote worker" still get that default — `_remote_worker_target` initializes to 1 when `LTX_REMOTE_SIDECAR_URL` is set, 0 otherwise.

### Fix: Modal /unload no longer breaks the manager

v1.5's pool scale-to-0 path called `ltx_remote_sidecar.unload()` which triggered Modal's `/unload` endpoint. That endpoint ran `manager.evict_all()`, which clears `self._workers` on `SplitModelManager`. Because `@modal.enter` only fires on container boot and not on subsequent requests, any future `/generate` against a still-warm container then failed with `"No LTX workers available — call load_all() first"`.

Fixes:
1. `_scale_remote_pool`'s scale-to-0 path no longer calls `/unload` — Modal's 5-min `scaledown_window` reclaims the GPU authoritatively.
2. Modal's `/unload` endpoint now uses `worker.evict_transformer()` per worker (frees the ~46 GB transformer while keeping the worker registry intact) instead of `manager.evict_all()`.
3. Modal's `/load` endpoint is now self-healing: if `manager.is_ready` is False (post-evict state), it re-runs `manager.load_all()` before returning.
4. Modal's `/generate` adds a defensive inline reload — if a future stale container ever exists, the first request to it triggers `load_all()` instead of 500'ing.

### Turbo toggle no longer /unloads the remote

`_exit_turbo_mode` previously looped over both sidecars and called `/unload` on each. Now it only /unloads the local sidecar (the remote is scale-down-eligible via Modal's native mechanism).

### Verified end-to-end

5 concurrent `POST /v2/text-to-video {model: ltx-2-3-fast, resolution: 1920x1080, duration: 3s}` with pool target=3 + turbo on:
- Dispatch: all 5 entered `processing` within 1 s (2 local + 3 remote warm containers).
- All 5 completed in ~60 s wall clock (local done ~50 s, Modal ~60 s with warm containers).
- 0 failures.

### Files changed

| File | Change |
|------|--------|
| `server.py` | `_remote_worker_tasks`/`_remote_worker_target`; `_scale_remote_pool()`; pool GET/POST endpoints; `_enter_turbo_mode` / `_exit_turbo_mode` now use the pool scaler instead of the single-worker v1.5 path |
| `ltx_sidecar_client.py` | (from v1.5) `auth_token` + `label` kwargs; `ltx_remote_sidecar` module instance (unchanged in v1.6) |
| `config.py` | `LTX_REMOTE_SIDECAR_MAX_WORKERS` (default 4) |
| `dashboard.html` | "Remote Pool" button grid (0..MAX) + `pollPool()` / `updatePoolUI()` / `setRemoteWorkers()` JS, polled every 5 s |
| `CLAUDE.md` / `CHANGELOG.md` | docs + version bump |

Companion (not in this repo): `modal_app.py` gained `max_containers=4`, self-healing `/load`, worker-preserving `/unload`, defensive `/generate` reload.

---

## v1.5 — 2026-04-17

### Turbo-mode hardening: systemctl-stop replaces HTTP /unload for cuda:1 tenants

Root cause (observed in production today): `_enter_turbo_mode` called `await joyai.unload()` and `await ernie.unload()` via HTTP. Those requests can succeed on the wire while the sidecar's Python process keeps tensors resident. The subsequent `ltx_sidecar.load()` then tries to allocate ~46 GB of LTX transformer on a cuda:1 that still has 44 GB of JoyAI resident → CUDA OOM, turbo-enter fails mid-sequence, ACE already stopped, leaving the system in a broken state.

Fix (`server.py`):
- New `_systemctl_unit(unit, action)` helper — runs `systemctl --user <action> <unit>` in a thread, raises `RuntimeError` with stderr on non-zero exit. Replaces per-service `_ace_systemctl` (kept as back-compat alias).
- New `_stop_cuda1_tenants()` — stops `ace-step`, `joyai-sidecar`, `ernie-image-sidecar`, and any stale `ltx-sidecar` via systemctl. Best-effort; "already stopped" is not an error.
- New `_restore_cuda1_tenants()` — inverse: systemctl-start each configured tenant (`LOAD_*=1`). Called at turbo exit AND on turbo-entry abort rollback.
- New `_wait_cuda1_free(threshold_mib=2000, timeout_s=20.0)` — polls `nvidia-smi` until cuda:1 drops below the threshold. Returns False on timeout.
- New `_list_cuda1_processes()` — enumerates compute-app PIDs on the cuda:1 bus for diagnostics.
- `_enter_turbo_mode` rewritten: Flux unload → systemctl-stop all cuda:1 tenants → **wait for cuda:1 to drain (20 s deadline)** → abort with detailed error + tenant restore if not drained → systemctl-start ltx-sidecar → poll /health → /load → spawn workers. No more OOM on turbo entry.
- `_exit_turbo_mode` rewritten: HTTP /unload both sidecars (graceful) → systemctl-stop ltx-sidecar → `_restore_cuda1_tenants()`.

### LTX remote-sidecar pool (3-worker turbo)

Turbo mode previously topped out at 2 concurrent video workers (main cuda:0 in-process + local cuda:1 sidecar). v1.5 adds an OPTIONAL third worker that dispatches to a remote HTTP sidecar — e.g., Modal RTX Pro 6000 for overflow capacity.

- `config.py` — new env vars `LTX_REMOTE_SIDECAR_URL` + `LTX_REMOTE_SIDECAR_TOKEN`. When URL is empty (default), behavior is unchanged from v1.4. When set, turbo enter warms the remote via `/health` then spawns a third `worker_loop` dispatching via `_dispatch_job_turbo_remote`.
- `ltx_sidecar_client.py` — `LtxSidecarClient` gained `auth_token` + `label` kwargs. `_headers()` injects `Authorization: Bearer <token>` when configured. Module now exposes two instances: `ltx_sidecar` (local, label="local") + `ltx_remote_sidecar` (label="remote", or None if not configured).
- `server.py::_dispatch_job_turbo_remote` — routes to the remote client. **Unlike the local path, remote transport failures do NOT auto-exit turbo** — the remote is treated as optional extra capacity; jobs on that worker fail individually but main + local-sidecar workers keep serving.
- `_exit_turbo_mode` also /unloads the remote (saves Modal credit burn if the backing host is scale-to-pay like Modal).

Verified live: 3 concurrent fast t2v submissions → `processing: 3` in queue → 3 parallel workers → all completed in ~40 s wall-clock (vs ~90 s sequential).

### Companion: Modal RTX Pro 6000 LTX sidecar deployment

Scaffolded at `/mnt/nvme-1/servers/ltx-sidecar-modal/modal_app.py` (NOT in the taco-backend repo — ops tree). Includes:
- Custom image: debian_slim + torch cu130 + transformers 5.3.0 (pinned — Gemma3TextConfig attr mismatch in older transformers) + local `/mnt/nvme-1/repos/LTX-2` editable install (user has uncommitted `getattr(rope_local_base_freq)` fallback that upstream master lacks).
- `modal.Volume` at `/mnt/nvme-1/huggingface` pre-populated with 125 GB of LTX-2.3 checkpoints + Gemma 3 12B PT from HF.
- `@app.cls` + `@modal.enter` eager-loads the model per container boot. Cold start: ~60–80 s. Warm: instant.
- FastAPI app with Bearer-token middleware (secret `taco-sidecar-auth`).
- Public URL: `https://tacos8me--taco-ltx-sidecar-ltxsidecar-fastapi-app.modal.run`

Free Modal credit: $30/mo → ~10 hrs of RTX Pro 6000 ($3.03/hr). Scales to zero after 10 min idle (no burn when unused).

### Files changed

| File | Change |
|------|--------|
| `server.py` | `_enter_turbo_mode` + `_exit_turbo_mode` rewritten; new systemctl + cuda:1 drain helpers; `_dispatch_job_turbo_remote`; remote-worker spawn logic |
| `ltx_sidecar_client.py` | `auth_token` + `label` kwargs; module-level `ltx_remote_sidecar` instance |
| `config.py` | `LTX_REMOTE_SIDECAR_URL` + `LTX_REMOTE_SIDECAR_TOKEN` env vars |
| `CLAUDE.md` / `CHANGELOG.md` | version bump + docs |

---

## v1.4.1 — 2026-04-16

### Hot-fix

- **`import subprocess` missing from `server.py` top-level** — `_enter_turbo_mode` (line 1436) invokes `subprocess.run(["systemctl", "--user", "start", "ltx-sidecar"], ...)` inside a lambda, and `_warmup_page_cache` (line 529) invokes `subprocess.run` in an `asyncio.to_thread(...)` call. Both resolve `subprocess` via module scope, where it wasn't imported. Any `POST /v1/system/turbo {enable:true}` hit `NameError: name 'subprocess' is not defined` and left the system in a half-transitioned state (ACE stopped by `_ace_systemctl("stop")` before the lambda executed; LTX sidecar never started; no turbo dual-worker). Latent bug present since turbo mode landed in v1.2 — the page-cache warmup was also silently failing (`asyncio.create_task` swallowed the unhandled exception). The two functions that imported `subprocess` locally (`_ace_systemctl` at ~1078, `_query_gpu_info` at ~1324) worked fine, which masked the broader issue.

Symptom in production today: `POST /v1/system/turbo {"enable":true}` → 500 `turbo_toggle_failed`; queued gens stuck because main worker held `_inference_lock` while `_enter_turbo_mode` crashed mid-handshake.

Fix: one-line `import subprocess` at module top, covering all four call sites. No behavior change for previously-working paths.

**File**: `server.py`.

---

## v1.4 — 2026-04-16

Five semi-independent changesets landed together.

### Full-fidelity history capture (schema v2)

History DB now stores everything needed to re-run a generation exactly.

- **Schema v2 migration** — four new columns on `generations`: `params_json` (raw Pydantic request body with `storage://` URIs preserved), `gen_config_json` (LTX `_gen_config` snapshot at dispatch time, or `{turbo_steps, turbo_guidance}` for Flux-turbo), `seed` (resolved integer — auto-generated if client omits), `enhanced_prompt` (LTX prompt rewrite when `enhance_prompt=true`). Online `ALTER TABLE ADD COLUMN` gated on `PRAGMA user_version`; idempotent; old rows left intact with NULL new columns.
- **New endpoint** `GET /v2/history/{generation_id}` — returns the full record including parsed `params` + `gen_config`. Bearer auth, 404 for not-yours-or-not-found. `/v2/history` list endpoint shape unchanged (backward compat preserved).
- **Path → URI sanitizer** (`_sanitize_params_for_history`) rewrites `image_path` / `audio_path` / `video_path` / `source_audio_path` / `reference_audio_path` / `image_paths` list / `keyframes[].image_path` back to stable `storage://<uuid>` form before persistence.
- **Enhanced prompt plumbing** — `on_prompt_enhanced` callback threaded through `split_model_manager._encode_prompts` → 5 `_run_*` methods → 4 public async wrappers. Dispatcher captures the rewritten text onto `Job.enhanced_prompt`; worker_loop ships it to history.
- **Files**: `history_store.py`, `job_queue.py`, `server.py`, `split_model_manager.py`, `docs/API.md`, `CLAUDE.md`.

### Flux dashboard controls

Dashboard now has a collapsible "Flux" section exposing the turbo sub-parameters that previously required server restart to change.

- `_flux_config` dict (2 tunables: `turbo_steps` default 8, `turbo_guidance` default 2.5), persisted to `.flux_config.json`, survives restart.
- Endpoints: `GET /v1/system/flux-config`, `POST /v1/system/flux-config` (merge-update), `POST /v1/system/flux-config/reset`.
- `flux_manager._generate` / `_img2img` / `_edit` gained `turbo_steps` + `turbo_guidance` kwargs; dispatcher injects from `_flux_config`.
- `gen_config_snapshot` captures the turbo subset when `turbo=true` so history reflects what actually ran.
- **Files**: `server.py`, `flux_manager.py`, `dashboard.html`, `config.py`.

### PR 1 — perf quick-wins

Small, uncontroversial latency savings across every CFG-enabled video gen.

- **P1 negprompt cache** — `DEFAULT_NEGATIVE_PROMPT` encoded once per encoder lifecycle (nulled in `evict_all`); subsequent CFG-path gens skip the 0.4–0.8 s Gemma encode. Lives on encoder device (cuda:0); survives CPU↔GPU paging because the cached tensor is independent of the encoder's parameter tensors.
- **P4 redundant synchronize drop** — removed `torch.cuda.synchronize()` after `text_encoder.encode` in `_encode_prompts`. Same default stream already serializes; the subsequent `.to(target)` syncs implicitly.
- **P5 MP4 tmpfile on tmpfs** — `_video_to_bytes` now writes the PyAV intermediate to `/dev/shm` (verified tmpfs) via new `config.MP4_TMPDIR`. Saves 50–200 ms/job on ext-backed `/tmp` (confirmed by `stat -f /tmp` → ext2/ext3). Fallback to `/tmp` if shm missing.
- **P7 tqdm TTY auto-detect** — `tqdm(range(...), disable=None)` silences the per-step progress bar under systemd (no TTY). Cleans up journalctl, saves ~1 ms × steps.
- **O3 timed encoder** — wrapped `_encode_prompts` in `_timed("encode_prompts")`; acts as proof-of-P1 post-deploy (once root logger is configured to reach journal).
- **Files**: `split_model_manager.py`, `config.py`.

### PR 2 — ops resilience

Four independent defensive changes; O-A is live-verified, the other three activate only in failure modes.

- **O-A cancellation propagation** — `DELETE /v2/jobs/{id}` now actually stops the LTX denoiser. New `GenerationCancelledError` raised from `ProgressDenoiser.__call__` when `job.status == CANCELLED`; the sigma loop unwinds naturally. `worker_loop` distinguishes cancellation from failure (status → `cancelled`, not `failed`; no error recorded). Verified: GPU util 100 % → 0 % within 3 s of DELETE at 11 % progress. Previously the denoiser would have kept burning for ~25 more steps.
- **O-B LTX OOM recovery** — new `_oom_recovery(worker)` context manager + `@_with_oom_recovery` decorator applied to all five `_run_*` methods. On CUDA OOM: evict transformer + `cleanup_memory()`, then re-raise. Mirrors the `flux_manager.py` pattern; prevents the classic failure where a mid-VAE-decode OOM leaks ~22 GB into the allocator cache and OOMs every subsequent request.
- **O-C half-load recovery** — new `SplitModelManager.reset()` nulls workers + encoder_ledger + the neg-prompt cache, then per-GPU sync + `empty_cache()`. `_load_all_impl` wrapped so `_last_load_failed` is set on exception; `_ensure_ltx_resident` calls `reset()` before retry when the flag is up. Prevents blind `load_all()` retries against partially-populated GPU memory.
- **O-D sidecar crash → auto-exit turbo** — `_dispatch_job_turbo` catches `LtxSidecarError` with status 502/503/504 (transport failures) and schedules `_auto_exit_turbo_on_sidecar_failure` via `asyncio.create_task` so the next queued job doesn't fail the same way. Job-level errors (4xx) don't trigger.
- **Files**: `split_model_manager.py`, `server.py`, `job_queue.py`.

### PR 3 — cleanup audit (documentation + 2 safe drops)

Audited the 10 open-coded `gc.collect() + torch.cuda.synchronize() + torch.cuda.empty_cache()` triples in `split_model_manager.py`. Finding:

- **Do not dedup to `cleanup_memory()`**: the helper uses current-device sync, but our multi-GPU paths (`evict_transformer`, `evict_all`, `reset`) require explicit per-device sync (`torch.cuda.synchronize(self.device)` / `torch.cuda.synchronize(torch.device(device_name))`). Blind dedup would silently regress DUAL_GPU_LTX correctness.
- **Two sites are truly redundant**: back-to-back `gc.collect()` after `worker.evict_transformer()` in `_run_retake` (the encode-prep path and the pre-VAE-decode path). `evict_transformer` already calls gc+sync+empty internally; a second gc immediately after picks up nothing. Dropped both.
- **Eight sites are load-bearing**: added multi-line comments at `evict_transformer`, `ensure_transformer`, `_page_encoder_to_cpu`, `evict_all`, and the four identical inter-stage cleanup blocks in `_run_t2v` / `_run_t2v_hq` / `_run_i2v` / `_run_a2v`. Comments explain why the cleanup can't be safely dedup-ed (device-specific sync, multi-GPU loop, flush of just-cleared auxiliary model refs before VAE decode's ~15 GB peak). Future optimizers: don't strip these.

Measured savings: ~40 ms on retake only. The original "100–300 ms/job" estimate over-promised — most sites exist for memory-safety, not ceremony.

### Infra

- `.gitignore`: `.flux_config.json`, `*.bak-pr*`, `cufile.log`.
- Backup files `*.bak-pr1-*` / `*.bak-pr2-*` / `*.bak-pr3-*` on disk for rollback; ignored by git.

### Deferred (not shipped)

Tracked as task IDs in the working-set task list for future PRs:

- **#95** — Gemma `tokenizer.chat_template` missing: blocks `enhance_prompt=true` on all LTX modes. Discovered during v1.4 smoke. Enhance_prompt history column is wired and will populate correctly once the tokenizer is fixed.
- **#96** — Smoke test for remaining gen types (i2v/a2v/retake/i2i/image-edit/ernie) with uploaded source media. Text-to-image + text-to-video + music were smoke-tested; the source-media-requiring types need a manual test pass.
- **#100** PR 4 (feature trivials): G1 client seed on video, G7 retake boundary validation, G8 retake `enhance_prompt` field, G9 batch priority honoring, G10 stage-2 latent preservation on OOM.
- **#101** PR 5 (feature smalls): G2 custom `negative_prompt` on video, G3 HQ guider params aligned to upstream (`stg_scale=0.0, rescale_scale=0.45`), G5 per-request `gen_config` override.
- **#102** PR 6: P2 queue-aware encoder residency — skip `_page_encoder_to_cpu` when the next queued LTX job fits alongside the encoder (fast-mode batches save ~10 s / 5 jobs).
- **P8** (after #95 lands): cache `generate_enhanced_prompt` output keyed by `(prompt_hash, image_hash, seed)`; noodle-i Char loop repays the 1–3 s Gemma generation every iteration otherwise.
- **G4** (after #95 lands): route `enhance_i2v` for image-to-video mode; currently we always call `enhance_t2v` and ignore the reference image.
- **Structural**: pinned-host-memory base-weight cache for `ensure_transformer` — HQ jobs cycling `dev_lora_025 → dev_lora_050` mid-generation reload from disk every swap. Real refactor, own epic.

---

## v1.3 — 2026-04-13

- ERNIE-Image sidecar (8B DiT text-to-image on cuda:1 port 8094, Apache 2.0). Swaps with JoyAI on cuda:1; coexists with ACE.
- Dashboard advanced controls: 14 tunable LTX generation parameters (sampler, steps, scheduler shifts, CFG/STG/rescale/modality scales, stage-2 sigmas, eta). Persisted to `.gen_config.json` via `GET/POST /v1/system/config`.
- CFG++ sampler (euler_ancestral_cfg_pp ported from ComfyUI, adapted to LTX-2 CONST flow-matching) — now default.
- LTX-2.3 v1.1 distilled models.
- SingleGPUModelBuilder + CachingModelFactory replacing ModelLedger.
- BatchSplitAdapter on every transformer call.
- bf16 reduced-precision accumulation restored to PyTorch default (was previously False — caused character movement artifacts).

## v1.2

- Turbo mode: dual-GPU LTX via cuda:1 sidecar for 2 concurrent video workers.
- ACE music sidecar on cuda:1 (18 GB, xl-base + 4B LM via vLLM).
- JoyAI image-edit sidecar migrated from cuda:0 to cuda:1.
- Dashboard + `GET /v1/system/gpu` telemetry endpoint.
- Batch scheduler (1–50 items, priority stored).
- v2 job observability: phases, SSE stream, `/v2/jobs/{id}/preview`.

## v1.1

- Single-GPU swap mode (LTX ↔ Flux auto-swap on cuda:0).
- Keyframe symbolic indices (`"first" | "middle" | "last"` + negative ints).
- Flux LoRA folder-drop discovery (adapter mode, strength changes free).
- Fast-mode audio-to-video.
- History store (SQLite, WAL, thumbnails, 30-day retention).
- Approved images pipeline (noodle-i → noodle-v).

## v1.0

Initial release.
