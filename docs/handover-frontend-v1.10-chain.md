# v1.10.0 — Frontend Chain Conditioning Spec (noodle-v)

> Handover doc. Backend v1.10.0 ships the endpoints below. Frontend team implements the orchestration flow described here.

## Appendix: Frontend Handover Spec (Unit C, ready to send)

[Copy-paste below into a handover doc to noodle-v team.]

**Goal**: Each non-first MusicVideo clip is conditioned on the last 3 safe frames of its predecessor, producing seamless playback in composition export.

**Backend contract (v1.10.0, on origin/master)**:

- `POST /v2/video/extract-frames` — body `{video_uri: "storage://...", frame_indices: [int]}` (1–16 non-negative). Response `{frames: [{frame_index, storage_uri, width, height}]}`. Errors: 404 video_not_found, 422 frame_index_out_of_range, 504 pyav_timeout, 429 upload_quota_exceeded.
- `POST /v2/audio-to-video` now accepts `keyframes: list[KeyframeInput] | None` (mutually exclusive with `image_uri`+`image_strength` in practice). KeyframeInput = `{image_uri, frame_index, strength}` where `strength ∈ [0, 1]`.
- `POST /v2/compositions` passes through per-clip `tailTrimFrames: int` (default 0) and composition-root `chainMode: "seamless" | "hardcut"`.

**Flow** (owning module: `musicVideoStore` + `useMusicVideoOrchestrator` hook):
1. Generate clip 0 with single-keyframe form (or no image).
2. On `completed` event: read `num_frames` from history metadata → `safeTail = [num_frames-6, num_frames-5, num_frames-4]` → `POST /v2/video/extract-frames`.
3. Submit clip 1 with `keyframes=[{image_uri:f[−6], frame_index:0, strength:1.0}, {image_uri:f[−5], frame_index:1, strength:1.0}, {image_uri:f[−4], frame_index:2, strength:1.0}]`.
4. Repeat for remaining clips.
5. On composition save: `tailTrimFrames=3` for clips 0..N-2, `0` for final. `chainMode="seamless"`.

**Error handling**:
- Extract 422 → fall back to single-keyframe (clip N's `f[−4]` URI) for next clip, toast "chain degraded", keep `tailTrimFrames=3`.
- Extract 404 → hard error modal, halt chain.
- A2V 422 on `keyframes` → backend not on v1.10; auto-downgrade to hardcut mode in-memory, warn user.

**Strength default**: `[1.0, 1.0, 1.0]`. Hidden dev flag `chainStrengthTaper` for `[1.0, 0.95, 0.9]` if stutter observed. No user-facing slider in v1.10.0.

**UI**: display effective duration (raw – trim) on clip cards; tooltip shows raw. Composition total = sum of effective. "Chaining clip N of M" progress indicator during extract+submit window.

**Backward compat**: legacy compositions default `tailTrimFrames=0`, `chainMode="hardcut"`. Byte-identical export to v1.9.9.

**Feature flag**: gate entire UI behind `flags.v110_seamless_chain`. Default `seamless` for new comps once flag enabled; `hardcut` for loaded legacy comps.

**Acceptance tests**:
1. 5-clip seamless happy path — asserts 3-keyframe submissions + correct tailTrimFrames on save + export length within 1 frame of expected.
2. Legacy hardcut load+export — byte-identical to v1.9.9 baseline.
3. Extract 422 fallback — chain continues with single-keyframe, `tailTrimFrames` preserved.
4. Extract 404 halt — modal shown, retry re-invokes.
5. Seamless→hardcut toggle mid-authoring — all submissions use single-keyframe form.
