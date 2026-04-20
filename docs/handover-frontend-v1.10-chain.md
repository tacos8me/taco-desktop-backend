# v1.11.0 — Frontend Chain Conditioning Spec (noodle-v)

> Handover doc. Backend v1.10.0 ships the endpoints; v1.10.1 fixed the a2v strength default; v1.11.0 clarifies the `tailTrimFrames` math to eliminate the backwards-jump stutter at every seam. Frontend team implements the orchestration flow described here.

## Goal

Each non-first MusicVideo clip is conditioned on the last 3 safe frames of its predecessor, producing seamless playback in composition export.

## Timeline math (the important part — got this wrong in the v1.10.0 spec)

LTX outputs `N` frames per clip (49, 97, 121, or 153 — always `8k+1`). The last ~3 frames (`N-3..N-1`) have Stage-2 sigma-schedule artifacts and must not be shown. The next 3 frames in from there (`N-6..N-4`) are the "safe tail" — clean enough to use as chain-conditioning keyframes for the follower clip.

For a 49-frame clip:
- **Unsafe tail**: frames 46, 47, 48 (Stage-2 artifact zone — always drop)
- **Safe tail**: frames 43, 44, 45 (stable content — send as keyframes to clip N+1)
- **Shown**: frames 0..42 (43 frames = 1.79 s @ 24 fps)
- **`tailTrimFrames = 6`** (safe_tail_count + unsafe_tail_count = 3 + 3)

Timeline:

```
Clip N  file  : [0, 1, …, 42, 43, 44, 45, 46, 47, 48]   ← full 49-frame LTX output
Clip N  shown : [0, 1, …, 42]                            ← tailTrimFrames=6 drops last 6
                                 ↓ (frames 43,44,45 sent as keyframes to N+1)
Clip N+1 file : [regen-43, regen-44, regen-45, 46', 47', …, (N+1 total frames)]
Clip N+1 shown: [regen-43, regen-44, regen-45, 46', 47', …, 42']   ← also tailTrimFrames=6
                                                           ↓
Clip N+2 file : [regen-43', …]
```

Viewer sees a monotonic timeline: `… 41, 42 | 43', 44', 45', 46', …` — no backwards jump, no frame repeat.

**Older (WRONG) value** `tailTrimFrames=3` caused clip N to show frames 0..45 AND clip N+1 to start with regenerated frames 43,44,45 — a 2-frame backwards jump + 3-frame content repeat at every seam. If your composition export feels like a "barely noticeable stutter" every few seconds, this is why.

## Backend contract (v1.11.0, on origin/master)

- `POST /v2/video/extract-frames` — body `{video_uri: "storage://...", frame_indices: [int]}` (1–16 non-negative). Response `{frames: [{frame_index, storage_uri, width, height}]}`. Errors: 404 video_not_found, 422 frame_index_out_of_range, 504 pyav_timeout, 429 upload_quota_exceeded.
- `POST /v2/audio-to-video` accepts `keyframes: list[KeyframeInput] | None` (mutually exclusive with `image_uri`+`image_strength`). `KeyframeInput = {image_uri, frame_index, strength: [0,1]}`. v1.10.1 note: the default `image_strength` is now `1.0` (was effectively `1.0` in every prior version, briefly regressed to `0.85` in v1.10.0).
- `POST /v2/compositions` passes through per-clip `tailTrimFrames: int` (default 0) and composition-root `chainMode: "seamless" | "hardcut"`.

## Flow (owning module: `musicVideoStore` + `useMusicVideoOrchestrator` hook)

1. Generate clip 0 with single-keyframe form (or no image).
2. On `completed` event: read `num_frames` from history metadata → `safeTail = [num_frames-6, num_frames-5, num_frames-4]` → `POST /v2/video/extract-frames`.
3. Submit clip 1 with `keyframes=[{image_uri: f[-6], frame_index: 0, strength: 1.0}, {image_uri: f[-5], frame_index: 1, strength: 1.0}, {image_uri: f[-4], frame_index: 2, strength: 1.0}]`.
4. Repeat for remaining clips.
5. On composition save: **`tailTrimFrames=6`** for clips 0..N-2 (= `safe_tail (3) + unsafe_tail (3)`), **`0`** for the final clip. `chainMode="seamless"`.

## Error handling

- Extract 422 → fall back to single-keyframe (clip N's `f[-4]` URI) for next clip, toast "chain degraded", keep `tailTrimFrames=6`.
- Extract 404 → hard error modal, halt chain.
- A2V 422 on `keyframes` → backend not on v1.10+; auto-downgrade to hardcut mode in-memory, warn user.

## Strength default

`[1.0, 1.0, 1.0]` across the 3 head keyframes. LTX strength-1.0 pins the conditioned latent at every sigma step. Hidden dev flag `chainStrengthTaper=[1.0, 0.95, 0.9]` available if residual drift appears after tailTrimFrames is correct; probably not needed.

## UI

- Display **effective** duration (raw − trim) on clip cards; tooltip shows raw. Composition total = sum of effective durations.
- "Chaining clip N of M" progress indicator during the extract+submit window between clips.

## Backward compat

- Legacy compositions default `tailTrimFrames=0`, `chainMode="hardcut"`. Byte-identical export to v1.9.9.
- No other endpoint shape changes.

## Feature flag

Gate entire UI behind `flags.v110_seamless_chain`. Default `chainMode="seamless"` for new comps once flag enabled; `"hardcut"` for loaded legacy comps.

## Acceptance tests

1. 5-clip seamless happy path — asserts 3-keyframe submissions + `tailTrimFrames=6` on save + export length within 1 frame of `sum(clip_durations) − 6 * (N-1) / fps`.
2. Legacy hardcut load+export — byte-identical to v1.9.9 baseline.
3. Extract 422 fallback — chain continues with single-keyframe, `tailTrimFrames` preserved at 6.
4. Extract 404 halt — modal shown, retry re-invokes.
5. Seamless→hardcut toggle mid-authoring — all submissions use single-keyframe form.
6. **Timeline-repeat regression check** — frame-scrub the exported MP4 at every seam; no frame from clip N should visually match a frame in the first ~3 frames of clip N+1.

## Migration from the v1.10.0 spec

If you already implemented with `tailTrimFrames=3`: change it to `6`. No code path or data-shape changes needed. Pre-existing saved compositions with `tailTrimFrames=3` will still export (they'll just keep the old 2-frame-jump behavior); re-save them to apply the new value, or add a one-shot migration that bumps 3 → 6 on load.
