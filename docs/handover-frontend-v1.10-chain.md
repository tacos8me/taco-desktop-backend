# v1.11.1 — Frontend Chain Conditioning Spec (noodle-v)

> Handover doc. Backend v1.10.0 ships the endpoints; v1.10.1 fixed the a2v strength default; v1.11.0 attempted to clarify `tailTrimFrames` math but traded a tiny video stutter for an audible audio clip at each seam. **v1.11.1 reverts the recommendation to `tailTrimFrames=3`** — the v1.10.0 value — because the audio clamp + beat-gap atrim math make that the only value where both audio dropout AND visual stutter are below perceptual threshold. Frontend team implements the orchestration flow described here.

## Goal

Each non-first MusicVideo clip is conditioned on the last 3 safe frames of its predecessor, producing seamless playback in composition export.

## Timeline math — `tailTrimFrames=3` is the right value (read the trade-off table below)

LTX outputs `N` frames per clip (49, 97, 121, or 153 — always `8k+1`). The last ~3 frames (`N-3..N-1`) can have Stage-2 sigma-schedule artifacts. The next 3 frames in (`N-6..N-4`) are the "safe tail" used as chain-conditioning keyframes for the follower clip.

For a 49-frame clip at 24 fps with a typical 2.0 s beat gap, here are the four candidate `tailTrimFrames` values and their measured trade-offs:

| `tailTrimFrames` | effective video / clip | audio dropout per seam | visible video seam | verdict |
|---|---|---|---|---|
| 0 (legacy, no chain) | 2.04 s | 0 ms | hard cut (color/motion discontinuity) | not chain-compatible |
| **3** (v1.10.0, v1.11.1 current) | 1.92 s | **83 ms** (below threshold) | 2-frame backwards jump (83 ms, below threshold) | **best trade** — both below perceptual threshold |
| 6 (v1.11.0 briefly) | 1.79 s | **208 ms** (audible) | smooth (0 ms) | audio clips audibly on every seam |
| ≥ 4 | intermediate | 125–167 ms | smaller-than-tail=3 jump or "hold" | worse audio than tail=3, marginal video gain |

### Why `tailTrimFrames=6` sounded audibly clipped

With `tailTrimFrames=6`:
- Effective clip duration = 43 frames / 24 fps = **1.7917 s**
- Beat gap = 2.0 s (song beats at 0, 2, 4, 6, 8 s)
- Backend's v1.9.6 beat-gap atrim clamps `slice = min(gap, effective) = 1.7917 s`
- **Song material between 1.79 s and 2.0 s (208 ms) is never played** at every seam
- For a 5-clip MusicVideo that's 4 × 208 ms = **832 ms of song silently dropped**

The reason the clamp exists is to keep visual cuts beat-aligned. Removing the clamp (letting audio play the full 2.0 s gap) drifts the visual cut progressively earlier than the beat — by seam 4 the cut is 0.83 s BEFORE the beat, which sounds MUCH worse than a 208 ms silent gap.

### Why `tailTrimFrames=3` is the sweet spot

With `tailTrimFrames=3`:
- Effective clip duration = 46 frames / 24 fps = **1.9167 s**
- Audio slice = 1.9167 s (per-seam song drop = **83 ms**, below most listeners' perceptual threshold for music)
- Clip N shows frames 0..45; clip N+1 regenerates frames 43, 44, 45 at its head
- Viewer sees `… 44, 45, 43', 44', 45', 46', …` — a 2-frame backwards jump + 3-frame content repeat
- Total visual stutter at the seam = 83 ms (= 2 frames @ 24 fps), also below most viewers' perceptual threshold

Both audio and video seam artifacts sit at ~83 ms. Users described this as "barely noticeable stutter". It's the closest you can get to seamless given the LTX `8k+1` quantization + 2.0 s beat gap + chain mechanics. The fundamental math cannot produce a perfectly seamless result at 49 frames / 24 fps / 2.0 s beats — one axis must take the hit; 83 ms split across both is the minimum-total-perception configuration.

### Timeline diagram (tailTrimFrames=3)

```
Clip N  file  : [0, 1, …, 45, 46, 47, 48]             ← full 49-frame LTX output
Clip N  shown : [0, 1, …, 45]                          ← tailTrimFrames=3 drops unsafe tail only
                               ↓ (frames 43,44,45 sent as keyframes to N+1)
Clip N+1 file : [regen-43, regen-44, regen-45, 46', 47', …, 48']
Clip N+1 shown: [regen-43, regen-44, regen-45, …, 45']   ← also tailTrimFrames=3
```

Viewer: `… 44, 45 | 43', 44', 45', 46', …` — 2-frame "shimmer" at the seam as frames 43–45 play twice (once from clip N, once regen'd from clip N+1). Imperceptible on a generating-subject music video.

## Backend contract (v1.11.0, on origin/master)

- `POST /v2/video/extract-frames` — body `{video_uri: "storage://...", frame_indices: [int]}` (1–16 non-negative). Response `{frames: [{frame_index, storage_uri, width, height}]}`. Errors: 404 video_not_found, 422 frame_index_out_of_range, 504 pyav_timeout, 429 upload_quota_exceeded.
- `POST /v2/audio-to-video` accepts `keyframes: list[KeyframeInput] | None` (mutually exclusive with `image_uri`+`image_strength`). `KeyframeInput = {image_uri, frame_index, strength: [0,1]}`. v1.10.1 note: the default `image_strength` is now `1.0` (was effectively `1.0` in every prior version, briefly regressed to `0.85` in v1.10.0).
- `POST /v2/compositions` passes through per-clip `tailTrimFrames: int` (default 0) and composition-root `chainMode: "seamless" | "hardcut"`.

## Flow (owning module: `musicVideoStore` + `useMusicVideoOrchestrator` hook)

1. Generate clip 0 with single-keyframe form (or no image).
2. On `completed` event: read `num_frames` from history metadata → `safeTail = [num_frames-6, num_frames-5, num_frames-4]` → `POST /v2/video/extract-frames`.
3. Submit clip 1 with `keyframes=[{image_uri: f[-6], frame_index: 0, strength: 1.0}, {image_uri: f[-5], frame_index: 1, strength: 1.0}, {image_uri: f[-4], frame_index: 2, strength: 1.0}]`.
4. Repeat for remaining clips.
5. On composition save: **`tailTrimFrames=3`** for clips 0..N-2 (drops unsafe tail only; safe tail plays and is regenerated by the follower), **`0`** for the final clip. `chainMode="seamless"`.

## Error handling

- Extract 422 → fall back to single-keyframe (clip N's `f[-4]` URI) for next clip, toast "chain degraded", keep `tailTrimFrames=3`.
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

1. 5-clip seamless happy path — asserts 3-keyframe submissions + `tailTrimFrames=3` on save + export length within 1 frame of `sum(clip_durations) − 3 * (N-1) / fps`.
2. Legacy hardcut load+export — byte-identical to v1.9.9 baseline.
3. Extract 422 fallback — chain continues with single-keyframe, `tailTrimFrames` preserved at 3.
4. Extract 404 halt — modal shown, retry re-invokes.
5. Seamless→hardcut toggle mid-authoring — all submissions use single-keyframe form.
6. **Timeline-repeat regression check** — frame-scrub the exported MP4 at every seam; no frame from clip N should visually match a frame in the first ~3 frames of clip N+1.

## Migration from the v1.11.0 spec

If you implemented against v1.11.0 with `tailTrimFrames=6`: revert to `3`. No code path or data-shape changes needed. Pre-existing saved compositions with `tailTrimFrames=6` will still export (they'll just exhibit the 208 ms per-seam audio dropout); re-save them to apply the new value, or add a one-shot migration that bumps 6 → 3 on load. The v1.10.0 value of `3` is canonical again.
