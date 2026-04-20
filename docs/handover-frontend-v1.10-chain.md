# v1.12 — Frontend Chain Conditioning Spec (noodle-v)

> Handover doc. **v1.12 is the current recommended path** and replaces v1.11.5's 3-PNG-keyframes flow with a single video-segment conditioning. v1.11.5 pinned only 1 pixel frame per seam, which drifted cliff-wise at seam 2+. v1.12 pins 9 consecutive pixel frames via a single VAE-encoded multi-frame latent and eliminates subject drift at the seam. See the "v1.12 flow" section below for the FE change. The v1.11.5 keyframes path stays as an unconditionally-supported legacy fallback — sending 3 PNG keyframes still works, it just has the known drift.
>
> Backend version history: v1.10.0 shipped the original multi-keyframe endpoints; v1.10.1 fixed the a2v strength default; v1.11.0→v1.11.2 churned on audio-video timing (beat-gap atrim clamp, eventually decoupled via `audioDurationSec`); v1.11.3→v1.11.5 discovered and reverted a wrong-granularity routing fix after v1.11.3's "slideshow" regression. v1.12 is the architecturally correct chain conditioning.

## v1.12 flow (RECOMMENDED — migrate here)

**What changed**: instead of `POST /v2/video/extract-frames` → 3 PNG URIs → 3 keyframes on the next a2v/i2v, you now call `POST /v2/video/extract-segment` → 1 MP4 URI → pass as `segment_uri` on the next a2v/i2v. Backend VAE-encodes the 9-frame segment as a multi-latent-frame tensor and hard-pins 9 consecutive target pixel frames at every sigma step.

**Endpoint**: `POST /v2/video/extract-segment` (new)

```jsonc
// request
{
  "video_uri":   "storage://<32hex>",
  "start_frame": <int>,            // 0-indexed; FE computes: num_frames - 9
  "num_frames":  9                 // must be 8k+1 ∈ {9, 17, 25, 33}; v1.12 ships with 9
}

// response (200)
{
  "segment_uri": "storage://<32hex>",
  "width":       <int>,
  "height":      <int>,
  "num_frames":  9,
  "fps":         <float>
}
```

Errors (same shape as `/v2/video/extract-frames`):
- `404 video_not_found` — source MP4 not resolvable.
- `422 invalid_num_frames` — not 8k+1 or not in {9, 17, 25, 33}.
- `422 segment_out_of_range` — `start_frame + num_frames` exceeds video length.
- `504 pyav_timeout` — decode/encode took > 30 s.
- `429 upload_quota_exceeded` — `PER_KEY_UPLOAD_BYTES_PER_DAY` cap hit.

**Request field**: `AudioToVideoRequest.segment_uri: string | null` and `ImageToVideoRequest.segment_uri: string | null`. Mutually exclusive with `image_uri` and `keyframes` (3-way exclusion enforced by a Pydantic validator — sending two returns 422).

**FE flow (MusicVideoStore / useMusicVideoOrchestrator)**:

1. Generate clip 0 normally via `image_uri` (user image) or no image.
2. On clip 0 completion: read `num_frames` from history metadata. Compute `start_frame = num_frames - 9`.
3. `POST /v2/video/extract-segment { video_uri: clip0_video, start_frame, num_frames: 9 }` → get `segment_uri`.
4. Submit clip 1 via `POST /v2/audio-to-video` (or `/v2/image-to-video`) with `segment_uri: <from step 3>`. Do NOT set `keyframes` or `image_uri` (validator rejects mixed modes).
5. Repeat for every subsequent clip. Final clip submission looks identical to earlier chained clips (it just has no successor).
6. On composition save: set `tailTrimFrames=9` on every non-final clip (because 9 pixel frames of the prior clip are re-shown as the follower's pinned head). Final clip: `tailTrimFrames=0`. `chainMode="seamless-segment"` at the composition root.

**Composition clip schema addition** (all other fields unchanged from v1.11.2):

```jsonc
{
  "historyId":        "h_abc",
  "sequenceIndex":    0,
  "duration":         2.042,     // LTX 8k+1 raw, unchanged
  "audioStart":       0.0,       // unchanged
  "tailTrimFrames":   9,         // v1.12: 9 on non-final clips (was 6 in v1.11.2)
  "audioDurationSec": 2.0,       // unchanged — decouples audio slice from video effective
  "segmentUri":       "storage://..."  // v1.12 NEW: the segment the FOLLOWER conditions on
}
```

`segmentUri` on a clip represents the segment extracted from THAT clip's tail, consumed by the next clip. Purely informational for audit / re-export; backend doesn't re-extract.

**Root-level composition field**:

```jsonc
{ "chainMode": "seamless-segment" }     // v1.12 flag, distinguishes from legacy "seamless"
```

Valid values:
- `"hardcut"` — no conditioning, full-clip cuts. Unchanged.
- `"seamless"` — v1.11.5 legacy, 3 PNG keyframes path. Still supported.
- `"seamless-segment"` — v1.12 path. Use for all new MusicVideo compositions.

**Clip-length guidance** (FE choice, backend honors either):

- **49-frame LTX clips (2.04 s, beat-gap 2.0 s)** — works. `tailTrimFrames=9` leaves 40 visible pixel frames = 1.667 s per clip. Against 2.0 s beats: 333 ms audio-video drift per seam (125 ms worse than v1.11.2). Acceptable for 2-3 clip comps.
- **97-frame LTX clips (4.04 s, beat-gap 2.0 s, 2 beats/clip)** — RECOMMENDED for 3+ clip comps. `tailTrimFrames=9` leaves 88 visible pixel frames = 3.667 s per clip. Subject identity held cleanly for 3.67 s per clip instead of 1.67 s → dramatically less drift across long chains. Simple composition math: 2 beats of audio per video clip.

**Strength**: hardcoded 1.0 on the backend. No slider.

**Error handling**:
- `extract-segment 422 segment_out_of_range`: halt chain with a user-facing toast "Clip N tail unavailable — regenerate or use a shorter chain". Don't fall back silently — segment continuity is the whole point.
- `extract-segment 429`: surface as "upload quota reached — wait or upgrade tier".
- `a2v/i2v 422` on `segment_uri`: backend not on v1.12 yet. Auto-downgrade in-memory to v1.11.5 keyframes path for that seam, warn user "chain degraded (legacy mode)".

**Feature flag**: gate FE UI behind `flags.v112_seamless_segment`. Default off until backend is verified on prod + FE does visual regression testing. New comps default `chainMode="seamless-segment"` once flag is on; legacy loaded comps keep their existing `chainMode`.

**Acceptance tests**:

1. 5-clip seamless-segment happy path — asserts 4 `extract-segment` calls + `segment_uri` on each non-first submission + `tailTrimFrames=9` on non-final clips + export length within 1 frame of `sum(clip_durations) − 9 * (N-1) / fps`.
2. Legacy "seamless" (keyframes, v1.11.5) — byte-identical export to v1.11.5 baseline.
3. Legacy "hardcut" — byte-identical to v1.9.9 baseline.
4. `extract-segment 422` halt — toast shown, partial composition preserved.
5. Seamless-segment → hardcut toggle mid-authoring — next submission uses single-keyframe form.
6. **Visual seam check** — frame-scrub the exported MP4 at every seam; target clip's pixel frames 0-8 should visually match the prior clip's last 9 frames. No "slideshow" artifact (v1.11.3) or subject cliff (v1.11.5).

**Known limitation**: LTX's causal VAE replicates the segment's frame 0 for padding, so the encoded 9-frame segment's latents aren't bit-identical to what the prior clip's full-length encoding produced for the same frames. Residual RMSE 1-3 on a 0-255 scale at the seam — visually imperceptible in most content. If you scrub and see visible blink at pixel 0, file it — v1.13 latent-reuse path eliminates it entirely.

**Migration from v1.11.5**:

- Keep `flags.v110_seamless_chain` controlling the overall "seamless" UI.
- Add `flags.v112_seamless_segment` on top. When ON:
  - Route to `extract-segment` instead of `extract-frames`.
  - Submit with `segment_uri` instead of `keyframes`.
  - Save with `tailTrimFrames=9` + `chainMode="seamless-segment"`.
- Legacy v1.11.5 comps on disk (with `keyframes` + `tailTrimFrames` ∈ {3, 6}) still load + export correctly. No migration required. Optionally add a "re-chain" button that re-extracts segments and saves in the v1.12 shape.

---

## v1.11.2 legacy path (still supported)

The v1.11.2 section below documents the prior 3-PNG keyframes flow with `tailTrimFrames` + `audioDurationSec`. It's fully supported by the backend — sending 3 keyframes + tailTrimFrames=3/6 still works as before. **New compositions should use the v1.12 flow above.**

> Handover doc (historical). Backend v1.10.0 shipped the endpoints; v1.10.1 fixed the a2v strength default; v1.11.0 briefly recommended `tailTrimFrames=6` to eliminate a tiny visual stutter but traded it for 208 ms of audible per-seam audio dropout; v1.11.1 reverted to `tailTrimFrames=3` as the minimum-total-perception config while the backend remained clamp-based. **v1.11.2 adds the FE-designed fix: a new optional per-clip `audioDurationSec` field that decouples the audio atrim from the video `effective_duration`** — enabling `tailTrimFrames=6` (0 ms visual seam) AND full-song audio continuity simultaneously. The two fields serve independent concerns and do not alias each other. FE now runs on the v1.11.2 path; the v1.11.1 path stays as an unconditionally-supported legacy fallback.

## Goal

Each non-first MusicVideo clip is conditioned on the last 3 safe frames of its predecessor, producing seamless playback in composition export — with full song continuity across every seam.

## Timeline math — the two paths

LTX outputs `N` frames per clip (49, 97, 121, or 153 — always `8k+1`). The last 3 frames (`N-3..N-1`) can have Stage-2 sigma-schedule artifacts. The 3 frames before them (`N-6..N-4`) are the "safe tail" used as chain-conditioning keyframes for the follower clip.

At 49 frames / 24 fps / 2.0 s beat gap, the backend supports two export modes depending on whether the frontend sends the new `audioDurationSec` field:

| Mode | `tailTrimFrames` | `audioDurationSec` | audio per seam | visible video seam | drift per seam |
|---|---|---|---|---|---|
| Hardcut (legacy) | 0 | — | 0 ms | hard cut (content discontinuity) | 0 |
| Chain, clamp-audio (v1.11.1 legacy) | **3** | — (omit) | 83 ms dropout | 83 ms backwards jump | 0 |
| **Chain, decoupled-audio (v1.11.2, FE-preferred)** | **6** | **beat_gap** (e.g. `2.0`) | 0 ms — full song | 0 ms (clean) | video cut `audioDurationSec - effective_duration` before beat |
| ≥ 4 without `audioDurationSec` | intermediate | — | 125–167 ms | intermediate | 0 |

### Why `audioDurationSec` is the right fix

The v1.9.6 backend ships an audio-side atrim slice that defaulted to `min(beat_gap, effective_duration)`. When chain mode trimmed the video (tail > 0), the clamp pulled audio down too, silently dropping song material at every seam. v1.11.0's recommendation to bump tail to 6 exposed this clamp: 208 ms per seam became audible.

The clamp was load-bearing ONLY because audio-side slice duration had no independent field to follow. v1.11.2 adds `audioDurationSec` per clip. When present, the exporter uses it verbatim for the atrim; the clamp no longer applies. Audio follows beat gaps; video follows its own trim math. Independent concerns, independent fields.

### What the decoupled-audio trade-off actually is

Per chained clip at tail=6 + audioDurationSec=2.0:
- Audio slice = 2.0 s (full beat gap, no song drop)
- Video shown = 43 / 24 = 1.792 s (chain-clean seam, no stutter)
- **Audio leads video by 208 ms per clip**

Over 5 clips (4 seams) the accumulated audio-lead is ~832 ms. The compiled MP4 will play:
- Song continuously from beat 0 → beat 5 with no dropouts or skips
- Video cuts progressively earlier than the beats: seam 1 at 1.792 s (beat at 2.0), seam 2 at 3.584 s (beat at 4.0), etc.
- At the end, the last video frame freezes for ~832 ms while the song tail plays out (backend omits `-shortest` in this mode)

Viewer experience: song is musically clean throughout. Cuts feel slightly rushed but still clearly beat-adjacent (each cut is within one beat's worth of its target). Final moment: the last composed frame (clip N's natural ending frame, no Stage-2 decay because `tailTrimFrames` is always 0 on the final clip) holds on screen for under a second while the music completes. This is the preferred failure mode — song integrity is higher-value than exact cut-on-beat.

### Timeline diagram (tail=6 + audioDurationSec=2.0)

```
song     : |----beat0----|----beat1----|----beat2----|----beat3----|
clip 0 v : [0..42]                                                    (1.792 s shown)
clip 1 v :          [regen-43, regen-44, regen-45, 46', …]            (1.792 s shown)
clip 2 v :                         [regen-43, regen-44, …]            (1.792 s shown)
output t : 0    1.79  2.0    3.58  4.0   5.38  6.0
                ^cut       ^cut        ^cut
                beat aligned audio  |  video cut 208 ms early
```

Viewer: `… 44, 45, 46 | 43', 44', 45', 46', …` is replaced by a clean monotonic sequence because `tailTrimFrames=6` drops both the safe tail (regenerated by the follower) and the unsafe tail (Stage-2 artifact zone). No repeat, no backwards jump. Audio plays uninterrupted.

## Backend contract (v1.11.2, on origin/master)

- `POST /v2/video/extract-frames` — body `{video_uri: "storage://...", frame_indices: [int]}` (1–16 non-negative). Response `{frames: [{frame_index, storage_uri, width, height}]}`. Errors: 404 video_not_found, 422 frame_index_out_of_range, 504 pyav_timeout, 429 upload_quota_exceeded.
- `POST /v2/audio-to-video` accepts `keyframes: list[KeyframeInput] | None` (mutually exclusive with `image_uri`+`image_strength`). `KeyframeInput = {image_uri, frame_index, strength: [0,1]}`. v1.10.1 note: the default `image_strength` is `1.0`.
- `POST /v2/compositions` passes through per-clip `tailTrimFrames: int` (default 0), **`audioDurationSec: float | null` (v1.11.2, default absent)**, and composition-root `chainMode: "seamless" | "hardcut"`.

## Flow (owning module: `musicVideoStore` + `useMusicVideoOrchestrator` hook)

1. Generate clip 0 with single-keyframe form (or no image).
2. On `completed` event: read `num_frames` from history metadata → `safeTail = [num_frames-6, num_frames-5, num_frames-4]` → `POST /v2/video/extract-frames`.
3. Submit clip 1 with `keyframes=[{image_uri: f[-6], frame_index: 0, strength: 1.0}, {image_uri: f[-5], frame_index: 1, strength: 1.0}, {image_uri: f[-4], frame_index: 2, strength: 1.0}]`.
4. Repeat for remaining clips.
5. On composition save (v1.11.2 FE-preferred path):
   - **`tailTrimFrames=6`** for clips 0..N-2, **`0`** for the final clip.
   - **`audioDurationSec = next.beatTime - this.beatTime`** for clips 0..N-2 (= the beat gap, typically `2.0` s).
   - **`audioDurationSec = clip.duration`** for the final clip (no next beat; matches no-trim behavior).
   - `chainMode="seamless"`.
6. (Legacy fallback for FE implementations not yet emitting `audioDurationSec`): omit the field entirely and use `tailTrimFrames=3` for clips 0..N-2, `0` for the final. Backend clamps audio to effective_duration as in v1.11.1.

## Error handling

- Extract 422 → fall back to single-keyframe (clip N's `f[-4]` URI) for next clip, toast "chain degraded", keep `tailTrimFrames=6` + `audioDurationSec` as normal.
- Extract 404 → hard error modal, halt chain.
- A2V 422 on `keyframes` → backend not on v1.10+; auto-downgrade to hardcut mode in-memory, warn user.

## Strength default

`[1.0, 1.0, 1.0]` across the 3 head keyframes. LTX strength-1.0 pins the conditioned latent at every sigma step. Hidden dev flag `chainStrengthTaper=[1.0, 0.95, 0.9]` available if residual drift appears; probably not needed.

## UI

- Display **video-effective** duration (raw − trim) on clip cards; tooltip shows raw and audio-slice duration. Composition video total = sum of effective durations; audio total = sum of `audioDurationSec`. The two can legitimately differ by up to one beat gap in total.
- "Chaining clip N of M" progress indicator during the extract+submit window between clips.

## Backward compat

- Legacy compositions default `tailTrimFrames=0`, `audioDurationSec` absent, `chainMode="hardcut"`. Byte-identical export to v1.9.9.
- Compositions saved under v1.11.1 (tail=3, no `audioDurationSec`) still export with v1.11.1 clamp behavior. Re-save to migrate to the v1.11.2 decoupled path.
- Compositions saved under v1.11.0 (tail=6, no `audioDurationSec`) still export; audio will clamp to 1.79 s per seam (the bug v1.11.2 exists to fix). FE should add a one-shot migration that bumps tail=6 comps with no `audioDurationSec` to include `audioDurationSec = beatGap` on load.
- No endpoint shape changes. `audioDurationSec` is a purely additive optional clip field.

## Feature flag

Gate entire UI behind `flags.v110_seamless_chain`. Default `chainMode="seamless"` for new comps once flag enabled; `"hardcut"` for loaded legacy comps.

## Acceptance tests

1. 5-clip seamless happy path (v1.11.2 path) — asserts 3-keyframe submissions + `tailTrimFrames=6` + `audioDurationSec` present on every clip + export video length within 1 frame of `sum(effective_durations)` and audio length within 1 frame of `sum(audioDurationSec)`.
2. Legacy hardcut load+export — byte-identical to v1.9.9 baseline.
3. Extract 422 fallback — chain continues with single-keyframe, `tailTrimFrames` + `audioDurationSec` preserved.
4. Extract 404 halt — modal shown, retry re-invokes.
5. Seamless→hardcut toggle mid-authoring — all submissions use single-keyframe form.
6. **Song-continuity regression check** — export a 5-clip seamless comp, confirm no audible seam in the audio (spectrogram continuous, no drop). Frame-scrub the video at every seam; no frame from clip N should visually match a frame in the first ~3 frames of clip N+1.
7. **Tail-end freeze check** — the final composed frame holds on screen for `sum(audioDurationSec) - sum(effective_durations)` ms while the song completes. Confirm no truncation, no visual glitch at the freeze boundary.

## Migration from the v1.11.1 spec

If you already implemented with `tailTrimFrames=3`, no change is strictly required — the v1.11.1 clamp path still works unchanged. To adopt the v1.11.2 decoupled path:

- On save, switch to `tailTrimFrames=6` for clips 0..N-2 AND add `audioDurationSec = next.beatTime - this.beatTime` (typically `2.0`). Final clip: `tailTrimFrames=0`, `audioDurationSec = clip.duration`.
- Add a one-shot load migration: any saved comp with `tailTrimFrames=3` and no `audioDurationSec` can stay as-is, or be rewritten to the v1.11.2 shape on user re-save.
- Any comp with `tailTrimFrames=6` and no `audioDurationSec` is the v1.11.0 bug shape — either bump to `tailTrimFrames=3` OR attach `audioDurationSec = 2.0` on load; the latter is the preferred migration.
