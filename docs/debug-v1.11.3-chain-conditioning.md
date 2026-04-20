# v1.11.3 — Chain conditioning root-cause: only frame 0 was being pinned

> Debugging record. Preserved for future archaeology — this bug masqueraded as an audio/video-timing problem for three releases (v1.10.0 → v1.11.2) before we found the real cause.

## Symptoms (user report)

- Clip 0 → clip 1 seam: excellent, seamless continuation.
- Clip 1 → clip 2 seam: good.
- Clip 2 → clip 3 seam: **step change — "entirely different scene"** on clip 3.
- Original user keyframe "vaguely resembles" the output by clip 3+ (subject drift).
- The drift was not gradual; it was a cliff at seam 2.

## False trails (chronological)

| Release | Hypothesis | Fix shipped | Actual effect |
|---|---|---|---|
| v1.9.6 | Beat-gap atrim slicing using quantized clip_duration overlapped seams | Use `beatTime[N+1] - beatTime[N]` for audio slice | Correct fix for its own problem, but didn't address the chain issue |
| v1.9.8 | PTS discontinuity at seams | Prepend `setpts=PTS-STARTPTS,format=yuv420p` per input | Made it worse (removed the only scene-change hint libopenh264 had) |
| v1.9.9 | libopenh264 cross-clip P-frame prediction bled adjacent clips | `-force_key_frames` at seam cumsum | Correctly fixed encoder-side bleeding |
| v1.10.0 | Content-level discontinuity between independently-generated clips | Multi-frame chain conditioning: extract 3 tail frames, pass as keyframes | **Broken by default** — only 1 of 3 keyframes actually pinned |
| v1.10.1 | a2v keyframe strength silently 0.85 | Default `image_strength=1.0` | Fixed a real regression but didn't surface the deeper bug |
| v1.11.0 | Visual stutter from tail=3 chain math | Recommend `tailTrimFrames=6` | Traded sub-threshold stutter for audible audio dropout |
| v1.11.1 | v1.11.0 audio clipping | Revert to `tailTrimFrames=3` | Returned to both-below-threshold 83/83 ms split |
| v1.11.2 | Audio/video slice should be independent | FE-designed `audioDurationSec` decoupling | Correct fix for its own problem — but the chain was still drifting |

Each fix was correct for the symptom it addressed. The subject drift user reported across v1.10.0–v1.11.2 was orthogonal to all of them.

## Root cause

In `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`, `combined_image_conditionings` (lines 127–161) **branches on `frame_idx == 0`** and uses two semantically different conditioning mechanisms:

### Frame 0 — `VideoConditionByLatentIndex` (hard pin)
`ltx-core/src/ltx_core/conditioning/types/latent_cond.py:40-42`:
```python
latent_state.latent[:, start_token:stop_token] = tokens          # REPLACES main video tokens
latent_state.clean_latent[:, start_token:stop_token] = tokens
latent_state.denoise_mask[:, start_token:stop_token] = 1.0 - self.strength
```
At `strength=1.0`, the denoise mask on those tokens is 0 — the main video's latent at frame position 0 IS the input image, pinned at every sigma step. Output frame 0 is pixel-exact (modulo VAE encode/decode roundtrip).

### Frames 1+ — `VideoConditionByKeyframeIndex` (soft guide)
`ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py:64-70`:
```python
return LatentState(
    latent=torch.cat([latent_state.latent, tokens], dim=1),       # APPENDS as extra context tokens
    denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
    positions=torch.cat([latent_state.positions, positions], dim=2),
    clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
    attention_mask=new_attention_mask,
)
```
The main video's latent at frame positions 1, 2 is **still noise that gets denoised from scratch**. The keyframe tokens are auxiliary context tokens with positional encoding at `frame_idx/fps`; the transformer can attend to them but is not forced to produce matching output. At `strength=1.0` the keyframe tokens themselves stay clean (so the model sees the reference), but output frames 1 and 2 are free-denoised with soft attention guidance only.

This is the classical LTX sparse-keyframes semantic (first/middle/last at indices [0, 24, 48]): frame 0 is the scene anchor, frames 24 and 48 are motion hints. For that use case the semantic is correct.

For our **chain conditioning** use case — three consecutive head frames at [0, 1, 2] meant to be exact reproductions of the prior clip's tail — it's the wrong helper.

## Why the symptoms match exactly

- **"Vaguely resembles our input"**: only clip 0's frame 0 ever sees the user image. Clip 1's frame 0 is clip 0's frame 43 (43 frames of free generation past the user anchor). Clip 2's frame 0 is clip 1's frame 43 (86 frames of drift). Clip 3's frame 0 is 129 frames of drift. Frames 1–2 never hold the anchor because they're soft-guided only.
- **Step change at seam 2**: clip 1's frames 43–45 were still naturally coherent (clip 0 was a clean initial generation). Clip 2's frames 43–45 are 2 generations deep — the soft-guiding of clip 2's frames 1, 2 can't hold the subject because its anchor (clip 1's frame 43) is already drifted. The model commits to a "new interpretation" of the ambiguous latent at frames 1, 2, and clip 3 inherits that.
- **Cliff-not-drift**: soft-guiding gets progressively weaker relative to the main video's free denoising as the anchor drifts. There's a threshold where the guide loses to the prompt + audio + noise — and that threshold was crossed at seam 2 for this user's content.

## The fix

`ltx-pipelines/src/ltx_pipelines/utils/helpers.py:164-191` already has the right helper: `image_conditionings_by_replacing_latent` uses `VideoConditionByLatentIndex` for **every** frame with `latent_idx=img.frame_idx`. Each keyframe directly replaces the main video's latent at its frame position — hard pin at every sigma step, for all three.

In taco-backend `split_model_manager.py`, the a2v and i2v pipelines call `combined_image_conditionings`. The fix detects chain pattern (consecutive `frame_idx` starting at 0) and routes those calls through `image_conditionings_by_replacing_latent`. Sparse-keyframe patterns ([0, 24, 48] first/middle/last) are unchanged — classical LTX semantics preserved.

Detection: `indices == list(range(len(indices)))` — simple, strict, no false positives on the legacy sparse-keyframe use case.

Scope:
- `_run_i2v` stage 1 + stage 2 (2 call sites)
- `_run_a2v` stage 1 + stage 2 (2 call sites)
- t2v and retake paths are untouched (t2v passes `images=[]`, retake uses single-latent conditioning for video input).

## Verification plan

- **120/120 pytest green** (no test covers denoising behavior directly; these tests catch shape/signature regressions only).
- Manual 3-clip chain test on noodle-v after deploy. Before v1.11.3: clip 3 drifts. After v1.11.3: all 3 clips visually coherent, subject preserved across all seams.
- If subject still drifts at clip 4+, that's a separate problem — each clip's frame 0 is still "clip N-1's frame 43", which is 43 frames of free generation from the previous anchor. Long chains may need periodic re-grounding with the user's original image. That's a FE concern, not a backend one.

## Follow-ups (not in v1.11.3)

1. **Long-chain grounding**: pass the user's original image as a 4th keyframe (sparse, mid-video) every N clips to re-anchor the subject. FE decision.
2. **Strength taper for motion**: pinning 3 consecutive frames at strength=1.0 may over-constrain motion trajectory. If motion looks "stuck" in early frames, taper to `[1.0, 0.95, 0.9]`. Hidden FE flag already documented in the handover doc.
3. **Upstream patch to `combined_image_conditionings`**: consider contributing a `mode="pin" | "guide"` parameter upstream so the branching isn't implicit. Not blocking.

## Addendum (v1.11.5): the v1.11.3 fix was wrong, reverted

Post-deploy testing of v1.11.3 + v1.11.4 on a 97-pixel-frame a2v request with keyframes at `[0, 1, 2]` at strength 1.0 showed pixel-frame RMSE against the 3 provided PNGs:

| Pixel frame | vs kf0 | vs kf1 | vs kf2 | Best match |
|---|---|---|---|---|
| 0 | **2.42** | 21.87 | 26.06 | kf0 (clean pin) |
| 1 | 13.62 | 10.37 | 18.53 | kf1 (bleeding) |
| 2 | 21.07 | **4.47** | 16.80 | kf1 (held) |
| 4 | 27.62 | 6.49 | 18.63 | kf1 (held) |
| 7 | 30.30 | 9.09 | 19.53 | kf1 (held) |
| 8 | 30.17 | 9.24 | 17.74 | kf1 (held) |
| 15 | 31.90 | 24.36 | **8.98** | kf2 (held) |
| 16 | 32.04 | 24.57 | 9.27 | kf2 (held) |

The v1.11.3 routing did exactly what I coded it to do — pin `latent_idx=[0, 1, 2]` via `VideoConditionByLatentIndex`. But **LTX's `VideoConditionByLatentIndex` takes a LATENT-frame index, not a pixel-frame index.** LTX's causal VAE maps latent frames to pixel-frame spans via the 8k+1 scheme:

- Latent 0 → pixel 0 (exactly 1 pixel frame)
- Latent 1 → pixel frames 1..8
- Latent 2 → pixel frames 9..16

So pinning `latent_idx=1` to `kf1` held pixel frames 1..8 ALL to `kf1`. Pinning `latent_idx=2` to `kf2` held pixel frames 9..16 all to `kf2`. The clip head visibly "slideshows" through three held keyframe images for 17 pixel frames before the model is free to denoise — far worse visually than the soft-guide drift it tried to eliminate.

My v1.10.0 plan phase claimed "consecutive `frame_index=0,1,2` is fully supported by `VideoConditionByLatentIndex` (frame 0) + `VideoConditionByKeyframeIndex` (frames 1, 2) — architecture has no sparse-keyframe assumption." That was wrong in both halves — `VideoConditionByKeyframeIndex` is a soft-guide mechanism by design (it appends keyframe tokens as context; the output's main video latent is still denoised from scratch), and `VideoConditionByLatentIndex` operates at latent-frame granularity which can't pin consecutive pixel frames without encoding a multi-frame video segment.

### v1.11.5 action

Revert v1.11.3's routing. `_image_conds_for_keyframes` now unconditionally delegates to `combined_image_conditionings` — the classical LTX semantic: `VideoConditionByLatentIndex` hard-pin on frame 0, `VideoConditionByKeyframeIndex` soft-guide on frames 1, 2. Keeps v1.11.4's `crf=0` change (reference images no longer CRF-33-compressed before VAE encode) and promotes the diagnostic log from `info` to `warning` so it actually emits (default Python logging threshold dropped `logger.info` silently).

### Proper fix deferred to v1.12

Hard-pinning multiple consecutive pixel frames requires encoding a **multi-frame video segment**, not three separate images. The correct chain-conditioning shape is:

1. Extract the prior clip's last 9 pixel frames as a short video (not 3 PNGs).
2. VAE-encode as a 2-latent-frame block.
3. Pin via a single `VideoConditionByLatentIndex(latent=encoded_segment, strength=1.0, latent_idx=0)`. That replaces latents 0-1 of the new clip simultaneously, giving temporally continuous content across the first 9 pixel frames that exactly continues the prior clip's motion trajectory.

Backend changes required:
- New helper for extracting a video segment (ffmpeg slice) keyed by start/end frame indices.
- Encode the segment via `video_encoder` as multi-frame latent.
- `KeyframeInput` schema extension: `video_uri + start_frame + end_frame` or a new `SegmentInput` type.
- FE switches from `/v2/video/extract-frames` (PNG output) to a new `/v2/video/extract-segment` (keeps as storage URI for re-use as conditioning input, or passes the video_uri + range inline).

Frontend changes:
- Instead of extracting 3 PNG frames, refer back to the prior clip's MP4 and send a segment reference to the chain endpoint.

Trade-off accepted until v1.12: pixel frame 0 pins cleanly; pixel frames 1, 2 drift softly. Subject-identity drift across many chain hops remains a known limitation best addressed by periodic re-grounding with the user's original image (FE concern, unchanged from earlier notes).
