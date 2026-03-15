# Plan: Proper Retake with Temporal Masking

## Current State

`_run_retake` in `split_model_manager.py` (line 626) accepts `start_time`, `duration`, and `mode` parameters but **ignores them for masking purposes**. It calls `denoise_audio_video()` with `conditionings=[]`, which means `denoise_mask = 1` everywhere. The result: the entire video is regenerated from scratch regardless of what temporal region the user requested.

The `mode` parameter (`regenerate_video` / `regenerate_audio` booleans) is computed but never used.

## Target State

Match the upstream `RetakePipeline` in `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/retake.py`: use `TemporalRegionMask` conditioning to set `denoise_mask = 0` outside `[start_time, end_time]` and `1` inside it, so only the specified temporal region is regenerated while the rest is preserved.

## How TemporalRegionMask Works

Defined in `retake.py` lines 109-145. It's a `@dataclass(frozen=True)` that implements the `ConditioningItem` protocol.

**Fields:**
- `start_time: float` -- seconds, inclusive
- `end_time: float` -- seconds, exclusive
- `fps: float` -- video FPS, needed to convert seconds to latent frame indices

**`apply_to(latent_state, latent_tools)`:**
1. Gets patch grid bounds from the patchifier (`get_patch_grid_bounds`)
2. For **video** (coords has 3 dims: temporal, height, width): converts patch bounds to pixel coordinates via `get_pixel_coords`, then to timestamps by dividing by `fps`. Patches overlapping `[start_time, end_time)` get `denoise_mask = 1`.
3. For **audio** (coords has 1 dim): patchifier already returns seconds. Same overlap logic.
4. Clones the state and overwrites `denoise_mask` with the computed mask.

**Effect on noising (GaussianNoiser):**
- `denoise_mask = 1`: token gets fully noised (pure Gaussian) -- will be regenerated
- `denoise_mask = 0`: token keeps original clean latent -- preserved as-is
- Formula: `latent = noise * mask + clean_latent * (1 - mask)`

## Key Difference from Current Code

The upstream `RetakePipeline.__call__()` does NOT use `denoise_audio_video()`. Instead it calls `noise_video_state()` and `noise_audio_state()` separately, passing **separate conditionings lists** to each:

```python
# Video conditionings
video_conditionings = [
    TemporalRegionMask(
        start_time=start_time if regenerate_video else 0.0,
        end_time=end_time if regenerate_video else 0.0,
        fps=fps,
    )
]

# Audio conditionings (separate list, only if audio exists)
audio_conditionings = [
    TemporalRegionMask(
        start_time=start_time if regenerate_audio else 0.0,
        end_time=end_time if regenerate_audio else 0.0,
        fps=fps,
    )
]
```

When `regenerate_video=False`, `start_time=end_time=0.0` so no patches overlap the range and `denoise_mask` stays `0` everywhere (fully preserved). Same logic for audio.

This is why we can't use `denoise_audio_video()` for retake -- it only takes a single `conditionings` list that it passes to video but hardcodes `conditionings=[]` for audio.

## Changes Needed in `_run_retake`

### 1. Import TemporalRegionMask

```python
from ltx_pipelines.retake import TemporalRegionMask
```

### 2. Replace `denoise_audio_video()` with separate `noise_video_state()` / `noise_audio_state()` calls

Follow the upstream pattern: call each with its own conditionings list containing a `TemporalRegionMask`, then run the denoising loop, then unpatchify.

### 3. Handle mode-dependent masking

Use the already-computed `regenerate_video` and `regenerate_audio` booleans to control which modality gets the temporal mask vs full preservation:

```python
video_conditionings = [
    TemporalRegionMask(
        start_time=start_time if regenerate_video else 0.0,
        end_time=end_time if regenerate_video else 0.0,
        fps=fps_vid,
    )
]
audio_conditionings = [
    TemporalRegionMask(
        start_time=start_time if regenerate_audio else 0.0,
        end_time=end_time if regenerate_audio else 0.0,
        fps=fps_vid,
    )
]
```

### 4. Build noised states separately

Replace the single `denoise_audio_video()` call (lines 679-685) with:

```python
video_state, video_tools = noise_video_state(
    output_shape=output_shape, noiser=noiser,
    conditionings=video_conditionings,
    components=worker.components, dtype=dtype, device=device,
    initial_latent=initial_video_latent,
)
audio_state, audio_tools = noise_audio_state(
    output_shape=output_shape, noiser=noiser,
    conditionings=audio_conditionings,
    components=worker.components, dtype=dtype, device=device,
    initial_latent=initial_audio_latent,
)

video_state, audio_state = retake_loop(sigmas, video_state, audio_state, stepper)

video_state = video_tools.clear_conditioning(video_state)
video_state = video_tools.unpatchify(video_state)
audio_state = audio_tools.clear_conditioning(audio_state)
audio_state = audio_tools.unpatchify(audio_state)
```

### 5. Add needed imports

The `noise_video_state` and `noise_audio_state` functions are already imported (used by other methods). `TemporalRegionMask` needs to be added.

### 6. Handle missing audio gracefully

If the source video has no audio track, `decode_audio_from_file` returns `None`. Currently the code doesn't guard against this. The upstream handles it by checking `audio_in is not None` before encoding. We should:
- Set `initial_audio_latent = None` and `audio_conditionings = []` when no audio
- Still pass them to `noise_audio_state` (it handles `None` initial latent)

## Changes Needed in the Endpoint

### server.py

No changes needed. The endpoint already sends `start_time`, `duration`, `mode`, and `prompt` through to `manager.retake()`. The `RetakeRequest` model already has the right fields:

```python
class RetakeRequest(BaseModel):
    video_uri: str
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0, le=30)
    mode: RetakeMode
    prompt: str | None = Field(default=None, max_length=10000)
```

The v2 endpoint similarly passes all params through.

### Request Body Params

| Param | Type | Semantics |
|-------|------|-----------|
| `video_uri` | str | `storage://` URI of the source video |
| `start_time` | float (seconds, >= 0) | Inclusive start of the region to regenerate |
| `duration` | float (seconds, > 0, <= 30) | Length of the region to regenerate |
| `mode` | RetakeMode | Controls which modalities to regenerate |
| `prompt` | str or null | Text prompt for the regenerated section |

**RetakeMode semantics:**

| Mode | `regenerate_video` | `regenerate_audio` | Effect |
|------|-------------------|-------------------|--------|
| `replace_audio_and_video` | true | true | Regenerate both modalities in `[start, start+duration]` |
| `replace_video` | true | true | Same as above (current definition -- possibly should be renamed) |
| `replace_video_only` | true | false | Regenerate only video; audio preserved everywhere |
| `replace_audio` | false | true | Regenerate only audio; video preserved everywhere |

Note: `replace_video` and `replace_audio_and_video` currently produce identical behavior. Consider if `replace_video` should actually mean video-only (making it equivalent to `replace_video_only`). This is a design question for the frontend team -- current implementation just mirrors the existing boolean logic.

## Edge Cases

### 1. Retake at video start (start_time = 0)
Works naturally. `TemporalRegionMask` will mask from time 0 to `end_time`.

### 2. Retake at video end
If `start_time + duration > video_length`, the mask extends past the video but patches only exist within the video bounds, so only existing patches are affected. No special handling needed.

### 3. Retake of entire video
If `start_time = 0` and `duration >= video_length`, all patches get `denoise_mask = 1` -- equivalent to full regeneration (current behavior). Works as expected.

### 4. Audio-only retake
Set `regenerate_video=False`: video gets mask `[0, 0)` which is empty, so `denoise_mask = 0` everywhere for video (preserved). Audio gets the temporal mask and regenerates. The encoded video latent still serves as the initial state for the denoising loop so the transformer has context.

### 5. Video without audio track
`decode_audio_from_file` returns `None`. We should skip audio encoding and pass `initial_audio_latent=None` with `audio_conditionings=[]`. The noiser will generate a random initial audio state. Since audio has `denoise_mask = 1` by default (no conditioning to zero it out), the full audio track will be generated from scratch. This is acceptable behavior.

If the user requests `replace_audio` on a video without an audio track, this effectively generates audio for the temporal region from the text prompt. We could alternatively fail with a 422 error for `replace_audio` mode on audio-less videos, but generating audio is probably the better UX.

### 6. Very short retake regions
If the retake region is shorter than one latent patch (approximately 0.33s at 24fps with 8x temporal downsampling), the patch that overlaps the region boundary will be regenerated in its entirety. This is inherent to the patchified representation and not something we need to work around.

### 7. Frame count validation
The source video must have `8k+1` frames (LTX constraint). The upstream CLI validates this. We should add validation in `_run_retake` and return a clear error if the constraint is violated. Alternatively, we could snap the frame count (truncate to nearest `8k+1`), but this could cause audio/video sync issues.

## Implementation Sequence

1. Add `TemporalRegionMask` import to `split_model_manager.py`
2. Replace the `denoise_audio_video()` call in `_run_retake` with separate `noise_video_state`/`noise_audio_state` calls using temporal conditionings
3. Handle the no-audio case with a guard
4. Add frame count validation (8k+1 check) with a clear error message
5. Test with each mode variant

## Testing Strategy

### Smoke tests (manual)
1. **Full retake** (start_time=0, duration=full): Should produce same result as current code (baseline regression check)
2. **Middle retake**: start_time=1.0, duration=2.0 on a 5s video. Verify first 1s and last 2s are visually identical to source, middle 2s are regenerated
3. **Start retake**: start_time=0, duration=1.5. Verify latter portion matches source
4. **End retake**: start_time=3.0, duration=2.0 on a 5s video. Verify first 3s matches source
5. **Audio-only retake**: mode=replace_audio. Verify video frames are pixel-identical to source
6. **Video-only retake**: mode=replace_video_only. Verify audio waveform matches source
7. **No-audio source**: Upload a video without audio, run retake. Should succeed and generate new audio

### Automated tests (pytest)
1. Unit test `TemporalRegionMask` application: create a dummy `LatentState`, apply mask, verify correct tokens are zeroed/oned
2. Integration test: mock the transformer, run `_run_retake` with a short test video, verify output shape matches input shape
3. Test mode matrix: for each `RetakeMode`, verify `regenerate_video`/`regenerate_audio` booleans are computed correctly

### Verification approach for temporal preservation
To verify that unmasked regions are truly preserved, compare decoded frames from the retake output against decoded frames from the original encoded-then-decoded video (not the raw source, since VAE encode/decode introduces minor reconstruction error). The comparison should show near-zero difference outside the retake window and significant difference inside it.
