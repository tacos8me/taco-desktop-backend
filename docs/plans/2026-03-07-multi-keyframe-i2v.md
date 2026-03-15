# Multi-Keyframe Image-to-Video

## Current State

The i2v endpoint accepts a single image via `image_uri` and always conditions on frame 0:

```python
# server.py — ImageToVideoRequest
class ImageToVideoRequest(BaseModel):
    prompt: str
    image_uri: str          # single image
    model: ModelName
    resolution: Resolution
    duration: float
    fps: float
    generate_audio: bool = False

# split_model_manager.py — _run_i2v
images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]
```

The underlying `combined_image_conditionings()` already supports multiple keyframes natively. It dispatches per image:

- **frame_idx == 0** -> `VideoConditionByLatentIndex`: replaces the latent tokens at the first frame position in-place (strong spatial control).
- **frame_idx > 0** -> `VideoConditionByKeyframeIndex`: appends extra keyframe tokens with positional encoding offset by `frame_idx` (softer guidance via cross-attention).

Both conditioning types take a `strength` parameter (1.0 = fully clean / maximum influence, 0.0 = fully denoised / no influence).

The `ImageConditioningInput` NamedTuple:
```python
class ImageConditioningInput(NamedTuple):
    path: str           # filesystem path to image
    frame_idx: int      # target frame index (pixel frames, not latent)
    strength: float     # conditioning strength [0.0, 1.0]
    crf: int = 33       # H.264 compression for VAE preprocessing
```

## Target State

Accept multiple keyframe images, each targeting a specific frame in the output video. This enables:
- First-frame + last-frame conditioning (start/end bookending)
- Mid-video keyframes for scene transitions
- Single-image requests continue to work unchanged

## API Changes

### Request Body

```python
class KeyframeInput(BaseModel):
    image_uri: str
    frame_index: int = 0          # target frame in output video (pixel frames)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)

class ImageToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str | None = None          # DEPRECATED but still supported for backward compat
    keyframes: list[KeyframeInput] | None = None  # new multi-keyframe field
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False
```

### Backward Compatibility

- If `image_uri` is set and `keyframes` is None/empty: treat as single keyframe at frame 0 with strength 1.0 (current behavior).
- If `keyframes` is set: use the keyframe list. `image_uri` must be None or an error is returned.
- Validation: at least one keyframe is required.

### Endpoint Logic (server.py)

```python
@app.post("/v1/image-to-video")
async def image_to_video(body: ImageToVideoRequest) -> Response:
    # Resolve keyframes
    if body.keyframes and body.image_uri:
        return _error(422, "Cannot specify both image_uri and keyframes")

    if body.keyframes:
        keyframe_inputs = []
        for kf in body.keyframes:
            path = str(uploads.resolve(kf.image_uri))
            keyframe_inputs.append({"image_path": path, "frame_index": kf.frame_index, "strength": kf.strength})
    elif body.image_uri:
        path = str(uploads.resolve(body.image_uri))
        keyframe_inputs = [{"image_path": path, "frame_index": 0, "strength": 1.0}]
    else:
        return _error(422, "Either image_uri or keyframes is required")

    # Pass keyframe_inputs to manager
    video_bytes = await manager.generate_image_to_video(
        prompt=body.prompt,
        keyframes=keyframe_inputs,
        model=body.model,
        ...
    )
```

Same pattern for V2 endpoint (`/v2/image-to-video`).

## SplitModelManager Changes

### Signature Change

```python
# Before
async def generate_image_to_video(self, ..., image_path: str, ...) -> bytes:

# After
async def generate_image_to_video(self, ..., keyframes: list[dict], ...) -> bytes:
```

Where each dict has `{"image_path": str, "frame_index": int, "strength": float}`.

### _run_i2v Changes

```python
def _run_i2v(self, worker, prompt, keyframes, model, width, height,
             num_frames, fps, seed, generate_audio, on_progress=None):
    # Build ImageConditioningInput list from keyframes
    images = [
        ImageConditioningInput(path=kf["image_path"], frame_idx=kf["frame_index"], strength=kf["strength"])
        for kf in keyframes
    ]

    # Rest of the method is unchanged — combined_image_conditionings() already
    # handles the list correctly for both stage 1 and stage 2.
```

The existing calls to `combined_image_conditionings()` on lines 450 and 489 already pass the full `images` list, so they work as-is with multiple entries.

### Prompt Enhancement

Currently `encode_prompts()` accepts `enhance_prompt_image` for i2v prompt enhancement. With multiple keyframes, use the first keyframe's image (frame_idx=0 if present, otherwise the first in the list):

```python
enhance_image = None
first_frame_imgs = [img for img in images if img.frame_idx == 0]
if first_frame_imgs:
    enhance_image = first_frame_imgs[0].path
elif images:
    enhance_image = images[0].path
```

This is not currently used in `_run_i2v` (no enhance_prompt flag), so this is a future consideration.

## How Keyframes Map to Frame Indices

`frame_index` in the API is in **pixel frame space** (0-indexed). The frame count is computed from duration and fps:

```
num_frames = 8k + 1  (snapped by _duration_to_frames)
```

For example: 3 seconds at 24fps = 73 frames (indices 0-72).

In `combined_image_conditionings()`:
- **frame_idx=0**: Uses `VideoConditionByLatentIndex(latent_idx=0)` which replaces the first latent frame in-place. This gives the strongest control.
- **frame_idx=N (N>0)**: Uses `VideoConditionByKeyframeIndex(frame_idx=N)` which appends separate keyframe tokens with positional encoding offset by N. The positions are divided by fps for temporal embedding.

### Important constraints
- `frame_index` must be in range `[0, num_frames - 1]`
- Only one image can have `frame_index=0` (latent replacement is destructive; second one would overwrite)
- Multiple images with `frame_index > 0` are fine (they append independent keyframe token sets)
- The API should validate these constraints before passing to the manager

### Stage 1 vs Stage 2

Both stages call `combined_image_conditionings()` with the same `images` list but different resolutions:
- Stage 1: `height//2, width//2` (half resolution)
- Stage 2: `height, width` (full resolution)

The image is re-encoded at each stage's resolution via `load_image_conditioning()` which center-crops and resizes. Frame indices don't need adjustment between stages.

## Validation Rules

Add to endpoint logic:

1. `frame_index` must be `>= 0` (Pydantic `ge=0`)
2. At most one keyframe can have `frame_index == 0`
3. `frame_index` should be `< num_frames` (validated after computing num_frames)
4. No duplicate `frame_index` values
5. `keyframes` list must be non-empty
6. Maximum of ~8 keyframes (practical limit; each adds tokens to attention)

## Audio-to-Video

The a2v endpoint also supports optional image conditioning. The same `keyframes` pattern could be applied there, but since a2v is a separate flow with different requirements (video-only denoising, audio passthrough), defer that to a follow-up. The current single-image support in a2v is sufficient.

## Testing Strategy

### Unit / Integration Tests

1. **Single-image backward compat**: Send request with `image_uri` only, verify it works identically to current behavior.

2. **Single keyframe**: Send `keyframes: [{image_uri: "...", frame_index: 0, strength: 1.0}]`, verify equivalent to `image_uri` path.

3. **Two keyframes (first + last)**: Send two keyframes at frame 0 and frame N-1. Verify:
   - Both conditioning types created correctly
   - Video generates without error
   - Output length matches requested duration

4. **Mid-video keyframe**: Single keyframe at frame N/2. Verify `VideoConditionByKeyframeIndex` is used (not latent replacement).

5. **Strength variation**: Keyframe at frame 0 with strength 0.5 should produce less constrained output than strength 1.0.

6. **Validation errors**:
   - `keyframes: []` -> 422
   - Both `image_uri` and `keyframes` set -> 422
   - `frame_index` >= num_frames -> 422
   - Duplicate frame indices -> 422
   - Two keyframes at frame 0 -> 422
   - Neither `image_uri` nor `keyframes` -> 422

### Smoke Test (Manual)

```bash
# Upload two images
IMG1=$(curl -s http://localhost:8090/v1/upload | jq -r .storage_uri)
curl -X PUT "http://localhost:8090/uploads/put/$(basename $IMG1)" --data-binary @first.png

IMG2=$(curl -s http://localhost:8090/v1/upload | jq -r .storage_uri)
curl -X PUT "http://localhost:8090/uploads/put/$(basename $IMG2)" --data-binary @last.png

# Generate with two keyframes
curl -X POST http://localhost:8090/v2/image-to-video \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A smooth transition between scenes",
    "keyframes": [
      {"image_uri": "'$IMG1'", "frame_index": 0, "strength": 1.0},
      {"image_uri": "'$IMG2'", "frame_index": 72, "strength": 0.8}
    ],
    "model": "ltx-2-3-fast",
    "resolution": "1920x1080",
    "duration": 3,
    "fps": 24
  }'
```

### Performance Considerations

Each additional keyframe adds:
- One `load_image_conditioning()` call (image decode + resize + H.264 preprocessing)
- One `video_encoder()` forward pass (VAE encoding of single frame)
- Additional tokens in the attention computation (for `frame_idx > 0` keyframes)

This happens twice (stage 1 + stage 2). For 2-3 keyframes, overhead is negligible (~100ms total). For 8+ keyframes, attention token count grows noticeably but should still be manageable on 96GB GPUs.

## Implementation Checklist

1. Add `KeyframeInput` model and update `ImageToVideoRequest` in `server.py`
2. Add validation logic in i2v endpoint (both v1 and v2)
3. Update `generate_image_to_video` and `_run_i2v` signatures to accept `keyframes: list[dict]`
4. Build `ImageConditioningInput` list from keyframes in `_run_i2v`
5. Update `_dispatch_job` params if needed (should flow through via `**p`)
6. Add tests for validation and backward compatibility
7. Smoke test with actual multi-keyframe generation
