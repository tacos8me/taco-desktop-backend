# LoRA API Design

## Overview

REST API for managing and applying user-supplied LoRA adapters in taco-backend. Follows existing patterns from `server.py` (Pydantic models, auth middleware, `_error()` helper, v1 sync + v2 async job endpoints).

LoRAs are `.safetensors` files stored on disk via `LoRARegistry` (see `lora_storage_design.md`). They are applied at inference time through `LoraPathStrengthAndSDOps` in `split_model_manager.py`.

---

## 1. Pydantic Models

### 1.1 LoRA Management Models

```python
from pydantic import BaseModel, Field
from datetime import datetime


class LoRAUploadRequest(BaseModel):
    """Metadata sent alongside LoRA file upload."""
    name: str = Field(max_length=200)
    description: str = Field(default="", max_length=2000)
    base_model: str = Field(default="ltx-2.3")  # "ltx-2.3" for all LTX variants


class LoRAInfo(BaseModel):
    """LoRA metadata returned by list/get endpoints."""
    id: str                          # UUID hex (32 chars)
    name: str
    filename: str                    # original upload filename
    base_model: str                  # e.g. "ltx-2.3"
    size_bytes: int
    uploaded_at: datetime
    description: str = ""


class LoRAListResponse(BaseModel):
    """Response for GET /v1/loras."""
    loras: list[LoRAInfo]
    count: int
```

### 1.2 LoRA Reference in Generation Requests

```python
class LoRAInput(BaseModel):
    """Reference to a user-uploaded LoRA for generation."""
    id: str = Field(description="LoRA ID from /v1/loras")
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
```

### 1.3 Modified Generation Request Models

Add an optional `lora` field to these existing models:

```python
class TextToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False
    camera_motion: str | None = Field(default=None, max_length=200)
    lora: LoRAInput | None = None           # NEW


class ImageToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str | None = None
    keyframes: list[KeyframeInput] | None = None
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False
    lora: LoRAInput | None = None           # NEW


class RetakeRequest(BaseModel):
    video_uri: str
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0, le=30)
    mode: RetakeMode
    prompt: str | None = Field(default=None, max_length=10000)
    lora: LoRAInput | None = None           # NEW
```

**Design decision: single `lora` not `loras` list.** Multiple user LoRAs would compound with preset LoRAs (pro stage 2, hq stages), making behavior unpredictable. Single LoRA is simpler and covers the primary use case. Can extend to a list later if needed.

---

## 2. Endpoint Specifications

### 2.1 GET /v1/loras

List all available LoRAs.

```
GET /v1/loras
Authorization: Bearer <key>

Response 200:
{
  "loras": [
    {
      "id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
      "name": "Cinematic Style",
      "filename": "cinematic_style.safetensors",
      "base_model": "ltx-2.3",
      "size_bytes": 156000000,
      "uploaded_at": "2026-03-09T10:30:00Z",
      "description": "Makes videos look cinematic"
    }
  ],
  "count": 1
}
```

Implementation: reads from `LoRARegistry` in-memory index. No pagination needed (practical LoRA count is <100).

### 2.2 POST /v1/loras

Upload and register a new LoRA.

Uses `multipart/form-data` to send the file and metadata together (mirrors common upload patterns, avoids a two-step upload+register flow).

```
POST /v1/loras
Authorization: Bearer <key>
Content-Type: multipart/form-data

Parts:
  file: <binary .safetensors>
  name: "Cinematic Style"
  description: "Makes videos look cinematic"         (optional)
  base_model: "ltx-2.3"                              (optional, default "ltx-2.3")

Response 201:
{
  "id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "name": "Cinematic Style",
  "filename": "cinematic_style.safetensors",
  "base_model": "ltx-2.3",
  "size_bytes": 156000000,
  "uploaded_at": "2026-03-09T10:30:00Z",
  "description": "Makes videos look cinematic"
}

Error 400: Invalid file format (not .safetensors or missing LoRA keys)
Error 413: File exceeds 500MB limit
Error 422: Validation error (missing name, etc.)
```

FastAPI handler signature:

```python
from fastapi import UploadFile, File, Form

@app.post("/v1/loras", status_code=201)
async def upload_lora(
    file: UploadFile = File(...),
    name: str = Form(..., max_length=200),
    description: str = Form("", max_length=2000),
    base_model: str = Form("ltx-2.3"),
) -> JSONResponse:
```

Validation steps:
1. Check file extension is `.safetensors`
2. Check file size <= 500MB (stream to temp, check as we go)
3. Parse SafeTensors header to verify LoRA key structure (`*.lora_A.weight`, `*.lora_B.weight`)
4. Generate UUID, save to `loras/models/{base_model}/{lora_id}/model.safetensors`
5. Write `metadata.json`
6. Update registry index
7. Return `LoRAInfo`

### 2.3 DELETE /v1/loras/{lora_id}

Remove a LoRA from storage and registry.

```
DELETE /v1/loras/{lora_id}
Authorization: Bearer <key>

Response 200:
{"deleted": true, "id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"}

Error 404: LoRA not found
```

Implementation: removes directory `loras/models/{base_model}/{lora_id}/`, updates registry. If the LoRA is currently fused into a loaded transformer, the transformer continues to work (fusion is baked in). Next generation with a different LoRA will reload anyway.

---

## 3. LoRA Flow Through to split_model_manager.py

### 3.1 Current Transformer State Machine

`DenoiserWorker.ensure_transformer(state)` handles these states:

| State | Checkpoint | Built-in LoRAs |
|-------|-----------|----------------|
| `"dev"` | dev | none |
| `"distilled"` | distilled | none |
| `"dev_lora"` | dev | distilled_lora @ 1.0 |
| `"dev_lora_025"` | dev | distilled_lora @ 0.25 |
| `"dev_lora_050"` | dev | distilled_lora @ 0.50 |

### 3.2 Extended ensure_transformer

Add `user_lora` parameter:

```python
def ensure_transformer(
    self,
    state: str,
    user_lora: tuple[str, float] | None = None,  # (path, strength)
) -> None:
    """Swap transformer checkpoint. Reloads if state or user_lora changed."""
    # Cache key includes user_lora to detect changes
    cache_key = (state, user_lora)
    if self._current_cache_key == cache_key:
        return

    # ... existing checkpoint/lora resolution for `state` ...

    # Append user LoRA if provided
    if user_lora:
        path, strength = user_lora
        loras = loras + (LoraPathStrengthAndSDOps(
            path=path,
            strength=strength,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        ),)

    # ... load transformer with ModelLedger(loras=loras) ...
    self._current_cache_key = cache_key
```

Key points:
- **LoRA fusion is permanent**: `ModelLedger` fuses LoRA weights into the transformer at load time. There's no `unfuse()`. Any change in LoRA identity or strength requires a full transformer reload (~2-3s).
- **User LoRA composes with preset LoRA**: For `ltx-2-3-pro` stage 2, the transformer is loaded with both `distilled_lora` (preset) and the user LoRA. Both get fused.
- **Cache key prevents unnecessary reloads**: If the same user LoRA + same state is requested again, skip reload.
- **ComfyUI compatibility**: `LTXV_LORA_COMFY_RENAMING_MAP` strips the `diffusion_model.` prefix, so LoRAs trained with ComfyUI work automatically.

### 3.3 Request Flow

```
Client                    server.py                 split_model_manager.py
  |                          |                              |
  |-- POST /v1/t2v --------->|                              |
  |   {lora: {id, strength}} |                              |
  |                          |-- validate lora_id exists -->|
  |                          |   via lora_registry.get()    |
  |                          |                              |
  |                          |-- resolve lora path -------->|
  |                          |   lora_registry.resolve_path()|
  |                          |                              |
  |                          |-- generate_text_to_video --->|
  |                          |   (lora_path=...,            |
  |                          |    lora_strength=...)        |
  |                          |                              |
  |                          |      ensure_transformer()    |
  |                          |      with user_lora tuple    |
  |                          |                              |
  |                          |      [reload if needed ~2-3s]|
  |                          |                              |
  |                          |      denoise + decode        |
  |<-- video/mp4 ------------|<-----------------------------|
```

### 3.4 Generation Method Changes

Add `lora_path` and `lora_strength` params to `generate_text_to_video()`, `generate_image_to_video()`, and `retake()`:

```python
async def generate_text_to_video(
    self, prompt: str, model: str, width: int, height: int,
    num_frames: int, fps: float, seed: int, generate_audio: bool,
    on_progress=None,
    lora_path: str | None = None,      # NEW
    lora_strength: float = 1.0,        # NEW
) -> bytes:
    worker = await self._acquire_worker()
    try:
        user_lora = (lora_path, lora_strength) if lora_path else None
        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_t2v, worker, prompt, model, width, height,
            num_frames, fps, seed, generate_audio, on_progress, user_lora,
        )
    finally:
        worker.lock.release()
```

Inside `_run_t2v`, pass `user_lora` to each `ensure_transformer()` call:

```python
# Stage 1
worker.ensure_transformer("distilled" if is_fast else "dev", user_lora=user_lora)

# Stage 2 (pro)
worker.ensure_transformer("dev_lora", user_lora=user_lora)

# Reset
worker.ensure_transformer("dev")
```

### 3.5 server.py Endpoint Changes

In each video generation endpoint, resolve the LoRA before calling the manager:

```python
@app.post("/v1/text-to-video")
async def text_to_video(body: TextToVideoRequest) -> Response:
    if not manager.is_ready:
        return _error(500, "No GPU workers loaded")

    # Resolve optional LoRA
    lora_path: str | None = None
    lora_strength: float = 1.0
    if body.lora:
        lora_info = lora_registry.get_lora(body.lora.id)
        if lora_info is None:
            return _error(404, f"LoRA not found: {body.lora.id}")
        lora_path = str(lora_registry.resolve_path(lora_info))
        lora_strength = body.lora.strength

    try:
        # ... existing param setup ...
        async with _inference_lock:
            video_bytes = await manager.generate_text_to_video(
                ...,
                lora_path=lora_path,
                lora_strength=lora_strength,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except Exception as exc:
        logger.exception("text-to-video failed")
        return _error(500, str(exc))
```

Same pattern for v2 async endpoints -- resolve LoRA ID to path during submission (before queuing), so a deleted LoRA during queue wait returns a clear error early.

---

## 4. Edge Cases

### 4.1 LoRA Not Found

- **At request time**: Return 404 immediately. Resolved before job submission (v2) or inference (v1).
- **Approach**: Validate `body.lora.id` against `lora_registry` in the endpoint handler, before `_submit_job()` or `_inference_lock`.

### 4.2 LoRA Deleted During Generation

- **Not a problem**: By the time generation starts, `ensure_transformer()` has already fused the LoRA weights into the transformer tensor. Deleting the file on disk doesn't affect the in-memory model.
- **Between queue and execution (v2)**: The LoRA path is resolved at submission time and stored in `job.params`. If the file is deleted between submission and execution, `ModelLedger` will raise `FileNotFoundError` when loading. The job worker catches this and marks the job as failed with error `"LoRA file not found"`.

### 4.3 LoRA Incompatible with Model

- **Only LTX LoRAs are supported**: `base_model` is always `"ltx-2.3"`. Flux LoRAs are out of scope (Flux uses a completely different model architecture and diffusers-based loading).
- **Wrong architecture**: If a LoRA has mismatched key names, `ModelLedger` will raise during fusion. Caught as a 500 error during generation. Mitigated by upload-time validation (step 3 in POST /v1/loras).
- **ltx-2-3-fast compatibility**: Fast model uses the distilled checkpoint. User LoRAs trained on dev may produce degraded results on distilled. This is a user responsibility -- no server-side enforcement, but we could log a warning.

### 4.4 LoRA + HQ Pipeline

The HQ pipeline (`ltx-2-3-hq`) already uses preset LoRAs at specific strengths:
- Stage 1: `dev_lora_025` (distilled_lora @ 0.25)
- Stage 2: `dev_lora_050` (distilled_lora @ 0.50)

User LoRA composes additively -- both the preset `distilled_lora` and user LoRA are fused into the transformer. The user controls only their LoRA's strength; preset strengths remain fixed.

### 4.5 Concurrent Requests with Different LoRAs

Each request may need a different LoRA, triggering a transformer reload (~2-3s). With the inference lock serializing GPU access, this is safe but adds latency. The cache key `(state, user_lora)` ensures no unnecessary reloads when consecutive requests use the same LoRA.

### 4.6 Upload Validation Failures

| Scenario | HTTP Status | Error Message |
|----------|------------|---------------|
| Not `.safetensors` extension | 400 | "File must be a .safetensors file" |
| File > 500MB | 413 | "File exceeds 500MB limit" |
| No LoRA keys in safetensors | 400 | "File does not contain LoRA weights" |
| Corrupt safetensors header | 400 | "Invalid safetensors file" |
| Missing `name` field | 422 | Pydantic validation error |

### 4.7 LoRA Strength Bounds

- Accepted range: `0.0` to `2.0` (Pydantic validation on `LoRAInput.strength`)
- `0.0` = LoRA has no effect (but still triggers a reload -- not worth optimizing)
- `1.0` = default, full strength
- `>1.0` = amplified effect, may produce artifacts (user's choice)

---

## 5. Global State Changes

### 5.1 New Global in server.py

```python
from lora_registry import LoRARegistry

lora_registry = LoRARegistry(config.LORAS_DIR)
```

### 5.2 New Config Entry

```python
# config.py
LORAS_DIR = Path("/mnt/nvme-1/servers/taco-backend/loras")
```

### 5.3 Startup

Registry loads from disk in lifespan (after LTX pipeline load):

```python
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # ... existing pipeline loads ...
    lora_registry.refresh()
    logger.info("LoRA registry loaded: %d loras", lora_registry.count())
    yield
```

---

## 6. Summary of Changes by File

| File | Changes |
|------|---------|
| `server.py` | Add 3 LoRA management endpoints, add `lora` field to 3 request models, resolve LoRA in generation handlers |
| `split_model_manager.py` | Add `user_lora` param to `ensure_transformer()`, `_run_t2v`, `_run_t2v_hq`, `_run_i2v`, `retake`; extend cache key |
| `config.py` | Add `LORAS_DIR` path constant |
| `lora_registry.py` | New file: `LoRARegistry` class (see `lora_storage_design.md`) |
