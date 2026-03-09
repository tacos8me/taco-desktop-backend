# LoRA Storage and Discovery System Design

## Overview

This document specifies the storage, discovery, and management system for LoRA adapters in taco-backend. The system enables users to upload, discover, organize, and apply LoRAs without modifying configuration files.

## 1. Storage Architecture

### 1.1 Directory Structure

```
/mnt/nvme-1/servers/taco-backend/loras/
├── registry.json              # Central registry metadata (LoRA catalog)
├── models/
│   ├── {model_id}/            # Namespace per base model (ltx-2.3, flux-2-dev, etc.)
│   │   ├── {lora_id}/
│   │   │   ├── model.safetensors    # LoRA weights
│   │   │   └── metadata.json        # LoRA metadata + discovery info
│   │   │   └── preview.jpg          # Optional preview image
│   │   └── ...
│   └── ...
└── uploads/                   # Temporary staging for new uploads
    ├── {upload_session_id}/
    │   ├── model.safetensors
    │   └── metadata.json
    └── ...
```

### 1.2 LoRA Naming & Identification

- **LoRA ID**: UUID hex (32 chars) — unique identifier for each LoRA
- **Model ID**: Canonical name (e.g., `ltx-2.3-dev`, `flux-2-dev`) — groups LoRAs by compatible base model
- **Full path**: `loras/models/{model_id}/{lora_id}/model.safetensors`

### 1.3 File Formats

#### Metadata Manifest: `metadata.json`
Required in each LoRA directory. Contains discovery and validation information.

```json
{
  "id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "name": "Cinematic Style",
  "description": "Makes videos look cinematic with enhanced contrast",
  "base_model": "ltx-2.3-dev",
  "author": "user@example.com",
  "version": "1.0",
  "created_at": "2024-03-01T10:30:00Z",
  "updated_at": "2024-03-01T10:30:00Z",
  "tags": ["style", "cinematic", "contrast"],
  "lora_metadata": {
    "rank": 32,
    "alpha": 16,
    "target_modules": ["attention", "mlp"],
    "reference_downscale_factor": 1,
    "training_params": {
      "base_model_name": "ltx-2.3-22b-dev",
      "training_steps": 5000,
      "learning_rate": 0.0001
    }
  },
  "stats": {
    "size_bytes": 156000000,
    "param_count": 2097152,
    "strength_recommend": 0.8,
    "strength_min": 0.0,
    "strength_max": 1.5
  },
  "compatibility": {
    "min_server_version": "1.0.0",
    "deprecated": false,
    "alternate_ids": []
  },
  "preview_image": "preview.jpg"
}
```

#### Central Registry: `registry.json`
Index of all available LoRAs. Updated when LoRAs are added/removed.

```json
{
  "version": "1.0",
  "last_updated": "2024-03-01T10:30:00Z",
  "models": {
    "ltx-2.3-dev": {
      "name": "LTX-2.3 Dev",
      "loras": [
        {
          "id": "a1b2c3d4...",
          "name": "Cinematic Style",
          "author": "user@example.com",
          "tags": ["style", "cinematic"],
          "strength_recommend": 0.8,
          "size_bytes": 156000000,
          "created_at": "2024-03-01T10:30:00Z"
        }
      ]
    },
    "flux-2-dev": {
      "name": "Flux 2 Dev",
      "loras": []
    }
  },
  "stats": {
    "total_loras": 3,
    "total_size_bytes": 468000000,
    "models_with_loras": 1
  }
}
```

## 2. Discovery System

### 2.1 Discovery Mechanism

LoRAs are discovered by:
1. **Scanning**: Directory walk of `loras/models/{model_id}/` on startup + periodic refresh
2. **Registry lookup**: Fast queries via in-memory registry built from `registry.json`
3. **Caching**: Registry kept in memory; rebuilt only on disk changes (5-min TTL minimum)

### 2.2 LoRA Loader Pattern

```python
class LoRARegistry:
    """In-memory registry of available LoRAs."""

    def __init__(self, loras_dir: Path):
        self.loras_dir = loras_dir
        self.registry: dict[str, dict[str, LoRAInfo]] = {}
        self._last_refresh = 0
        self._refresh_interval = 300  # 5 minutes

    def get_lora(self, model_id: str, lora_id: str) -> LoRAInfo | None:
        """Fetch LoRA metadata by model and LoRA ID."""
        self._refresh_if_needed()
        return self.registry.get(model_id, {}).get(lora_id)

    def list_loras(self, model_id: str) -> list[LoRAInfo]:
        """List all LoRAs for a base model."""
        self._refresh_if_needed()
        return list(self.registry.get(model_id, {}).values())

    def _refresh_if_needed(self) -> None:
        """Rebuild registry from disk if 5+ minutes have passed."""
        now = time.time()
        if now - self._last_refresh > self._refresh_interval:
            self._load_from_disk()
            self._last_refresh = now

    def _load_from_disk(self) -> None:
        """Scan loras/models/ and rebuild registry."""
        # Implementation: walk directory, parse metadata.json files
        pass
```

## 3. Upload & Registration Flow

### 3.1 Upload Workflow (Existing UploadStore Pattern)

1. **POST /v2/loras/upload** (new endpoint)
   - Receive LoRA file + metadata
   - Generate upload ID (UUID hex)
   - Save to `loras/uploads/{upload_id}/model.safetensors`
   - Return upload ID + storage:// URI

2. **POST /v2/loras/register** (new endpoint)
   - Accept upload ID + final metadata (name, description, tags, etc.)
   - Validate LoRA file format (SafeTensors)
   - Extract actual LoRA metadata from file
   - Move from `uploads/` to `models/{model_id}/{lora_id}/`
   - Update registry
   - Return registered LoRA metadata

### 3.2 Validation Flow

On registration:
1. Parse SafeTensors header (fast, no full load)
2. Verify key structure: `{module}.lora_A.weight`, `{module}.lora_B.weight`
3. Infer `base_model` from key prefix matching known models
4. Check compatibility: model_id matches inferred base model
5. Extract SafeTensors metadata (if present) → lora_metadata.training_params
6. Calculate file size, estimate param count from shapes

## 4. LoRA Application in Inference

### 4.1 Current Pattern (split_model_manager.py)

**Before**:
```python
elif state == "dev_lora_050":
    distilled_lora = LoraPathStrengthAndSDOps(
        path=config.DISTILLED_LORA, strength=0.5,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )
    checkpoint, loras = config.DEV_CHECKPOINT, (distilled_lora,)
```

**After**:
```python
def ensure_transformer(self, state: str, lora_ids: list[str] | None = None) -> None:
    """Swap transformer with optional user-supplied LoRAs."""
    if lora_ids is None:
        lora_ids = []

    # Load base checkpoint + matching LoRAs
    loras = []
    for lora_id in lora_ids:
        lora_info = self.lora_registry.get_lora("ltx-2.3-dev", lora_id)
        if not lora_info:
            raise ValueError(f"LoRA not found: {lora_id}")
        lora_path = self.lora_registry.resolve_path(lora_info)
        loras.append(LoraPathStrengthAndSDOps(
            path=lora_path,
            strength=lora_info.strength_recommend,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        ))

    # ... load transformer with loras tuple
```

### 4.2 Request Flow

User requests generation with LoRAs:
```json
{
  "prompt": "cinematic scene...",
  "loras": [
    {"id": "a1b2c3d4...", "strength": 0.8},
    {"id": "x9y8z7w6...", "strength": 0.6}
  ]
}
```

Server:
1. Validates LoRA IDs exist and are compatible with selected model variant
2. Builds LoRA list with requested strengths (override defaults)
3. Passes to `ensure_transformer()` to load model + LoRAs
4. Proceeds with generation

## 5. API Endpoints (Summary)

### Discovery Endpoints

- **GET /v2/loras** — List all models with LoRAs
- **GET /v2/loras/{model_id}** — List LoRAs for a model
- **GET /v2/loras/{model_id}/{lora_id}** — Get LoRA metadata + preview

### Upload & Registration Endpoints

- **POST /v2/loras/upload** — Upload LoRA file
- **POST /v2/loras/register** — Register uploaded LoRA to catalog
- **DELETE /v2/loras/{model_id}/{lora_id}** — Remove LoRA (admin)

### Generation with LoRAs

Existing endpoints accept new `loras` field:
- **POST /v1/text-to-video** (modified)
- **POST /v1/image-to-video** (modified)
- **POST /v1/audio-to-video** (modified)

## 6. Error Handling & Validation

### Invalid LoRA Scenarios

- **LoRA ID not found**: HTTP 404 with helpful message
- **Incompatible base model**: HTTP 400 — LoRA trained for different model
- **Corrupted file**: HTTP 400 — SafeTensors parse error
- **File too large**: HTTP 413 — exceeds max size (e.g., 500MB)
- **Invalid key structure**: HTTP 400 — doesn't follow expected LoRA format

### Quota Management

- Per-model LoRA limit: e.g., max 100 LoRAs per base model
- Total storage limit: e.g., max 5GB LoRA storage
- User upload limit: e.g., max 100 uploads per user per day (if auth added)

## 7. Backend Implementation Details

### LoRA Registry Class

```python
class LoRARegistry:
    """Manages LoRA discovery, loading, and caching."""

    def __init__(self, loras_dir: Path):
        self.loras_dir = loras_dir / "models"
        self.registry_path = loras_dir / "registry.json"
        self._registry: dict[str, dict[str, LoRAInfo]] = {}
        self._last_loaded = 0.0

    def refresh(self) -> None:
        """Force rebuild registry from disk."""

    def get_lora(self, model_id: str, lora_id: str) -> LoRAInfo | None:
        """Get LoRA metadata by ID."""

    def list_loras(self, model_id: str) -> list[LoRAInfo]:
        """List all LoRAs for a model."""

    def resolve_path(self, lora_info: LoRAInfo) -> Path:
        """Get file path to LoRA weights."""

    def upload_lora(self, upload_id: str, metadata: dict) -> LoRAInfo:
        """Register uploaded LoRA to catalog."""

    def delete_lora(self, model_id: str, lora_id: str) -> None:
        """Remove LoRA from disk and registry."""
```

### Integration with UploadStore

Reuse existing `UploadStore` pattern:
- New `LoRAUploadStore` extends or parallels `UploadStore`
- Handles LoRA-specific validation (SafeTensors format, key structure)
- Uses same UUID-based storage paradigm

## 8. Backward Compatibility

- Config-based LoRAs (hardcoded paths in `config.py`) continue working
- New dynamic LoRAs loaded via registry
- Inference code treats both identically: `LoraPathStrengthAndSDOps(path, strength, sd_ops)`
- No breaking changes to existing pipelines

## 9. Future Enhancements

- **LoRA marketplace**: Public registry syncing from external source
- **Version control**: Track LoRA updates, rollback support
- **Access control**: Per-user LoRA visibility (after auth system)
- **Merging**: Pre-merge multiple LoRAs for faster inference
- **Quantization**: Store LoRAs in FP8 for smaller files
- **Composition**: Define LoRA combinations (style + character = style+char)
