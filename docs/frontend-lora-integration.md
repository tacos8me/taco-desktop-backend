# Frontend LoRA Integration Guide

> ⚠️ **SUPERSEDED (2026-04-11)** — the canonical LoRA usage reference is now **[API.md § LoRAs](./API.md)** alongside the Flux LoRA (`lora: {id, strength}`) field documented on each image endpoint. This doc is preserved for historical context. For new integrations use [QUICKSTART.md](./QUICKSTART.md) and [API.md](./API.md).

> **Audience**: taco-desktop frontend team
> **Date**: 2026-03-09 (LTX) · 2026-04-09 (Flux 2 added)
> **Server**: `http://<host>:8090`

This document covers how to integrate LoRA adapter management and usage into the frontend. taco-backend supports **two separate LoRA systems**:

- **LTX video LoRAs** (sections 1–9 below): user-uploadable `.safetensors` files that modify LTX video generation. Managed via `/v1/loras` (list / upload / delete). Applied to `TextToVideoRequest`, `ImageToVideoRequest`, `AudioToVideoRequest`, `RetakeRequest`.
- **Flux 2 image LoRAs** (section 10): server-side folder-drop. Files are placed by an operator into `flux_loras/` on the server; **no upload UI**. Clients discover via `GET /v1/flux-loras` and apply to `TextToImageRequest`, `ImageToImageRequest`, `ImageEditRequest`.

Both systems share the same `{id, strength}` request shape for consistency, but have **separate ID namespaces** and **distinct endpoints**.

---

## Table of Contents

1. [TypeScript Definitions](#1-typescript-definitions)
2. [API Reference: LoRA Management](#2-api-reference-lora-management)
3. [API Reference: Using LoRAs in Generation](#3-api-reference-using-loras-in-generation)
4. [Settings Panel UI Flow](#4-settings-panel-ui-flow)
5. [Generation Flow](#5-generation-flow)
6. [Strength Slider UX](#6-strength-slider-ux)
7. [Upload Flow](#7-upload-flow)
8. [Error Handling](#8-error-handling)
9. [Integration Checklist](#9-integration-checklist)
10. [Flux 2 Image LoRAs (v1.1)](#10-flux-2-image-loras-v11)

---

## 1. TypeScript Definitions

```typescript
// --- LoRA management types ---

interface LoRAInfo {
  id: string;             // UUID hex (32 chars)
  name: string;
  filename: string;       // original upload filename
  base_model: string;     // e.g. "ltx-2.3"
  size_bytes: number;
  uploaded_at: string;    // ISO 8601
  description: string;
}

interface LoRAListResponse {
  loras: LoRAInfo[];
  count: number;
}

interface LoRADeleteResponse {
  deleted: boolean;
  id: string;
}

// --- LoRA reference in generation requests ---

interface LoRAInput {
  id: string;             // LoRA ID from GET /v1/loras
  strength: number;       // 0.0 - 2.0, default 1.0
}

// --- Updated generation request types ---
// These extend the existing request types with an optional `lora` field.

interface TextToVideoRequest {
  prompt: string;
  model: ModelName;
  resolution: Resolution;
  duration: number;
  fps: number;
  generate_audio?: boolean;
  camera_motion?: string | null;
  lora?: LoRAInput | null;          // NEW
}

interface ImageToVideoRequest {
  prompt: string;
  image_uri?: string | null;
  keyframes?: KeyframeInput[] | null;
  model: ModelName;
  resolution: Resolution;
  duration: number;
  fps: number;
  generate_audio?: boolean;
  lora?: LoRAInput | null;          // NEW
}

interface RetakeRequest {
  video_uri: string;
  start_time: number;
  duration: number;
  mode: RetakeMode;
  prompt?: string | null;
  lora?: LoRAInput | null;          // NEW
}
```

---

## 2. API Reference: LoRA Management

All endpoints require `Authorization: Bearer <key>` (same as every other endpoint except `/health`).

### 2.1 List LoRAs

```
GET /v1/loras
```

Returns all available LoRAs.

**curl:**

```bash
curl -s http://192.168.1.100:8090/v1/loras \
  -H "Authorization: Bearer $API_KEY"
```

**TypeScript:**

```typescript
async function listLoras(): Promise<LoRAListResponse> {
  const res = await fetch(`${API_BASE}/v1/loras`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
  });
  if (!res.ok) throw new Error(`Failed to list LoRAs: HTTP ${res.status}`);
  return res.json();
}
```

**Response 200:**

```json
{
  "loras": [
    {
      "id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
      "name": "Cinematic Style",
      "filename": "cinematic_style.safetensors",
      "base_model": "ltx-2.3",
      "size_bytes": 156000000,
      "uploaded_at": "2026-03-09T10:30:00Z",
      "description": "Makes videos look cinematic with enhanced contrast"
    }
  ],
  "count": 1
}
```

### 2.2 Upload LoRA

```
POST /v1/loras
Content-Type: multipart/form-data
```

Uploads a `.safetensors` file with metadata. Returns `201` on success.

**curl:**

```bash
curl -s http://192.168.1.100:8090/v1/loras \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@cinematic_style.safetensors" \
  -F "name=Cinematic Style" \
  -F "description=Makes videos look cinematic" \
  -F "base_model=ltx-2.3"
```

**TypeScript:**

```typescript
async function uploadLora(
  file: File,
  name: string,
  description: string = "",
  baseModel: string = "ltx-2.3"
): Promise<LoRAInfo> {
  const form = new FormData();
  form.append("file", file);
  form.append("name", name);
  form.append("description", description);
  form.append("base_model", baseModel);

  const res = await fetch(`${API_BASE}/v1/loras`, {
    method: "POST",
    headers: { Authorization: `Bearer ${API_KEY}` },
    body: form,
  });

  if (res.status === 400) {
    const err = await res.json();
    throw new Error(err.message ?? "Invalid LoRA file");
  }
  if (res.status === 413) throw new Error("File exceeds 500MB limit");
  if (res.status === 422) {
    const err = await res.json();
    throw new Error(err.detail?.[0]?.msg ?? "Validation error");
  }
  if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`);

  return res.json();
}
```

**Form fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | binary | yes | -- | `.safetensors` file, max 500MB |
| `name` | string | yes | -- | Display name, max 200 chars |
| `description` | string | no | `""` | Description, max 2000 chars |
| `base_model` | string | no | `"ltx-2.3"` | Base model compatibility |

**Response 201:** Returns a `LoRAInfo` object (same shape as list items).

**Important:** Do NOT set `Content-Type: multipart/form-data` manually in fetch -- the browser sets it automatically with the correct boundary when you pass a `FormData` body.

### 2.3 Delete LoRA

```
DELETE /v1/loras/{lora_id}
```

Removes a LoRA from the server.

**curl:**

```bash
curl -s -X DELETE http://192.168.1.100:8090/v1/loras/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6 \
  -H "Authorization: Bearer $API_KEY"
```

**TypeScript:**

```typescript
async function deleteLora(loraId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/loras/${loraId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${API_KEY}` },
  });
  if (res.status === 404) throw new Error("LoRA not found");
  if (!res.ok) throw new Error(`Delete failed: HTTP ${res.status}`);
}
```

**Response 200:**

```json
{ "deleted": true, "id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6" }
```

---

## 3. API Reference: Using LoRAs in Generation

Add the optional `lora` field to any LTX video generation request (text-to-video, image-to-video, retake). The field is the same across all three endpoints.

**Only one LoRA can be applied per request.** This is by design -- multiple user LoRAs would compound with preset LoRAs used by pro and hq pipelines, making behavior unpredictable.

### Request field

```json
{
  "lora": {
    "id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    "strength": 0.8
  }
}
```

| Field | Type | Required | Default | Range | Description |
|---|---|---|---|---|---|
| `id` | string | yes | -- | -- | LoRA ID from `GET /v1/loras` |
| `strength` | number | no | `1.0` | `0.0` - `2.0` | How strongly the LoRA affects generation |

### Example: text-to-video with LoRA (v2 async)

**curl:**

```bash
curl -s http://192.168.1.100:8090/v2/text-to-video \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cinematic aerial shot of mountains at sunrise",
    "model": "ltx-2-3-fast",
    "resolution": "1920x1080",
    "duration": 5,
    "fps": 24,
    "generate_audio": false,
    "lora": {
      "id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
      "strength": 0.8
    }
  }'
```

**TypeScript:**

```typescript
const { job_id } = await submitJob("text-to-video", {
  prompt: "A cinematic aerial shot of mountains at sunrise",
  model: "ltx-2-3-fast",
  resolution: "1920x1080",
  duration: 5,
  fps: 24,
  generate_audio: false,
  lora: {
    id: selectedLora.id,
    strength: loraStrength,
  },
});
```

### Example: image-to-video with LoRA

```typescript
const { job_id } = await submitJob("image-to-video", {
  prompt: "Smooth camera pan across the landscape",
  image_uri: "storage://uploaded-image-id",
  model: "ltx-2-3-pro",
  resolution: "1920x1080",
  duration: 3,
  fps: 24,
  lora: {
    id: selectedLora.id,
    strength: 1.0,
  },
});
```

### Example: retake with LoRA

```typescript
const { job_id } = await submitJob("retake", {
  video_uri: "storage://original-video-id",
  start_time: 2.0,
  duration: 1.5,
  mode: "replace_video_only",
  prompt: "An explosion with cinematic lighting",
  lora: {
    id: selectedLora.id,
    strength: 0.6,
  },
});
```

### Omitting the LoRA field

To generate without a LoRA, simply omit the `lora` field or set it to `null`. Existing requests without `lora` continue to work unchanged.

```typescript
// No LoRA -- works exactly as before
const { job_id } = await submitJob("text-to-video", {
  prompt: "A cat walking on a beach",
  model: "ltx-2-3-fast",
  resolution: "1920x1080",
  duration: 5,
  fps: 24,
});
```

---

## 4. Settings Panel UI Flow

Build a LoRA management section in the app settings. This lets users discover, upload, and remove LoRAs independently of generation.

### Recommended layout

```
Settings > LoRAs
┌──────────────────────────────────────────────────┐
│  LoRA Adapters                      [Upload New] │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │ Cinematic Style               148.7 MB     │  │
│  │ Makes videos look cinematic                │  │
│  │ Base: ltx-2.3  •  Uploaded Mar 9, 2026     │  │
│  │                               [Delete]     │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │ Anime Style                    92.3 MB     │  │
│  │ Anime-inspired visual style                │  │
│  │ Base: ltx-2.3  •  Uploaded Mar 8, 2026     │  │
│  │                               [Delete]     │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  No more LoRAs.                                  │
└──────────────────────────────────────────────────┘
```

### Load flow

1. On mount, call `GET /v1/loras`
2. Render each `LoRAInfo` as a card showing `name`, `description`, formatted `size_bytes`, `base_model`, and formatted `uploaded_at`
3. Show a loading spinner while fetching
4. Show "No LoRAs uploaded yet" empty state with an upload prompt

### Delete flow

1. User clicks Delete on a LoRA card
2. Show confirmation dialog: "Delete {name}? This cannot be undone."
3. On confirm, call `DELETE /v1/loras/{id}`
4. Remove the card from the list (optimistic or after response)
5. If the deleted LoRA is currently selected for generation, clear the selection

### Formatting helpers

```typescript
function formatFileSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  return `${(bytes / 1_000).toFixed(1)} KB`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}
```

---

## 5. Generation Flow

Integrate LoRA selection into the generation form (text-to-video, image-to-video, retake).

### UI placement

Add a LoRA selector below the model selector in the generation form:

```
Model:    [ltx-2-3-fast ▼]
LoRA:     [None ▼]              ← dropdown of available LoRAs
Strength: [========|--] 0.80    ← slider, visible only when a LoRA is selected
```

### LoRA selector dropdown

1. Populate from `GET /v1/loras` (cache the response; refresh when the settings panel changes)
2. First option: "None" (no LoRA applied)
3. Remaining options: LoRA names from the list
4. Show `base_model` as a subtitle or badge if multiple base models exist

### Building the request

```typescript
function buildGenerationRequest(
  formValues: FormValues,
  selectedLora: LoRAInfo | null,
  loraStrength: number
): TextToVideoRequest {
  const request: TextToVideoRequest = {
    prompt: formValues.prompt,
    model: formValues.model,
    resolution: formValues.resolution,
    duration: formValues.duration,
    fps: formValues.fps,
    generate_audio: formValues.generateAudio,
  };

  if (selectedLora) {
    request.lora = {
      id: selectedLora.id,
      strength: loraStrength,
    };
  }

  return request;
}
```

### LoRA state management

```typescript
// Component state
const [loras, setLoras] = useState<LoRAInfo[]>([]);
const [selectedLoraId, setSelectedLoraId] = useState<string | null>(null);
const [loraStrength, setLoraStrength] = useState(1.0);

// Derived
const selectedLora = loras.find((l) => l.id === selectedLoraId) ?? null;
```

---

## 6. Strength Slider UX

The strength slider controls how strongly the LoRA influences generation. Only show it when a LoRA is selected.

### Recommended configuration

| Parameter | Value |
|---|---|
| Min | `0.0` |
| Max | `2.0` |
| Default | `1.0` |
| Step | `0.05` |
| Display | Show numeric value to 2 decimal places |

### What strength values mean

| Range | Effect | Notes |
|---|---|---|
| `0.0` | No effect | LoRA is loaded but has zero influence. Useful for A/B comparison. |
| `0.1 - 0.5` | Subtle | Light stylistic influence. Good starting point for strong LoRAs. |
| `0.5 - 1.0` | Moderate to full | Most LoRAs are designed for this range. `1.0` is the trained default. |
| `1.0 - 1.5` | Amplified | Exaggerated effect. May introduce visual artifacts. |
| `1.5 - 2.0` | Extreme | Strong distortion likely. Only useful for intentional stylization. |

### Recommended presets

Offer quick-select buttons or labeled ticks alongside the slider:

```
Subtle [0.3]   Normal [1.0]   Strong [1.5]
```

### Slider component example

```typescript
<div className="flex items-center gap-3">
  <label className="text-sm text-zinc-400 w-16">Strength</label>
  <input
    type="range"
    min={0}
    max={2}
    step={0.05}
    value={loraStrength}
    onChange={(e) => setLoraStrength(parseFloat(e.target.value))}
    className="flex-1"
  />
  <span className="text-sm text-zinc-300 w-10 text-right">
    {loraStrength.toFixed(2)}
  </span>
</div>
```

---

## 7. Upload Flow

LoRA upload uses `multipart/form-data` to send the file and metadata in a single request.

### UI flow

1. User clicks "Upload New" in the settings panel
2. Show an upload dialog/modal:
   - **File input**: Accept `.safetensors` files only
   - **Name**: Required text field (max 200 chars)
   - **Description**: Optional textarea (max 2000 chars)
   - **Base Model**: Dropdown, default `"ltx-2.3"` (currently the only option)
3. Client-side validation before upload:
   - File extension must be `.safetensors`
   - File size must be <= 500MB
   - Name must not be empty
4. Show upload progress (use `XMLHttpRequest` or a progress-tracking fetch wrapper)
5. On success (201), close the dialog and refresh the LoRA list
6. On error, show the error message in the dialog

### Upload with progress tracking

```typescript
async function uploadLoraWithProgress(
  file: File,
  name: string,
  description: string,
  onProgress: (percent: number) => void
): Promise<LoRAInfo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/v1/loras`);
    xhr.setRequestHeader("Authorization", `Bearer ${API_KEY}`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status === 201) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.message ?? err.detail?.[0]?.msg ?? `HTTP ${xhr.status}`));
        } catch {
          reject(new Error(`Upload failed: HTTP ${xhr.status}`));
        }
      }
    };

    xhr.onerror = () => reject(new Error("Network error during upload"));

    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    form.append("description", description);
    form.append("base_model", "ltx-2.3");
    xhr.send(form);
  });
}
```

### Client-side validation

```typescript
function validateLoraFile(file: File): string | null {
  if (!file.name.endsWith(".safetensors")) {
    return "File must be a .safetensors file";
  }
  if (file.size > 500 * 1024 * 1024) {
    return `File is too large (${formatFileSize(file.size)}). Maximum is 500 MB.`;
  }
  return null; // valid
}
```

### File input configuration

```html
<input
  type="file"
  accept=".safetensors"
  onChange={handleFileSelect}
/>
```

---

## 8. Error Handling

### Upload errors

| HTTP Status | Cause | User message |
|---|---|---|
| `400` | Not a `.safetensors` file, or file doesn't contain LoRA weights | Show server error message directly |
| `413` | File exceeds 500MB | "File is too large. Maximum size is 500 MB." |
| `422` | Missing or invalid `name` field | "Please provide a name for the LoRA." |
| `401` | Invalid API key | Show auth error (same as all other endpoints) |

### Delete errors

| HTTP Status | Cause | User message |
|---|---|---|
| `404` | LoRA already deleted or doesn't exist | Remove from list silently, or show "LoRA not found" |

### Generation errors (LoRA-related)

| HTTP Status | Cause | User message |
|---|---|---|
| `404` | LoRA ID in request doesn't exist (deleted between selection and generation) | "Selected LoRA was not found. It may have been deleted." Clear selection and refresh the LoRA list. |
| `500` | LoRA incompatible with model architecture (mismatched weights) | "Generation failed. The LoRA may be incompatible with this model." |

### Handling deleted LoRAs gracefully

A LoRA could be deleted between when the user selects it and when they submit generation. Handle this:

```typescript
async function submitWithLoraFallback(
  endpoint: string,
  body: Record<string, unknown>
): Promise<JobSubmitResponse> {
  try {
    return await submitJob(endpoint, body);
  } catch (err) {
    if (err instanceof Error && err.message.includes("LoRA not found")) {
      // Refresh LoRA list, clear selection, notify user
      await refreshLoras();
      setSelectedLoraId(null);
      showNotification("Selected LoRA was deleted. Please select another or generate without one.");
      throw err;
    }
    throw err;
  }
}
```

### Model compatibility note

LoRAs are currently only supported on **LTX models** (`ltx-2-3-fast`, `ltx-2-3-pro`, `ltx-2-3-hq`). They are **not** supported on Flux models. If the user selects a Flux model, hide or disable the LoRA selector.

```typescript
const supportsLora = model.startsWith("ltx-");
```

---

## 9. Integration Checklist

### LoRA management (settings panel)

- [ ] Add LoRA section to settings
- [ ] Fetch and display LoRA list from `GET /v1/loras`
- [ ] Show name, description, file size, base model, upload date per LoRA
- [ ] Implement upload dialog with file input (`.safetensors`), name, description fields
- [ ] Client-side validation: file extension, file size <= 500MB, name required
- [ ] Upload via `POST /v1/loras` multipart form data
- [ ] Show upload progress bar
- [ ] Handle upload errors (400, 413, 422)
- [ ] Implement delete with confirmation dialog via `DELETE /v1/loras/{id}`
- [ ] Refresh LoRA list after upload or delete

### Generation form

- [ ] Add LoRA dropdown to text-to-video, image-to-video, and retake forms
- [ ] Populate dropdown from cached LoRA list
- [ ] Include "None" as the default/first option
- [ ] Show strength slider when a LoRA is selected (range 0.0-2.0, step 0.05, default 1.0)
- [ ] Display numeric strength value next to slider
- [ ] Include `lora: { id, strength }` in request body when a LoRA is selected
- [ ] Omit `lora` field when no LoRA is selected
- [ ] For Flux image forms: use the **separate** `GET /v1/flux-loras` endpoint (see section 10). LTX and Flux LoRAs are distinct namespaces — do not mix the dropdowns.

### Error handling

- [ ] Handle 404 on generation (LoRA deleted) -- clear selection, refresh list, notify user
- [ ] Handle 500 on generation with LoRA -- suggest incompatibility
- [ ] Clear LoRA selection if the selected LoRA is deleted from settings

### Types

- [ ] Add `LoRAInfo`, `LoRAListResponse`, `LoRADeleteResponse`, `LoRAInput` interfaces
- [ ] Add optional `lora?: LoRAInput | null` to `TextToVideoRequest`, `ImageToVideoRequest`, `RetakeRequest`

---

## 10. Flux 2 Image LoRAs (v1.1)

Flux image LoRAs live in a **separate folder-drop system**. Operators place `.safetensors` files into `/mnt/nvme-1/servers/taco-backend/flux_loras/` on the server; the frontend has **no upload or delete UI**.

### 10.1 TypeScript Definitions

```typescript
// Flux LoRA metadata returned by GET /v1/flux-loras
interface FluxLoRAInfo {
  id: string;                // slug from filename stem, e.g. "my-style-v2"
  name: string;              // display name (sidecar .json or filename stem)
  filename: string;          // on-disk filename
  size_bytes: number;
  model_compat: string[];    // e.g. ["flux2-dev"] or ["flux2-dev","flux2-klein"]
  description: string;       // may be ""
  trigger_word: string | null;
}

interface FluxLoRAListResponse {
  loras: FluxLoRAInfo[];
  count: number;
}

// The `lora` field on Flux request types reuses the same LoRAInput shape
// as LTX, but the ID namespace is separate (different registry).
interface TextToImageRequest {
  prompt: string;
  model?: "flux2-dev" | "flux2-klein";
  width?: number;
  height?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  seed?: number | null;
  turbo?: boolean;
  lora?: LoRAInput | null;   // NEW in v1.1
}

// Same lora field added to ImageToImageRequest and ImageEditRequest.
```

### 10.2 Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/flux-loras` | GET | Yes | List discovered Flux LoRAs |
| `/v1/flux-loras/rescan` | POST | Yes | Re-scan the folder (use after an operator drops new files) |

**Note:** there is intentionally **no** upload or delete endpoint. Files are managed on the server side via `cp`/`rm`. A future version may add an upload UI; for now the frontend should treat the LoRA list as read-only.

### 10.3 UI Flow

- **Discovery**: on app init (or when the user opens the Flux image form), call `GET /v1/flux-loras`. Cache the list; offer a manual "Refresh" button that calls `POST /v1/flux-loras/rescan` followed by a fresh `GET`.
- **Selection**: add a LoRA dropdown to text-to-image, image-to-image, and image-edit forms. Default to "None". Show `name` (fall back to `id`), `description`, and `trigger_word` if present.
- **Model compatibility**: `model_compat` is advisory — the backend does **not** enforce it. Prefer to filter the dropdown client-side by the currently selected `model` (`flux2-dev` / `flux2-klein`) and grey out incompatible LoRAs with a tooltip. This prevents obvious mistakes like loading a Klein-only LoRA on Dev.
- **Strength**: same `0.0 – 2.0` slider as LTX, default `1.0`.
- **Empty state**: if `count === 0`, show "No Flux LoRAs available. Ask your server admin to drop `.safetensors` files into `flux_loras/`." (no upload button).

### 10.4 Generation Flow

```typescript
const body: TextToImageRequest = {
  prompt: "a cyberpunk cat",
  model: "flux2-dev",
  lora: selectedFluxLora
    ? { id: selectedFluxLora.id, strength: loraStrength }
    : undefined,
  turbo: turboEnabled,
  seed: userSeed,
};

const resp = await fetch(`${SERVER}/v1/text-to-image`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
```

**Latency note (v1.1.1 + v1.1.4 updates):** the first request with a new `(model, lora_id)` **pair** triggers a ~30–60 s pipeline reload on the server (full bf16 load + CPU offload hook setup on Dev). Subsequent requests with the **same** pair are cached. **Changing `strength` is free** — it's applied via a runtime `set_adapters([...], [strength])` call, no reload. The strength slider can be scrubbed freely without blocking the UI. Only `lora_id` changes, model switches (`flux2-dev` ↔ `flux2-klein`), or adding/removing the LoRA field trigger the full reload. Show the "Loading LoRA…" indicator only on `(model, lora_id)` changes, NOT on strength-only changes.

**v1.1.4 cross-type consideration**: if the user submits a **video** request between two Flux image requests, the first Flux request after the video pays an additional **~3 s LTX eviction** cost on top of any LoRA reload (Flux 2 and LTX now share `cuda:0` and are auto-swapped on dispatch — see `docs/frontend-api-changes.md` section 9). Pure image-only workloads are unaffected; strength-slider scrubbing remains O(ms) free regardless.

### 10.5 Error Handling

| Status | Cause | UI Action |
|--------|-------|-----------|
| `404` `"Flux LoRA not found: {id}"` | `lora.id` not in registry (stale cache, or operator `rm`'d the file) | Clear selection, refetch `GET /v1/flux-loras`, notify user |
| `500` during generation with LoRA | LoRA incompatible with model (e.g., malformed weights) | Surface error, suggest removing the LoRA or trying the other model |
| `401` on `/v1/flux-loras*` | Missing/invalid API key | Redirect to auth flow |

### 10.6 Integration Checklist (Flux)

- [ ] Add `FluxLoRAInfo`, `FluxLoRAListResponse` TypeScript interfaces
- [ ] Add `lora?: LoRAInput | null` to `TextToImageRequest`, `ImageToImageRequest`, `ImageEditRequest`
- [ ] Fetch Flux LoRAs via `GET /v1/flux-loras` on form open (cache between sessions is fine)
- [ ] Add LoRA dropdown to t2i, i2i, edit forms
- [ ] Add "Refresh" button that calls `POST /v1/flux-loras/rescan`
- [ ] Client-side filter by `model_compat` against the selected model
- [ ] Show strength slider (0.0–2.0, step 0.05, default 1.0)
- [ ] Show "Loading LoRA…" indicator on first request with a new `(model, lora, strength)` combo
- [ ] Handle 404 by clearing selection and refetching the list
- [ ] Empty-state message explaining folder-drop (no upload UI)
