# Gallery / Review Webapp — Integration Guide

Everything a fresh gallery/review project needs to bootstrap against `taco-backend` (v1.13.0). One-stop reference for browsing, displaying, inspecting, deleting, exporting, and curating the full output of every generation pipeline (image, video, music, edits, compositions).

**Base URL**: `https://api.noodlefinger.io` (Cloudflare-proxied)
**Local dev**: `http://localhost:8090`
**Auth**: every endpoint except `/health` requires `Authorization: Bearer <api-key>`. Keys are scoped — every list/read/delete is implicitly filtered by `sha256(api_key)`. Two clients with different keys see disjoint galleries.

---

## 1. Pillar endpoints

The gallery is built on **four** data sources. Most webapps will only need the first two.

| # | Source | Purpose | Scope |
|---|--------|---------|-------|
| 1 | `/v2/history` | Auto-saved record of every completed v2 job (image, video, music, edit) — last 30 days | per-key |
| 2 | `/v1/approved-images` | Curated list of images explicitly "approved" by a client (noodle-i → noodle-v handoff) | per-key |
| 3 | `/v2/compositions` | Saved MusicVideo timelines (clip arrangement + audio overlay) | per-key |
| 4 | `/v2/jobs/{id}` | Live job state (queued/processing/completed) — for *in-flight* gallery cards | per-key |

Everything below is built on top of these.

---

## 2. Authentication

```
Authorization: Bearer <api-key>
```

Keys are configured server-side in `.api_keys` (one per line). When the file is empty, auth is disabled (dev-mode).

### SSE / EventSource (browsers)

`EventSource` cannot set custom headers. For SSE endpoints (`/v2/jobs/{id}/stream`, `/v1/approved-images/events`), exchange your bearer for a 5-minute disposable token:

```http
POST /v1/sse-token
Authorization: Bearer <api-key>

→ 200 { "token": "<urlsafe-32>", "expires_in": 300 }
```

Then connect: `new EventSource("/v2/jobs/abc/stream?token=<token>")`.

### 2.1 Frontend auth pattern (mirror what the dashboard already does)

The taco-backend dashboard (`dashboard.html`) uses a deliberately minimal pattern that the gallery webapp should copy verbatim — there is **no OAuth, no session cookie, no login screen** on the backend itself. The bearer key IS the identity.

**Storage**: `localStorage` under a single key. Password-typed input prevents shoulder-surfing; not encrypted at rest (browsers don't expose a real secret store, and keys are revocable server-side via `.api_keys`).

```html
<label for="apiKey">API Key</label>
<input type="password" id="apiKey"
       placeholder="Bearer token" autocomplete="off">
```

```javascript
// Restore on load
const apiKeyInput = document.getElementById("apiKey");
apiKeyInput.value = localStorage.getItem("noodlegal_api_key") || "";

// Persist on every keystroke (no submit button needed)
apiKeyInput.addEventListener("input", () => {
  localStorage.setItem("noodlegal_api_key", apiKeyInput.value);
});

function getAuthHeaders() {
  const key = apiKeyInput.value.trim();
  if (!key) return {};
  return { "Authorization": "Bearer " + key };
}
```

**Use a single fetch wrapper** that injects the header for every request:

```javascript
async function api(method, path, body) {
  const headers = { ...getAuthHeaders() };
  const opts = { method, headers };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  if (resp.status === 401) {
    // Clear and prompt — the user's key is wrong/revoked
    toast("Invalid API key", "err");
    apiKeyInput.focus();
    throw new Error("unauthorized");
  }
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.message || `HTTP ${resp.status}`);
  }
  return resp;
}
```

**SSE token cache** — call `/v1/sse-token` lazily, cache for 4 minutes (one minute under the 5-min server expiry):

```javascript
let _sseToken = null, _sseTokenExp = 0;
async function getSseToken() {
  if (_sseToken && Date.now() < _sseTokenExp) return _sseToken;
  const r = await api("POST", "/v1/sse-token");
  const data = await r.json();
  _sseToken = data.token;
  _sseTokenExp = Date.now() + 4 * 60 * 1000;
  return _sseToken;
}

async function streamJob(jobId, onEvent) {
  const token = await getSseToken();
  const es = new EventSource(`/v2/jobs/${jobId}/stream?token=${token}`);
  es.onmessage = (e) => onEvent(JSON.parse(e.data));
  return es;
}
```

**Image / video element auth** — `<img>` and `<video>` tags can't set headers. Two options:

1. **Same-origin / Cloudflare proxy** (recommended) — the gallery webapp is served behind the same auth gate (e.g. Cloudflare Access SSO, or a reverse proxy that forwards bearer headers). Tags work directly: `<img src="/v2/history/abc/thumbnail">`.
2. **Blob URL fallback** — when bearer must travel in JS:
   ```javascript
   async function authedBlobUrl(path) {
     const r = await api("GET", path);
     const blob = await r.blob();
     return URL.createObjectURL(blob);  // remember to revokeObjectURL on unmount
   }
   ```
   This costs an extra fetch round-trip and bypasses HTTP caching — use only when same-origin proxy isn't possible.

**Logout / key rotation** — there is no server-side logout. To "log out" the gallery:

```javascript
function logout() {
  localStorage.removeItem("noodlegal_api_key");
  apiKeyInput.value = "";
  _sseToken = null;
  // optionally: location.reload();
}
```

Server-side revocation: edit `.api_keys` on the box (one key per line, blank-line-separated reload not needed — the file is read on every request). When a key is removed, all clients holding it start receiving 401.

**Multi-tenant note**: gallery views are key-scoped on the backend (sha256 of bearer keyed against history rows). To run a "team gallery" with multiple users seeing the same content, all users must share the same key. The backend has no concept of users-vs-keys.

**Defense in depth (recommended for production)**:

| Layer | Purpose |
|---|---|
| Cloudflare Access (SSO/email/IP allow-list) in front of `api.noodlefinger.io` | Blocks unauthenticated traffic before it reaches the backend |
| HTTPS-only (set by CF) | Prevents bearer-token sniffing |
| `Content-Security-Policy: default-src 'self'` on the gallery HTML | Defends against XSS exfiltrating localStorage |
| Short-lived rotating keys (manual today; cron-rotate `.api_keys`) | Limits blast radius of leaks |
| `httpOnly` cookie + server-side proxy *(future)* | Removes JS access to bearer entirely; gallery would call its own `/api/*` BFF |

The dashboard ships the minimal pattern (item 1 alone). Mirror that for v0; layer in CSP + cookie-BFF only when the gallery goes public.

---

## 3. History API (the gallery backbone)

### 3.1 List

```http
GET /v2/history?limit=50&offset=0&type=<filter>
```

| Query | Default | Notes |
|---|---|---|
| `limit` | 50 | Capped server-side at 200 |
| `offset` | 0 | Naive pagination — combine with `created_at` for stable cursors |
| `type` | (none) | Filter shortcuts: `image`, `video`, `music`, OR exact `JobType` (see §10) |

**Response — array of slim cards** (gallery grid is built directly from this):

```jsonc
[
  {
    "id": "abc123...",                          // generation_id (= job_id)
    "prompt": "a cat in space",
    "model": "flux2-klein",
    "width": 1024,
    "height": 1024,
    "turbo": false,
    "status": "completed",                      // "completed" | "failed"
    "created_at": 1776796257.1,                 // unix epoch seconds (float)
    "error": null,                              // string when status="failed"
    "thumbnail_url": "/v2/history/abc/thumbnail", // 256px JPEG, present iff result was decodable
    "image_url": "/v2/history/abc/image"          // full-resolution media, present iff completed
  },
  ...
]
```

The `thumbnail_url` and `image_url` keys are **omitted** (not null) when no thumbnail/result exists. Render a placeholder when missing.

### 3.2 Detail (full record)

```http
GET /v2/history/{generation_id}
```

```jsonc
{
  "id": "...",
  "job_type": "image-to-video",                 // see §10
  "prompt": "raw user prompt",
  "enhanced_prompt": "LTX-rewritten ...",       // null if enhance_prompt was off / Flux job
  "model": "ltx-2-3-fast",
  "width": 1280,
  "height": 720,
  "seed": 42,                                   // captured per-job
  "turbo": true,
  "status": "completed",
  "created_at": 1776796257.1,
  "completed_at": 1776796340.7,
  "error": null,
  "params": { /* raw request body, exact Pydantic dump */ },
  "gen_config": { /* LTX gen-config snapshot OR {turbo_steps,turbo_guidance} for Flux-turbo. null otherwise */ },
  "result_url": "/v2/history/.../image",
  "thumbnail_url": "/v2/history/.../thumbnail"
}
```

**Reproducibility**: `params + gen_config + seed + model` is a complete recipe. POSTing `params` back to the same `/v2/<job_type>` endpoint reproduces the result bit-for-bit (modulo non-determinism in Modal/RunPod backends).

**Truncation**: `params` and `gen_config` are capped at 100 KB and 50 KB respectively. Oversized payloads are stored as:

```jsonc
{ "__truncated__": true, "original_bytes": 312000, "preview": "..." }
```

Clients should check for `__truncated__` before keying off arbitrary fields.

### 3.3 Media

```http
GET /v2/history/{id}/image       # full resolution
GET /v2/history/{id}/thumbnail   # 256px JPEG
```

- Image jobs serve `image/webp` (Q95).
- Video jobs serve `video/mp4` (H.264, CRF 18, fast preset).
- Music jobs serve `audio/wav` or `audio/mp3` per ACE.
- Thumbnails are always `image/jpeg` (Q70, content-addressed, immutable).

**HTTP caching**: results are immutable for 30 days (`max-age=2592000, immutable`). Thumbnails for 1 year. ETag/304 supported. Browsers will refetch automatically when the URL stops resolving (after retention sweep).

### 3.4 Delete

```http
DELETE /v2/history/{id}
→ 200 { "ok": true }
→ 404 (also returned when the row exists but belongs to a different key — no existence oracle)
```

Removes the row, the result MP4/WEBP, and the thumbnail. Best-effort on the filesystem; 404 returned only if the SQLite row was missing.

### 3.5 Retention

Server runs `cleanup(max_age_days=30)` on a cron. Anything older than 30 days vanishes — both the row and the on-disk media. Plan UX around this (allow user-driven re-export to permanent storage if needed).

---

## 4. Approved Images

A client-curated subset of generated images. Use this for "reviewer accepted these" gallery views, or as the input feed for a downstream tool (e.g. noodle-v consuming approved noodle-i renders).

### 4.1 Approve

```http
POST /v1/approved-images
Content-Type: application/json

{
  "image_uri": "storage://<uuid>",   // required — usually copy from history.params or job.result_storage_uri
  "prompt": "...",                   // optional — copy from history.prompt
  "model": "flux2-klein",
  "width": 1024,
  "height": 1024
}
→ 201 { "id": "<16-hex>", "status": "approved" }
```

The server stores `sha256(api_key)` alongside each entry — the bearer key is never persisted in plaintext.

### 4.2 List

```http
GET /v1/approved-images?limit=50&offset=0

→ [
  {
    "id": "16-hex",
    "image_uri": "storage://<uuid>",
    "prompt": "...",
    "model": "...",
    "width": 1024,
    "height": 1024,
    "created_at": 1776796257.1,
    "image_url": "/v1/approved-images/{id}/file"
  },
  ...
]
```

### 4.3 Fetch image

```http
GET /v1/approved-images/{id}/file
→ image/webp (immutable, max-age=2592000)
```

### 4.4 Live updates (SSE)

```http
GET /v1/approved-images/events?token=<sse-token>
```

Server emits one `data: <json>` line per newly-approved image, polling the manifest mtime every 2 s. Same payload shape as the `list` endpoint.

> No DELETE endpoint exists yet (v1.13.0). Manage curation by adding-only or by mutating `approved-images/manifest.json` server-side.

---

## 5. Live job tracking (for "currently rendering" tiles)

A polished gallery should show queued + in-flight jobs alongside completed ones.

### 5.1 Single job

```http
GET /v2/jobs/{job_id}

→ {
  "job_id": "abc",
  "status": "queued" | "processing" | "completed" | "failed" | "cancelled",
  "type": "image-to-video",
  "progress": 0.45,                          // 0..1, 0..0.90 = denoising; 0.90..1 = post-denoise
  "phase": "denoising" | "decoding" | "encoding" | "saving" | null,
  "queue_position": 3,                       // null unless QUEUED
  "error": { "code": "...", "message": "..." } | null,
  "result_url": "/v2/jobs/abc/result",       // null until COMPLETED
  "result_storage_uri": "storage://<uuid>",
  "result_media_type": "video/mp4"
}
```

### 5.2 Live preview (in-flight tile)

```http
GET /v2/jobs/{job_id}/preview
→ 200 image/jpeg              // cached preview OR first-frame fallback
→ 204                         // no preview available yet (still encoding text)
```

Four code paths inside the server (cached → on-disk thumb → lazy PyAV decode → 204). Polling is cheap; 1-2 s cadence is fine.

### 5.3 SSE stream (zero-poll)

Preferred for video — replaces 240 GETs per 2-min job with one EventSource:

```http
GET /v2/jobs/{job_id}/stream?token=<sse-token>
```

Each event is a `data: {...}\n\n` line with the same JSON shape as `GET /v2/jobs/{id}`. Stream closes itself on terminal status (completed/failed/cancelled). Keepalive comments every 15 s to prevent proxy timeouts.

### 5.4 Result download

```http
GET /v2/jobs/{job_id}/result
→ FileResponse, Content-Type matches result_media_type, Cache-Control: no-store
→ 409 if not COMPLETED
```

(For *historical* gallery views, prefer `/v2/history/{id}/image` — it's cached and survives Job-store eviction.)

### 5.5 Cancel

```http
DELETE /v2/jobs/{job_id}
→ 200 { "job_id": "...", "status": "cancelled" }
→ 409 if already terminal
```

Cancellation unwinds the denoising loop on the next sigma step (1-3 s latency).

---

## 6. Compositions (saved MusicVideo timelines)

Compositions are saved arrangements of multiple history clips with optional audio overlay. Useful for a "projects" tab in your gallery webapp.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v2/compositions` | Create new |
| `GET` | `/v2/compositions?limit=&offset=` | List per-key |
| `GET` | `/v2/compositions/{id}` | Read one |
| `PUT` | `/v2/compositions/{id}` | Update (full replace of `name` + `data`) |
| `DELETE` | `/v2/compositions/{id}` | Remove |
| `POST` | `/v2/compositions/{id}/export` | Render to MP4 (returns a job-id; poll via §5) |

### Body shape (create / update)

```jsonc
{
  "name": "My MV",
  "data": {
    "clips": [
      { "history_id": "...", "tailTrimFrames": 9, "audioDurationSec": 4.2 },
      ...
    ],
    "audio_uri": "storage://<uuid>",      // optional
    "transitions": "xfade" | "cut",
    ...                                    // FE-defined fields persist as-is
  }
}
```

Compositions are opaque-data on the server — the FE owns the schema for `data`. The export pipeline reads only `clips[].history_id`, `tailTrimFrames`, `audioDurationSec`, and `audio_uri`.

---

## 7. Generic upload / download

For when the gallery accepts user-uploaded comparison images, audio, etc.

```http
PUT /uploads/put/{uuid4-hex}
Content-Type: <correct mime — magic-byte verified>
<binary body, ≤ MAX_UPLOAD_BYTES>
→ 201
```

```http
GET /uploads/get/{uuid4-hex}
→ FileResponse with sniffed media_type
```

The `upload_id` IS the capability — bearer + ID required. Uploads count against `PER_KEY_UPLOAD_BYTES_PER_DAY` quota (returned as 429 on exceed).

---

## 8. URI conventions

Three URI schemes appear in payloads:

| Scheme | Resolves to | Where |
|---|---|---|
| `storage://<uuid>` | A file in `UPLOAD_DIR`, served via `/uploads/get/<uuid>` (or `/v2/history/{id}/image` for results) | result_uri, image_uri, audio_uri, video_uri, segment_uri, keyframe.image_uri |
| `thumb://<id>` | A file in `THUMBNAIL_DIR`, served via `/v2/history/{id}/thumbnail` | thumbnail_uri (internal — clients use the `_url` fields) |
| `/v2/history/.../image` etc. | Concrete URL — use directly in `<img src>`, `<video src>`, `fetch()` | history list/detail responses |

> Always prefer the `*_url` fields the server returns. Don't hand-construct URLs from `storage://` URIs — the server may rewrite them later (e.g. signed URLs, CDN).

---

## 9. Identifier rules

- `generation_id` (history) === `job_id` (live job) — they're the same UUID. A history row appears the moment a v2 job completes (via `worker_loop`'s fire-and-forget `history.save`).
- `composition.id`, `approved-image.id` — independent ID spaces, NOT job IDs.
- All IDs are opaque strings — don't parse, don't sort lexicographically (use `created_at` for ordering).

---

## 10. JobType enum (filter values)

```python
class JobType(StrEnum):
    TEXT_TO_VIDEO     = "text-to-video"
    IMAGE_TO_VIDEO    = "image-to-video"
    AUDIO_TO_VIDEO    = "audio-to-video"
    RETAKE            = "retake"
    VIDEO_OUTPAINT    = "video-outpaint"
    TEXT_TO_IMAGE     = "text-to-image"
    IMAGE_TO_IMAGE    = "image-to-image"
    IMAGE_EDIT        = "image-edit"
    EXPORT_COMPOSITION = "export-composition"
    MUSIC_GENERATION  = "music-generation"
```

Pass any of these as `?type=...` to `/v2/history`. Or use the shortcuts: `image`, `video`, `music`.

> `image` matches `text-to-image`, `image-to-image`, `image-edit` (excludes `image-to-video`).
> `video` matches anything `*-video*` and `video-outpaint` and `retake` and `export-composition`.

---

## 11. Result media types

| `job_type` | media | Content-Type |
|---|---|---|
| `text-to-image`, `image-to-image`, `image-edit` | WEBP (Q95) | `image/webp` |
| `text-to-video`, `image-to-video`, `audio-to-video`, `retake`, `video-outpaint`, `export-composition` | MP4 (H.264, CRF 18) | `video/mp4` |
| `music-generation` | WAV / MP3 (per ACE) | `audio/wav` or `audio/mpeg` |

For each completed history row, `image_url` (or `result_url` on jobs) returns the right MIME — read `Content-Type` rather than branching on `job_type`.

---

## 12. Errors

Standard FastAPI envelopes:

```jsonc
{ "error": "<code>", "message": "<human>", "detail": "<optional>" }
```

Common gallery-side codes:

| Status | code | When |
|---|---|---|
| 401 | `missing_api_key` | No bearer header |
| 403 | `quota_exceeded` | Per-key upload/queue/music/batch cap hit |
| 404 | (no code) | Row missing OR belongs to other key — never distinguish |
| 409 | (varies) | Job not yet completed, or already terminal |
| 413 | `upload_too_large` | Body > `MAX_UPLOAD_BYTES` |
| 422 | `content_type_mismatch` | Magic bytes ≠ declared `Content-Type` on PUT |
| 429 | `<resource>_queue_full` | Per-key queue (default cap 10), music (5), batch (5) |
| 503 | `turbo_mode_active` | Image / music / edit endpoints during turbo |

Treat 429 as transient (retry with backoff). Treat 401/403/404 as terminal for that request.

---

## 13. Pagination + caching strategy

The list endpoints are offset-based. For an infinite-scroll gallery:

1. `GET /v2/history?limit=50&offset=0`
2. Render with stable keys (`item.id`).
3. Cache thumbnails aggressively — they're immutable (`max-age=31_536_000`).
4. Subsequent pages: `offset += limit`. Stop when result length < `limit`.
5. For "fresh updates" (without re-fetching everything), poll `?limit=20&offset=0` every ~30 s and merge by `id`.

Gallery grid TTL recommendation: 30 s for the latest page, indefinite for older pages (safe — server retention is 30 days, no in-place mutation).

---

## 14. Bootstrap checklist

Minimum viable gallery webapp:

- [ ] **Auth**: store `Authorization: Bearer ...` header in fetch wrapper.
- [ ] **Grid view**: `GET /v2/history?type=<filter>&limit=50&offset=...` → grid of `<img src={thumbnail_url}>` with prompt subtitle.
- [ ] **Detail panel**: on click → `GET /v2/history/{id}` → show full media via `image_url`/`result_url`, render `params` + `gen_config` + `seed` + `enhanced_prompt`.
- [ ] **Live tiles**: subscribe to a job-list source (your own UI state — track `job_id`s returned by your `/v2/<...>` POSTs); render with `GET /v2/jobs/{id}/preview` + status from `/stream`.
- [ ] **Delete UX**: `DELETE /v2/history/{id}` → optimistic remove with undo toast.
- [ ] **Approved-images tab** (optional): `GET /v1/approved-images` for the curated set; `POST /v1/approved-images` to add from detail panel.
- [ ] **Compositions tab** (optional): `GET /v2/compositions` for saved timelines; click a composition to load its referenced clips.

Nice-to-have:

- [ ] `?type=` filter chips: image / video / music / all.
- [ ] Date-range filter (client-side over `created_at`).
- [ ] Reproduce button: re-POST `params` to the original endpoint (URL inferred from `job_type`).
- [ ] Permalink: `/gallery/<id>` deep-links to the detail panel for `id`.
- [ ] Export-to-album: hit `POST /v2/compositions` with selected `history_id`s.

---

## 15. Reference: every endpoint a gallery may touch

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/sse-token` | Issue 5-min token for EventSource auth |
| `GET` | `/v2/history` | List per-key history (paginated, filterable) |
| `GET` | `/v2/history/{id}` | Full record incl. params + gen_config + seed |
| `GET` | `/v2/history/{id}/image` | Full-res result media (immutable, 30d) |
| `GET` | `/v2/history/{id}/thumbnail` | 256px JPEG (immutable, 1y) |
| `DELETE` | `/v2/history/{id}` | Remove row + media + thumbnail |
| `GET` | `/v2/jobs/{id}` | Live job state |
| `GET` | `/v2/jobs/{id}/preview` | In-flight preview JPEG (or 204) |
| `GET` | `/v2/jobs/{id}/stream` | SSE live status (preferred over polling) |
| `GET` | `/v2/jobs/{id}/result` | Result media for in-flight downloads |
| `DELETE` | `/v2/jobs/{id}` | Cancel queued/processing |
| `POST` | `/v1/approved-images` | Approve a generated image |
| `GET` | `/v1/approved-images` | List per-key approvals |
| `GET` | `/v1/approved-images/events` | SSE for new approvals |
| `GET` | `/v1/approved-images/{id}/file` | Download approved image |
| `POST` | `/v2/compositions` | Save timeline |
| `GET` | `/v2/compositions` | List per-key |
| `GET` | `/v2/compositions/{id}` | Read one |
| `PUT` | `/v2/compositions/{id}` | Update |
| `DELETE` | `/v2/compositions/{id}` | Remove |
| `POST` | `/v2/compositions/{id}/export` | Render → returns job_id, poll via `/v2/jobs/{id}` |
| `PUT` | `/uploads/put/{uuid}` | Upload comparison/reference asset |
| `GET` | `/uploads/get/{uuid}` | Read uploaded asset |

For the rest (generation endpoints, system controls, dashboard), see `docs/API.md` (canonical) and `docs/QUICKSTART.md`.

---

## 16. Gotchas

- **No cross-key visibility**. Two keys cannot see each other's content. If you want a shared "team gallery", the FE must aggregate across multiple keys client-side.
- **404 ambiguity is intentional** for `/v2/history/{id}` and `DELETE /v2/history/{id}` — the server won't tell you whether a row is missing or just owned by another key. Don't try to probe.
- **Thumbnails may be missing** even on `status=completed`. Decode failures (corrupt MP4, exotic image format) yield no `thumbnail_url`. Always render a placeholder fallback.
- **Music thumbnails don't exist** — `_make_thumbnail` only handles image + MP4. Music-generation rows ship with `image_url` (audio file) but no `thumbnail_url`. Plan UI accordingly.
- **`params` is the EXACT request body** — including `storage://` URIs the user may have since deleted. Resolving a 404 on a referenced upload is the FE's responsibility.
- **`enhanced_prompt` is null for Flux/ERNIE/JoyAI/retake**. Only LTX videos with `enhance_prompt: true` populate it.
- **`gen_config` is null for non-turbo Flux, ERNIE, JoyAI**. Only LTX (full snapshot) and Flux-turbo (`{turbo_steps, turbo_guidance}`) write it.
- **Retention is hard**. Anything 30+ days old is gone. If your webapp needs forever-archives, copy the bytes via `/v2/history/{id}/image` to your own storage on view.
- **`/v2/jobs/{id}` evicts after a while** (in-memory job store). For older jobs always use the `/v2/history/{id}` endpoint instead.

---

**Last updated**: 2026-04-24, v1.13.0
**Canonical API spec**: `docs/API.md`
**Quick start**: `docs/QUICKSTART.md`
