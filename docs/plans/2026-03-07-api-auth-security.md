# API Key Authentication & Security Hardening Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-key API authentication to all server endpoints and harden input validation.

**Architecture:** Multiple API keys stored in `.api_keys` file (one per line), validated via FastAPI middleware. Timing-safe comparison with `secrets.compare_digest` against all valid keys. Health endpoint exempt for monitoring. Input bounds added to Pydantic models.

**Tech Stack:** FastAPI middleware, Python `secrets`, Pydantic field validators, pytest

---

## Threat Analysis Summary

### Current State
- **No authentication** — any device on the LAN can hit all endpoints
- **No CORS** — browser-based attacks possible from any origin
- **No rate limiting** — single inference lock prevents concurrent GPU use but requests queue unbounded
- **Partial input validation** — Pydantic models enforce types but not value ranges
- **Good path sanitization** — `_error()` strips internal paths, `upload_store` validates IDs with regex
- **Good upload limits** — 500MB max enforced in both header and body

### Gaps (by severity)
| Severity | Gap | Impact |
|----------|-----|--------|
| Critical | No auth on any endpoint | LAN devices can consume GPU, exfil data |
| High | No input bounds on numeric fields | DoS via extreme values (width=100000, duration=9999) |
| Medium | No CORS restriction | Browser-based CSRF from LAN |
| Medium | No request body size limit | Large JSON payloads to non-upload endpoints |
| Low | Chat proxy forwards all message fields | Extra fields passed to upstream LLM |
| Low | No .api_keys in .gitignore | Keys could be committed accidentally |

---

### Task 1: Add multi-key config and generation script

**Files:**
- Modify: `config.py`
- Create: `generate_keys.py`
- Modify: `.gitignore`
- Modify: `run.sh`

**Step 1: Add .api_keys to .gitignore**

Append to `.gitignore`:

```
.env
.api_keys
```

**Step 2: Add API key loading to config.py**

Add after the existing `from pathlib import Path` import:

```python
import os

def _load_api_keys() -> set[str]:
    """Load API keys from .api_keys file and/or TACO_API_KEY env var."""
    keys: set[str] = set()
    # Load from file (one key per line, # comments allowed)
    keys_file = Path(__file__).parent / ".api_keys"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.add(line)
    # Also accept single key from env var
    env_key = os.environ.get("TACO_API_KEY", "").strip()
    if env_key:
        keys.add(env_key)
    return keys

API_KEYS: set[str] = _load_api_keys()
```

**Step 3: Create key generation script**

```python
#!/usr/bin/env python3
"""Generate API keys for taco-backend.

Usage:
    python generate_keys.py           # Generate 5 keys, write to .api_keys
    python generate_keys.py 3         # Generate 3 keys
    python generate_keys.py --append  # Append to existing file
"""
import secrets
import sys
from pathlib import Path

KEYS_FILE = Path(__file__).parent / ".api_keys"

def main():
    count = 5
    append = False
    for arg in sys.argv[1:]:
        if arg == "--append":
            append = True
        elif arg.isdigit():
            count = int(arg)

    keys = [secrets.token_urlsafe(32) for _ in range(count)]

    mode = "a" if append else "w"
    with open(KEYS_FILE, mode) as f:
        if not append:
            f.write("# taco-backend API keys (one per line)\n")
            f.write("# Distribute one key per client. Regenerate with: python generate_keys.py\n")
        for key in keys:
            f.write(f"{key}\n")

    print(f"{'Appended' if append else 'Wrote'} {count} keys to {KEYS_FILE}")
    print()
    for i, key in enumerate(keys, 1):
        print(f"  Key {i}: {key}")
    print()
    print("Frontend config: set Authorization header to 'Bearer <key>'")

if __name__ == "__main__":
    main()
```

**Step 4: Commit**

```bash
git add config.py generate_keys.py .gitignore
git commit -m "feat: add multi-key API auth config and key generation"
```

---

### Task 2: Write auth middleware tests

**Files:**
- Create: `tests/test_auth.py`

**Step 1: Write failing tests**

```python
"""Tests for API key authentication middleware."""
import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

TEST_KEYS = {"key-alpha-111", "key-bravo-222", "key-charlie-333"}


@pytest.fixture
def authed_client():
    """Client with auth configured (multi-key)."""
    import config
    original_keys = config.API_KEYS
    config.API_KEYS = TEST_KEYS.copy()
    from server import app
    client = TestClient(app)
    yield client
    config.API_KEYS = original_keys


def test_health_no_auth_required(authed_client):
    """GET /health should work without API key."""
    resp = authed_client.get("/health")
    assert resp.status_code == 200


def test_endpoint_rejects_missing_key(authed_client):
    """POST endpoints should reject requests without API key."""
    resp = authed_client.post(
        "/v1/chat/completions",
        json={"model": "gemma-3-12b-nvfp4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401
    assert "error" in resp.json()


def test_endpoint_rejects_wrong_key(authed_client):
    """POST endpoints should reject requests with wrong API key."""
    resp = authed_client.post(
        "/v1/chat/completions",
        json={"model": "gemma-3-12b-nvfp4", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401


def test_endpoint_accepts_any_valid_key(authed_client):
    """POST endpoints should accept any of the configured keys."""
    for key in TEST_KEYS:
        resp = authed_client.post(
            "/v1/chat/completions",
            json={"model": "gemma-3-12b-nvfp4", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {key}"},
        )
        # Should get past auth (may fail for other reasons like model not loaded)
        assert resp.status_code != 401, f"Key {key} was rejected"


def test_no_auth_when_no_keys_configured():
    """When no API keys configured, auth should be disabled (dev mode)."""
    import config
    original_keys = config.API_KEYS
    config.API_KEYS = set()
    from server import app
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gemma-3-12b-nvfp4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code != 401
    config.API_KEYS = original_keys


def test_upload_put_requires_auth(authed_client):
    """PUT /uploads/put/{id} should require auth."""
    resp = authed_client.put("/uploads/put/abc123", content=b"data")
    assert resp.status_code == 401
```

**Step 2: Run tests, verify they fail**

```bash
uv run pytest tests/test_auth.py -v
```

Expected: FAIL (no auth middleware exists yet)

**Step 3: Commit**

```bash
git add tests/test_auth.py
git commit -m "test: add multi-key API auth tests (red)"
```

---

### Task 3: Implement auth middleware

**Files:**
- Modify: `server.py`

**Step 1: Add auth middleware to server.py**

Add `import secrets as _secrets` to the imports at the top of the file.

After the `app = FastAPI(lifespan=lifespan)` line, add:

```python
@app.middleware("http")
async def check_api_key(request: Request, call_next):
    # Skip auth if no keys configured (development mode)
    if not config.API_KEYS:
        return await call_next(request)

    # Health check is always public
    if request.url.path == "/health":
        return await call_next(request)

    # Extract Bearer token
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = ""

    # Timing-safe comparison against all valid keys
    if not token or not any(
        _secrets.compare_digest(token, key) for key in config.API_KEYS
    ):
        return _error(401, "Invalid or missing API key")

    return await call_next(request)
```

**Step 2: Run auth tests**

```bash
uv run pytest tests/test_auth.py -v
```

Expected: ALL PASS

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -v
```

Existing tests should still pass (no keys configured = auth disabled).

**Step 4: Commit**

```bash
git add server.py
git commit -m "feat: add multi-key API auth middleware with timing-safe comparison"
```

---

### Task 4: Add input validation bounds to Pydantic models

**Files:**
- Modify: `server.py` (Pydantic model fields)
- Create: `tests/test_validation.py`

**Step 1: Write failing validation tests**

```python
"""Tests for input validation bounds."""
from fastapi.testclient import TestClient
import config
config.GPU_DEVICES = []
from server import app

client = TestClient(app)


def test_text_to_video_rejects_extreme_duration():
    resp = client.post("/v1/text-to-video", json={
        "prompt": "test",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 999999,
        "fps": 24.0,
    })
    assert resp.status_code == 422


def test_text_to_image_rejects_extreme_dimensions():
    resp = client.post("/v1/text-to-image", json={
        "prompt": "test",
        "model": "flux2-dev",
        "width": 100000,
        "height": 100000,
    })
    assert resp.status_code == 422


def test_chat_rejects_extreme_max_tokens():
    resp = client.post("/v1/chat/completions", json={
        "model": "gemma-3-12b-nvfp4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 999999,
    })
    assert resp.status_code == 422


def test_text_to_video_rejects_negative_fps():
    resp = client.post("/v1/text-to-video", json={
        "prompt": "test",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 6.0,
        "fps": -1,
    })
    assert resp.status_code == 422


def test_text_to_image_accepts_valid_request():
    """Valid params should get past validation (will fail at pipeline level)."""
    resp = client.post("/v1/text-to-image", json={
        "prompt": "test",
        "model": "flux2-dev",
        "width": 1024,
        "height": 1024,
    })
    # 500 = got past validation, failed at pipeline (not loaded)
    assert resp.status_code == 500
```

**Step 2: Add Field constraints to Pydantic models in server.py**

Add `Field` to the pydantic import:

```python
from pydantic import BaseModel, Field
```

Update all request models:

```python
class TextToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False
    camera_motion: str | None = Field(default=None, max_length=200)


class ImageToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False


class AudioToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    audio_uri: str
    image_uri: str | None = None
    model: ModelName
    resolution: Resolution
    duration: float = Field(default=6.0, gt=0, le=30)
    fps: float = Field(default=24.0, gt=0, le=60)


class RetakeRequest(BaseModel):
    video_uri: str
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0, le=30)
    mode: RetakeMode
    prompt: str | None = Field(default=None, max_length=10000)


class TextToImageRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    model: ImageModelName = "flux2-dev"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None


class ImageToImageRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str
    model: ImageModelName = "flux2-dev"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list  # str for text, list for multimodal


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3-12b-nvfp4"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
```

**Step 3: Run tests**

```bash
uv run pytest tests/test_validation.py tests/ -v
```

**Step 4: Commit**

```bash
git add server.py tests/test_validation.py
git commit -m "security: add input validation bounds to all request models"
```

---

### Task 5: Add CORS middleware

**Files:**
- Modify: `server.py`

**Step 1: Add CORS middleware**

Add import and middleware after the auth middleware in `server.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|192\.168\.\d+\.\d+)(:\d+)?$",
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)
```

**Step 2: Commit**

```bash
git add server.py
git commit -m "security: add CORS middleware restricting to LAN origins"
```

---

### Task 6: Sanitize chat proxy payload

**Files:**
- Modify: `chat_manager.py`

**Step 1: Whitelist fields in chat proxy**

In `generate_chat_completion`, sanitize messages before forwarding:

```python
    # Whitelist message fields to prevent injection to upstream LLM
    clean_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
    ]

    payload = {
        "model": config.CHAT_MODEL,
        "messages": clean_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
```

**Step 2: Commit**

```bash
git add chat_manager.py
git commit -m "security: whitelist chat proxy message fields"
```

---

### Task 7: Generate keys, integration test, update run.sh

**Files:**
- Modify: `run.sh`
- Generate: `.api_keys`

**Step 1: Update run.sh to load .api_keys**

No changes needed — `config.py` reads `.api_keys` directly from `Path(__file__).parent`.

But add `.env` sourcing for any other env vars:

```bash
# Load environment variables (optional, for TACO_API_KEY override)
if [ -f .env ]; then
    set -a; source .env; set +a
fi
```

Add this before the `exec` line in `run.sh`.

**Step 2: Generate 5 keys**

```bash
python generate_keys.py 5
```

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -v
```

**Step 4: Smoke test with live server**

```bash
# Restart server (it will load .api_keys on startup)
# Test without key — should get 401
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-3-12b-nvfp4","messages":[{"role":"user","content":"hi"}]}'

# Test with key — should get 200
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key-1-from-api_keys>" \
  -d '{"model":"gemma-3-12b-nvfp4","messages":[{"role":"user","content":"hi"}]}'

# Health — always works
curl -s http://localhost:8090/health
```

**Step 5: Commit**

```bash
git add run.sh
git commit -m "feat: add .env sourcing to run.sh for optional env overrides"
```
