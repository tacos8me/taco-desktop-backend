"""Tests for the FastAPI server (no GPU required)."""

import config

# Disable GPU loading before importing the server module
config.GPU_DEVICES = []

from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "ltx" in data
    assert "flux" in data
    assert "chat" in data


def test_upload_flow():
    # Step 1: create upload slot
    resp = client.post("/v1/upload")
    assert resp.status_code == 200
    data = resp.json()
    assert "upload_url" in data
    assert "storage_uri" in data
    assert data["storage_uri"].startswith("storage://")
    assert data["required_headers"] == {}

    upload_url = data["upload_url"]
    # upload_url is absolute, but TestClient needs a relative path
    # Extract path from the full URL
    from urllib.parse import urlparse

    path = urlparse(upload_url).path

    # Step 2: PUT file bytes
    file_bytes = b"\x89PNG\r\n\x1a\nfake-image-data"
    resp = client.put(path, content=file_bytes)
    assert resp.status_code == 201


def test_chat_completions_returns_error_without_gpu():
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "loco-operator",
            "messages": [
                {"role": "system", "content": "You are a prompt engineer."},
                {"role": "user", "content": "a cat on a windowsill"},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
        },
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data


def test_chat_completions_validates_messages():
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "loco-operator",
            "messages": [],
        },
    )
    assert resp.status_code in (422, 500)


def test_text_to_video_returns_error_without_gpu():
    resp = client.post(
        "/v1/text-to-video",
        json={
            "prompt": "a cat sitting on a windowsill",
            "model": "ltx-2-3-fast",
            "resolution": "1920x1080",
            "duration": 6.0,
            "fps": 24.0,
            "generate_audio": False,
        },
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data
