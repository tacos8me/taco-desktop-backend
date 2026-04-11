"""Tests for input validation bounds."""
import config

config.GPU_DEVICES = []
config.API_KEYS = set()  # disable auth for validation tests

from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402

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


def test_text_to_video_rejects_negative_fps():
    resp = client.post("/v1/text-to-video", json={
        "prompt": "test",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 6.0,
        "fps": -1,
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


def test_text_to_image_accepts_valid_request():
    """Valid params should get past validation (will fail at pipeline level)."""
    resp = client.post("/v1/text-to-image", json={
        "prompt": "test",
        "model": "flux2-dev",
        "width": 1024,
        "height": 1024,
    })
    assert resp.status_code == 500  # past validation, fails at pipeline


def test_image_edit_accepts_joyai_edit_model():
    """joyai-edit should pass schema validation with one image_uri."""
    resp = client.post("/v1/image-edit", json={
        "prompt": "make it a cat",
        "image_uris": ["storage://00000000-0000-0000-0000-000000000000"],
        "model": "joyai-edit",
        "width": 1024,
        "height": 1024,
    })
    # Not 422 — the schema accepts joyai-edit. Handler returns 404 (image not
    # found in uploads) or 503 (joyai disabled) depending on state; both mean
    # validation passed.
    assert resp.status_code != 422


def test_image_edit_joyai_rejects_multiple_images():
    """joyai-edit only supports exactly one image_uri — handler returns 422."""
    resp = client.post("/v1/image-edit", json={
        "prompt": "make it a cat",
        "image_uris": [
            "storage://00000000-0000-0000-0000-000000000000",
            "storage://00000000-0000-0000-0000-000000000001",
        ],
        "model": "joyai-edit",
        "width": 1024,
        "height": 1024,
    })
    assert resp.status_code == 422


def test_image_edit_rejects_unknown_model():
    """Unknown model values should fail pydantic literal validation."""
    resp = client.post("/v1/image-edit", json={
        "prompt": "test",
        "image_uris": ["storage://00000000-0000-0000-0000-000000000000"],
        "model": "nonsense-model",
    })
    assert resp.status_code == 422
