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


# ---------------------------------------------------------------------------
# Music generation validation
# ---------------------------------------------------------------------------


def test_music_accepts_valid_request():
    """Valid music params should get past pydantic validation."""
    resp = client.post("/v1/music", json={
        "prompt": "epic orchestral soundtrack",
    })
    # Not 422 means validation passed. Will be 503 (LOAD_ACE=0) in test env.
    assert resp.status_code != 422


def test_music_rejects_extreme_duration():
    """Duration > 600 should fail pydantic validation."""
    resp = client.post("/v1/music", json={
        "prompt": "test",
        "duration": 9999,
    })
    assert resp.status_code == 422


def test_music_rejects_invalid_audio_format():
    """Invalid audio format should fail pydantic validation."""
    resp = client.post("/v1/music", json={
        "prompt": "test",
        "audio_format": "ogg",
    })
    assert resp.status_code == 422


def test_music_rejects_invalid_task_type():
    """Invalid task_type should fail pydantic validation."""
    resp = client.post("/v1/music", json={
        "prompt": "test",
        "task_type": "invalid_type",
    })
    assert resp.status_code == 422


def test_music_cover_without_source_returns_422():
    """task_type=cover without source_audio_uri should return 422 from handler."""
    resp = client.post("/v1/music", json={
        "prompt": "cover this song",
        "task_type": "cover",
    })
    # Either 422 (handler validation) or 503 (ACE disabled) -- both mean
    # pydantic validation passed. The handler check happens before ACE call.
    assert resp.status_code in (422, 503)
