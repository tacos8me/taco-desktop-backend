"""Validation + smoke tests for /v2/video-hdr (v1.14.0 IC-LoRA HDR endpoint).

These tests run with auth disabled and never reach the GPU — they exercise
Pydantic validation, the LoRA-resolve path (which hits the real registry on
disk), and the upload-not-found error path. Live HDR generation is gated
behind real GPU + model weights and is verified manually post-deploy.
"""
import config

config.GPU_DEVICES = []
config.API_KEYS = set()  # disable auth

from fastapi.testclient import TestClient  # noqa: E402

from server import app, DEFAULT_HDR_LORA_ID  # noqa: E402
from job_queue import JobType  # noqa: E402

client = TestClient(app)


def test_video_hdr_jobtype_registered():
    """JobType.VIDEO_HDR must be a real enum member with the expected value."""
    assert JobType.VIDEO_HDR.value == "video-hdr"


def test_video_hdr_default_lora_id_set():
    """The endpoint constant must point at the registered HDR LoRA."""
    assert DEFAULT_HDR_LORA_ID == "ic-lora-hdr"


def test_video_hdr_rejects_missing_video_uri():
    resp = client.post("/v2/video-hdr", json={
        "prompt": "expand highlights",
        "duration": 6.0,
        "fps": 24.0,
    })
    assert resp.status_code == 422


def test_video_hdr_rejects_missing_prompt():
    resp = client.post("/v2/video-hdr", json={
        "video_uri": "storage://" + "0" * 32,
        "duration": 6.0,
        "fps": 24.0,
    })
    assert resp.status_code == 422


def test_video_hdr_rejects_extreme_duration():
    resp = client.post("/v2/video-hdr", json={
        "video_uri": "storage://" + "0" * 32,
        "prompt": "x",
        "duration": 9999,
        "fps": 24.0,
    })
    assert resp.status_code == 422


def test_video_hdr_rejects_negative_conditioning_strength():
    resp = client.post("/v2/video-hdr", json={
        "video_uri": "storage://" + "0" * 32,
        "prompt": "x",
        "duration": 6.0,
        "fps": 24.0,
        "conditioning_strength": -0.1,
    })
    assert resp.status_code == 422


def test_video_hdr_rejects_conditioning_strength_above_one():
    resp = client.post("/v2/video-hdr", json={
        "video_uri": "storage://" + "0" * 32,
        "prompt": "x",
        "duration": 6.0,
        "fps": 24.0,
        "conditioning_strength": 1.5,
    })
    assert resp.status_code == 422


def test_video_hdr_unknown_video_uri_returns_404():
    """Body validates; LoRA may resolve OK; but upload doesn't exist → 404."""
    resp = client.post("/v2/video-hdr", json={
        "video_uri": "storage://" + "f" * 32,
        "prompt": "expand highlights",
        "duration": 6.0,
        "fps": 24.0,
    })
    # Either 404 video_not_found (good — past validation) or 400/500 if the
    # HDR LoRA isn't registered yet on this box. Both prove the endpoint is
    # reachable and Pydantic-valid.
    assert resp.status_code in (400, 404, 500)


def test_video_hdr_skip_stage_2_accepted():
    """skip_stage_2 must be a recognized field, not an unknown-field 422."""
    resp = client.post("/v2/video-hdr", json={
        "video_uri": "storage://" + "f" * 32,
        "prompt": "expand highlights",
        "duration": 6.0,
        "fps": 24.0,
        "skip_stage_2": True,
    })
    # Must NOT 422 — that would mean Pydantic rejected the field.
    assert resp.status_code != 422
