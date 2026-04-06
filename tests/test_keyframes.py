"""Tests for multi-keyframe image-to-video validation."""
import config

config.GPU_DEVICES = []
config.API_KEYS = set()

from fastapi.testclient import TestClient  # noqa: E402

from server import app, uploads  # noqa: E402

client = TestClient(app)

# Create a fake uploaded image for URI resolution
_upload_id_1, _uri_1 = uploads.create()
uploads.save(_upload_id_1, b"\x89PNG\r\n\x1a\nfake1")
_upload_id_2, _uri_2 = uploads.create()
uploads.save(_upload_id_2, b"\x89PNG\r\n\x1a\nfake2")

_BASE = {
    "prompt": "test",
    "model": "ltx-2-3-fast",
    "resolution": "1920x1080",
    "duration": 3.0,
    "fps": 24.0,
}


def _post(extra: dict, endpoint: str = "/v1/image-to-video"):
    return client.post(endpoint, json={**_BASE, **extra})


def test_single_image_uri_backward_compat():
    """image_uri alone should still be accepted (500 = past validation, no GPU)."""
    resp = _post({"image_uri": _uri_1})
    assert resp.status_code == 500


def test_single_keyframe():
    """Single keyframe equivalent to image_uri."""
    resp = _post({"keyframes": [{"image_uri": _uri_1, "frame_index": 0, "strength": 1.0}]})
    assert resp.status_code == 500


def test_two_keyframes():
    """Two keyframes at different frame indices."""
    resp = _post({"keyframes": [
        {"image_uri": _uri_1, "frame_index": 0, "strength": 1.0},
        {"image_uri": _uri_2, "frame_index": 40, "strength": 0.8},
    ]})
    assert resp.status_code == 500


def test_both_image_uri_and_keyframes_rejected():
    resp = _post({
        "image_uri": _uri_1,
        "keyframes": [{"image_uri": _uri_2, "frame_index": 0}],
    })
    assert resp.status_code == 422


def test_neither_image_uri_nor_keyframes_rejected():
    resp = _post({})
    assert resp.status_code == 422


def test_empty_keyframes_rejected():
    resp = _post({"keyframes": []})
    assert resp.status_code == 422


def test_duplicate_frame_index_rejected():
    resp = _post({"keyframes": [
        {"image_uri": _uri_1, "frame_index": 5},
        {"image_uri": _uri_2, "frame_index": 5},
    ]})
    assert resp.status_code == 422


def test_too_many_keyframes_rejected():
    kfs = [{"image_uri": _uri_1, "frame_index": i} for i in range(9)]
    resp = _post({"keyframes": kfs})
    assert resp.status_code == 422


def test_negative_frame_index_accepted():
    """Negative frame_index is now supported (Python-style: -1 = last)."""
    resp = _post({"keyframes": [{"image_uri": _uri_1, "frame_index": -1}]})
    # Should pass validation (500 = no GPU, not 422)
    assert resp.status_code in (500, 202)


def test_symbolic_first_last():
    """Symbolic 'first' and 'last' frame indices."""
    resp = _post({"keyframes": [
        {"image_uri": _uri_1, "frame_index": "first", "strength": 1.0},
        {"image_uri": _uri_2, "frame_index": "last", "strength": 1.0},
    ]})
    assert resp.status_code in (500, 202)


def test_symbolic_first_mid_last():
    """Symbolic 'first', 'middle', 'last' frame indices."""
    resp = _post({"keyframes": [
        {"image_uri": _uri_1, "frame_index": "first"},
        {"image_uri": _uri_2, "frame_index": "middle", "strength": 0.5},
        {"image_uri": _uri_1, "frame_index": "last"},
    ]})
    assert resp.status_code in (500, 202)


def test_symbolic_invalid_rejected():
    """Invalid symbolic value should be rejected."""
    resp = _post({"keyframes": [{"image_uri": _uri_1, "frame_index": "center"}]})
    assert resp.status_code == 422


def test_strength_out_of_range_rejected():
    resp = _post({"keyframes": [{"image_uri": _uri_1, "strength": 1.5}]})
    assert resp.status_code == 422


# V2 endpoint tests

def test_v2_single_image_uri():
    resp = _post({"image_uri": _uri_1}, endpoint="/v2/image-to-video")
    assert resp.status_code == 202


def test_v2_keyframes():
    resp = _post({"keyframes": [
        {"image_uri": _uri_1, "frame_index": 0},
        {"image_uri": _uri_2, "frame_index": 30},
    ]}, endpoint="/v2/image-to-video")
    assert resp.status_code == 202


def test_v2_both_rejected():
    resp = _post({
        "image_uri": _uri_1,
        "keyframes": [{"image_uri": _uri_2}],
    }, endpoint="/v2/image-to-video")
    assert resp.status_code == 422


def test_v2_neither_rejected():
    resp = _post({}, endpoint="/v2/image-to-video")
    assert resp.status_code == 422
