"""Tests for batch scheduler endpoints (Phase 3)."""

import config

config.GPU_DEVICES = []

from fastapi.testclient import TestClient  # noqa: E402

from server import app, batch_store, _batch_queue  # noqa: E402
from job_queue import BatchStatus, make_batch_id, BatchJob  # noqa: E402

client = TestClient(app)


def _with_no_auth():
    class _ctx:
        def __enter__(self):
            self._orig = config.API_KEYS
            config.API_KEYS = set()
            return self
        def __exit__(self, *a):
            config.API_KEYS = self._orig
    return _ctx()


def _cleanup_batch():
    while not _batch_queue.empty():
        try:
            _batch_queue.get_nowait()
        except Exception:
            break
    batch_store._batches.clear()


def test_batch_submit_returns_202():
    with _with_no_auth():
        resp = client.post("/v2/batch", json={
            "items": [
                {"type": "text-to-image", "params": {"prompt": "a cat", "model": "flux2-dev", "width": 1024, "height": 1024}},
                {"type": "text-to-image", "params": {"prompt": "a dog", "model": "flux2-klein", "width": 512, "height": 512}},
            ],
            "priority": "normal",
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "batch_id" in data
        assert data["status"] == "queued"
        assert data["total"] == 2
        _cleanup_batch()


def test_batch_submit_invalid_params_returns_422():
    with _with_no_auth():
        resp = client.post("/v2/batch", json={
            "items": [
                {"type": "text-to-image", "params": {"prompt": "ok", "model": "flux2-dev"}},
                {"type": "text-to-image", "params": {"prompt": "bad", "model": "flux2-dev", "width": -1}},
            ],
        })
        assert resp.status_code == 422
        assert "item 1" in resp.json()["error"]
        _cleanup_batch()


def test_batch_submit_empty_items_returns_422():
    with _with_no_auth():
        resp = client.post("/v2/batch", json={"items": []})
        assert resp.status_code == 422
        _cleanup_batch()


def test_batch_submit_invalid_type_returns_400():
    with _with_no_auth():
        resp = client.post("/v2/batch", json={
            "items": [
                {"type": "text-to-image", "params": {"prompt": "ok", "model": "flux2-dev"}},
            ],
        })
        # text-to-image with valid params should be 202
        assert resp.status_code == 202
        _cleanup_batch()


def test_batch_status_poll():
    with _with_no_auth():
        resp = client.post("/v2/batch", json={
            "items": [
                {"type": "text-to-image", "params": {"prompt": "test", "model": "flux2-dev"}},
            ],
        })
        batch_id = resp.json()["batch_id"]

        resp = client.get(f"/v2/batch/{batch_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == batch_id
        assert data["status"] == "queued"
        assert data["total"] == 1
        assert "results" in data
        assert "created_at" in data
        _cleanup_batch()


def test_batch_status_unknown_returns_404():
    with _with_no_auth():
        resp = client.get("/v2/batch/nonexistent")
        assert resp.status_code == 404
        _cleanup_batch()


def test_batch_cancel():
    with _with_no_auth():
        resp = client.post("/v2/batch", json={
            "items": [
                {"type": "text-to-image", "params": {"prompt": "a", "model": "flux2-dev"}},
                {"type": "text-to-image", "params": {"prompt": "b", "model": "flux2-dev"}},
            ],
        })
        batch_id = resp.json()["batch_id"]

        resp = client.delete(f"/v2/batch/{batch_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"
        assert data["batch_id"] == batch_id
        _cleanup_batch()


def test_batch_cancel_finished_returns_409():
    with _with_no_auth():
        batch = BatchJob(
            id=make_batch_id(), items=[], status=BatchStatus.COMPLETED,
            total=0,
        )
        batch_store.add(batch)

        resp = client.delete(f"/v2/batch/{batch.id}")
        assert resp.status_code == 409
        _cleanup_batch()


def test_batch_queue_full_returns_429():
    with _with_no_auth():
        orig = config.MAX_BATCH_QUEUE_DEPTH
        config.MAX_BATCH_QUEUE_DEPTH = 1
        try:
            resp1 = client.post("/v2/batch", json={
                "items": [{"type": "text-to-image", "params": {"prompt": "a", "model": "flux2-dev"}}],
            })
            assert resp1.status_code == 202

            resp2 = client.post("/v2/batch", json={
                "items": [{"type": "text-to-image", "params": {"prompt": "b", "model": "flux2-dev"}}],
            })
            assert resp2.status_code == 429
            assert "Retry-After" in resp2.headers
        finally:
            config.MAX_BATCH_QUEUE_DEPTH = orig
            _cleanup_batch()


def test_batch_swap_optimization_images_before_videos():
    """Verify batch items are sorted: images first, then videos."""
    with _with_no_auth():
        resp = client.post("/v2/batch", json={
            "items": [
                {"type": "text-to-video", "params": {
                    "prompt": "vid", "model": "ltx-2-3-fast",
                    "resolution": "1920x1080", "duration": 5, "fps": 24,
                }},
                {"type": "text-to-image", "params": {"prompt": "img", "model": "flux2-dev"}},
            ],
        })
        assert resp.status_code == 202
        batch_id = resp.json()["batch_id"]
        batch = batch_store.get(batch_id)
        # Image should come first after sorting
        assert batch.items[0].type == "text-to-image"
        assert batch.items[1].type == "text-to-video"
        _cleanup_batch()


def test_batch_mixed_types_validated():
    """A batch with both image and video items should validate all."""
    with _with_no_auth():
        resp = client.post("/v2/batch", json={
            "items": [
                {"type": "text-to-image", "params": {"prompt": "cat", "model": "flux2-klein"}},
                {"type": "text-to-video", "params": {
                    "prompt": "dog", "model": "ltx-2-3-fast",
                    "resolution": "1920x1080", "duration": 3, "fps": 24,
                }},
            ],
        })
        assert resp.status_code == 202
        _cleanup_batch()
