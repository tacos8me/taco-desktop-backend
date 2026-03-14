"""Tests for async job queue v2 endpoints."""
import config

config.GPU_DEVICES = []

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient  # noqa: E402

from server import app, job_store, _job_queue  # noqa: E402
from job_queue import Job, JobStatus, JobType, make_job_id  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key-123"}


def _with_no_auth():
    """Context manager to disable auth for testing."""
    class _ctx:
        def __enter__(self):
            self._orig = config.API_KEYS
            config.API_KEYS = set()
            return self
        def __exit__(self, *a):
            config.API_KEYS = self._orig
    return _ctx()


def _cleanup_queue():
    """Drain the queue and clear the store."""
    while not _job_queue.empty():
        try:
            _job_queue.get_nowait()
        except Exception:
            break
    job_store._jobs.clear()


def test_v2_submit_returns_202():
    with _with_no_auth():
        resp = client.post("/v2/text-to-video", json={
            "prompt": "test", "model": "ltx-2-3-fast",
            "resolution": "1920x1080", "duration": 5.0, "fps": 24.0,
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert "poll_url" in data
        _cleanup_queue()


def test_v2_submit_hq_returns_202():
    with _with_no_auth():
        resp = client.post("/v2/text-to-video", json={
            "prompt": "test", "model": "ltx-2-3-hq",
            "resolution": "1920x1080", "duration": 5.0, "fps": 24.0,
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        _cleanup_queue()


def test_v2_submit_rejects_invalid_input():
    with _with_no_auth():
        resp = client.post("/v2/text-to-video", json={
            "prompt": "test", "model": "ltx-2-3-fast",
            "resolution": "1920x1080", "duration": 999999, "fps": 24.0,
        })
        assert resp.status_code == 422


def test_v2_poll_returns_job_status():
    with _with_no_auth():
        # Submit a job
        resp = client.post("/v2/text-to-image", json={
            "prompt": "test", "model": "flux2-dev",
            "width": 1024, "height": 1024,
        })
        job_id = resp.json()["job_id"]

        # Poll it
        resp = client.get(f"/v2/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] == "queued"
        assert data["queue_position"] is not None
        _cleanup_queue()


def test_v2_poll_unknown_job_returns_404():
    with _with_no_auth():
        resp = client.get("/v2/jobs/nonexistent-id")
        assert resp.status_code == 404


def test_v2_result_not_ready_returns_409():
    with _with_no_auth():
        resp = client.post("/v2/text-to-image", json={
            "prompt": "test", "model": "flux2-dev",
        })
        job_id = resp.json()["job_id"]

        resp = client.get(f"/v2/jobs/{job_id}/result")
        assert resp.status_code == 409
        _cleanup_queue()


def test_v2_result_completed_returns_binary():
    with _with_no_auth():
        # Create a completed job manually
        job = Job(id=make_job_id(), type=JobType.TEXT_TO_IMAGE, status=JobStatus.COMPLETED,
                  result_media_type="image/webp")
        # Store some fake result via upload_store
        from server import uploads
        uid, uri = uploads.create()
        uploads.save(uid, b"fake-webp-data")
        job.result_uri = uri
        job_store.add(job)

        resp = client.get(f"/v2/jobs/{job.id}/result")
        assert resp.status_code == 200
        assert resp.content == b"fake-webp-data"
        assert resp.headers["content-type"] == "image/webp"
        _cleanup_queue()


def test_v2_cancel_queued_job():
    with _with_no_auth():
        resp = client.post("/v2/text-to-image", json={
            "prompt": "test", "model": "flux2-dev",
        })
        job_id = resp.json()["job_id"]

        resp = client.delete(f"/v2/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

        # Verify it's cancelled
        resp = client.get(f"/v2/jobs/{job_id}")
        assert resp.json()["status"] == "cancelled"
        _cleanup_queue()


def test_v2_cancel_finished_job_returns_409():
    with _with_no_auth():
        job = Job(id=make_job_id(), type=JobType.TEXT_TO_IMAGE, status=JobStatus.COMPLETED)
        job_store.add(job)

        resp = client.delete(f"/v2/jobs/{job.id}")
        assert resp.status_code == 409
        _cleanup_queue()


def test_v2_queue_full_returns_429():
    with _with_no_auth():
        orig = config.MAX_QUEUE_DEPTH
        config.MAX_QUEUE_DEPTH = 2
        try:
            # Fill the queue
            for _ in range(2):
                resp = client.post("/v2/text-to-image", json={
                    "prompt": "test", "model": "flux2-dev",
                })
                assert resp.status_code == 202

            # Third should be rejected
            resp = client.post("/v2/text-to-image", json={
                "prompt": "test", "model": "flux2-dev",
            })
            assert resp.status_code == 429
            assert "Retry-After" in resp.headers
        finally:
            config.MAX_QUEUE_DEPTH = orig
            _cleanup_queue()


def test_v2_submit_requires_auth():
    from tests.test_auth import _with_keys, TEST_KEYS
    with _with_keys(TEST_KEYS):
        resp = client.post("/v2/text-to-image", json={
            "prompt": "test", "model": "flux2-dev",
        })
        assert resp.status_code == 401
        _cleanup_queue()


def test_health_includes_queue_stats():
    with _with_no_auth():
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "queue" in resp.json()
