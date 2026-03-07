"""Tests for API key authentication middleware."""
import config

config.GPU_DEVICES = []

TEST_KEYS = {"key-alpha-111", "key-bravo-222", "key-charlie-333"}

from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402


def _with_keys(keys):
    """Context manager to temporarily set API keys."""
    class _ctx:
        def __enter__(self):
            self._orig = config.API_KEYS
            config.API_KEYS = keys
            return self
        def __exit__(self, *a):
            config.API_KEYS = self._orig
    return _ctx()


client = TestClient(app)


def test_health_no_auth_required():
    with _with_keys(TEST_KEYS):
        resp = client.get("/health")
        assert resp.status_code == 200


def test_endpoint_rejects_missing_key():
    with _with_keys(TEST_KEYS):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gemma-3-12b-nvfp4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
        assert "error" in resp.json()


def test_endpoint_rejects_wrong_key():
    with _with_keys(TEST_KEYS):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gemma-3-12b-nvfp4", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401


def test_endpoint_accepts_any_valid_key():
    with _with_keys(TEST_KEYS):
        for key in TEST_KEYS:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gemma-3-12b-nvfp4", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": f"Bearer {key}"},
            )
            assert resp.status_code != 401, f"Key {key} was rejected"


def test_no_auth_when_no_keys_configured():
    with _with_keys(set()):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gemma-3-12b-nvfp4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code != 401


def test_upload_put_requires_auth():
    with _with_keys(TEST_KEYS):
        resp = client.put("/uploads/put/abc123", content=b"data")
        assert resp.status_code == 401


def test_upload_post_requires_auth():
    with _with_keys(TEST_KEYS):
        resp = client.post("/v1/upload")
        assert resp.status_code == 401
