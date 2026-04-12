"""Tests for AceClient using httpx.MockTransport -- no real sidecar."""

from __future__ import annotations

import json

import httpx
import pytest

from ace_client import AceClient, AceError


def _install_mock(monkeypatch, handler):
    """Monkeypatch httpx.AsyncClient so every constructor uses MockTransport."""
    transport = httpx.MockTransport(handler)
    real_cls = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_cls(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_health_success(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/health"
        return httpx.Response(200, json={
            "data": {"status": "ok", "models_initialized": True},
            "code": 200,
            "error": None,
            "timestamp": 1234567890,
            "extra": None,
        })

    _install_mock(monkeypatch, handler)
    c = AceClient(base_url="http://ace.test")
    result = await c.health()
    assert result["status"] == "ok"
    assert result["models_initialized"] is True


@pytest.mark.asyncio
async def test_generate_success(monkeypatch):
    """Full submit -> poll -> fetch cycle."""
    call_count = {"release": 0, "query": 0, "audio": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/release_task":
            call_count["release"] += 1
            body = json.loads(req.content)
            assert body["caption"] == "epic orchestral"
            return httpx.Response(200, json={
                "data": {"task_id": "test-task-123"},
                "code": 200, "error": None, "timestamp": 0, "extra": None,
            })
        elif req.url.path == "/query_result":
            call_count["query"] += 1
            body = json.loads(req.content)
            assert body["task_id_list"] == ["test-task-123"]
            # First poll: running; second poll: done
            if call_count["query"] == 1:
                return httpx.Response(200, json={
                    "data": [{"task_id": "test-task-123", "status": 0, "result": "[]", "progress_text": ""}],
                    "code": 200, "error": None, "timestamp": 0, "extra": None,
                })
            return httpx.Response(200, json={
                "data": [{
                    "task_id": "test-task-123", "status": 1,
                    "result": json.dumps([{"file": "/v1/audio?path=%2Ftmp%2Fsong.mp3", "status": 1}]),
                    "progress_text": "",
                }],
                "code": 200, "error": None, "timestamp": 0, "extra": None,
            })
        elif req.url.path == "/v1/audio":
            call_count["audio"] += 1
            assert "path=" in str(req.url)
            return httpx.Response(200, content=b"ID3fake-mp3-bytes", headers={"content-type": "audio/mpeg"})
        return httpx.Response(404)

    _install_mock(monkeypatch, handler)
    c = AceClient(base_url="http://ace.test", generate_timeout=10.0)

    progress_ticks: list[float] = []
    result = await c.generate(
        params={"caption": "epic orchestral", "audio_duration": 30},
        on_progress=lambda elapsed: progress_ticks.append(elapsed),
    )
    assert result == b"ID3fake-mp3-bytes"
    assert call_count["release"] == 1
    assert call_count["query"] == 2
    assert call_count["audio"] == 1
    assert len(progress_ticks) >= 1


@pytest.mark.asyncio
async def test_generate_failure(monkeypatch):
    """Task status=2 should raise AceError."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/release_task":
            return httpx.Response(200, json={
                "data": {"task_id": "fail-task"},
                "code": 200, "error": None, "timestamp": 0, "extra": None,
            })
        elif req.url.path == "/query_result":
            return httpx.Response(200, json={
                "data": [{"task_id": "fail-task", "status": 2, "result": "[]", "progress_text": ""}],
                "code": 200, "error": None, "timestamp": 0, "extra": None,
            })
        return httpx.Response(404)

    _install_mock(monkeypatch, handler)
    c = AceClient(base_url="http://ace.test", generate_timeout=10.0)
    with pytest.raises(AceError) as excinfo:
        await c.generate(params={"caption": "test"})
    assert excinfo.value.status_code == 500
    assert "ace_generation_failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_generate_timeout(monkeypatch):
    """Should raise AceError with 504 when timeout exceeded."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/release_task":
            return httpx.Response(200, json={
                "data": {"task_id": "slow-task"},
                "code": 200, "error": None, "timestamp": 0, "extra": None,
            })
        elif req.url.path == "/query_result":
            # Always running -- never completes
            return httpx.Response(200, json={
                "data": [{"task_id": "slow-task", "status": 0, "result": "[]", "progress_text": ""}],
                "code": 200, "error": None, "timestamp": 0, "extra": None,
            })
        return httpx.Response(404)

    _install_mock(monkeypatch, handler)
    # Very short timeout to trigger quickly
    c = AceClient(base_url="http://ace.test", generate_timeout=0.2)
    with pytest.raises(AceError) as excinfo:
        await c.generate(params={"caption": "test"})
    assert excinfo.value.status_code == 504
    assert "ace_timeout" in str(excinfo.value)


@pytest.mark.asyncio
async def test_generate_unreachable(monkeypatch):
    """ConnectError should map to 503."""
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    _install_mock(monkeypatch, handler)
    c = AceClient(base_url="http://ace.test")
    with pytest.raises(AceError) as excinfo:
        await c.generate(params={"caption": "test"})
    assert excinfo.value.status_code == 503
    assert "ace_unreachable" in str(excinfo.value)


@pytest.mark.asyncio
async def test_health_unreachable(monkeypatch):
    """Health check on unreachable sidecar should raise 503."""
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    _install_mock(monkeypatch, handler)
    c = AceClient(base_url="http://ace.test")
    with pytest.raises(AceError) as excinfo:
        await c.health()
    assert excinfo.value.status_code == 503
