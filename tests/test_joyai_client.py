"""Tests for JoyAIClient using httpx.MockTransport — no real sidecar."""

from __future__ import annotations

import httpx
import pytest

from joyai_client import JoyAIClient, JoyAIError


def _install_mock(monkeypatch, handler):
    """Monkeypatch httpx.AsyncClient so every constructor uses MockTransport."""
    transport = httpx.MockTransport(handler)
    real_cls = httpx.AsyncClient

    def factory(*args, **kwargs):
        # Strip any transport kwarg from the client's call — we force ours.
        kwargs.pop("transport", None)
        return real_cls(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_health_success(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/health"
        return httpx.Response(200, json={
            "status": "ready",
            "model_path": "/mnt/models/joyai",
            "device": "cuda:0",
            "peak_vram_gb": 65.0,
        })

    _install_mock(monkeypatch, handler)
    c = JoyAIClient(base_url="http://joyai.test")
    result = await c.health()
    assert result["status"] == "ready"
    assert result["peak_vram_gb"] == 65.0


@pytest.mark.asyncio
async def test_load_success(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/load"
        return httpx.Response(200, json={"status": "loaded"})

    _install_mock(monkeypatch, handler)
    c = JoyAIClient(base_url="http://joyai.test")
    result = await c.load()
    assert result == {"status": "loaded"}


@pytest.mark.asyncio
async def test_load_failure(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={
            "error": "load_failed: CUDA OOM",
            "message": "load_failed: CUDA OOM",
            "detail": "load_failed: CUDA OOM",
        })

    _install_mock(monkeypatch, handler)
    c = JoyAIClient(base_url="http://joyai.test")
    with pytest.raises(JoyAIError) as excinfo:
        await c.load()
    assert excinfo.value.status_code == 500
    assert "load_failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_unload_already_unloaded(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "already_unloaded"})

    _install_mock(monkeypatch, handler)
    c = JoyAIClient(base_url="http://joyai.test")
    result = await c.unload()
    assert result == {"status": "already_unloaded"}


@pytest.mark.asyncio
async def test_edit_success(monkeypatch):
    webp_placeholder = b"RIFF\x00\x00\x00\x00WEBPVP8 fake"

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/edit"
        return httpx.Response(200, content=webp_placeholder, headers={"content-type": "image/webp"})

    _install_mock(monkeypatch, handler)
    c = JoyAIClient(base_url="http://joyai.test")
    result = await c.edit(
        prompt="make it a cat",
        image_path="/tmp/in.png",
        width=1024,
        height=1024,
    )
    assert result == webp_placeholder
    assert isinstance(result, bytes)


@pytest.mark.asyncio
async def test_edit_image_not_found(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={
            "error": "image_path not found",
            "message": "image_path not found",
            "detail": "image_path not found",
        })

    _install_mock(monkeypatch, handler)
    c = JoyAIClient(base_url="http://joyai.test")
    with pytest.raises(JoyAIError) as excinfo:
        await c.edit(prompt="x", image_path="/missing.png", width=512, height=512)
    assert excinfo.value.status_code == 404
    assert "image_not_found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_edit_pipeline_not_loaded(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={
            "error": "pipeline_not_loaded",
            "message": "pipeline_not_loaded",
            "detail": "pipeline_not_loaded",
        })

    _install_mock(monkeypatch, handler)
    c = JoyAIClient(base_url="http://joyai.test")
    with pytest.raises(JoyAIError) as excinfo:
        await c.edit(prompt="x", image_path="/x.png", width=512, height=512)
    assert excinfo.value.status_code == 503
    assert "pipeline_not_loaded" in str(excinfo.value)
