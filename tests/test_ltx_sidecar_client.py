"""Tests for LtxSidecarClient using httpx.MockTransport — no real sidecar."""

from __future__ import annotations

import json

import httpx
import pytest

from ltx_sidecar_client import LtxSidecarClient, LtxSidecarError


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
            "status": "ready",
            "device": "cuda:1",
        })

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")
    result = await c.health()
    assert result["status"] == "ready"


@pytest.mark.asyncio
async def test_load_success(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/load"
        return httpx.Response(200, json={"status": "loaded"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")
    result = await c.load()
    assert result == {"status": "loaded"}


@pytest.mark.asyncio
async def test_load_failure(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "load_failed: CUDA OOM"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")
    with pytest.raises(LtxSidecarError) as excinfo:
        await c.load()
    assert excinfo.value.status_code == 500
    assert "load_failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_unload_already_unloaded(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "already_unloaded"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")
    result = await c.unload()
    assert result == {"status": "already_unloaded"}


@pytest.mark.asyncio
async def test_generate_success(monkeypatch):
    mp4_placeholder = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00"

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/generate"
        body = json.loads(req.content)
        assert body["prompt"] == "a cat"
        assert body["model"] == "ltx-2-3-fast"
        assert body["width"] == 768
        assert body["height"] == 512
        assert body["num_frames"] == 97
        assert body["seed"] == 42
        return httpx.Response(200, content=mp4_placeholder, headers={"content-type": "video/mp4"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")
    result = await c.generate(
        job_type="text_to_video",
        prompt="a cat",
        model="ltx-2-3-fast",
        width=768,
        height=512,
        num_frames=97,
        fps=24,
        seed=42,
    )
    assert result == mp4_placeholder
    assert isinstance(result, bytes)


@pytest.mark.asyncio
async def test_generate_pipeline_not_loaded(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "pipeline_not_loaded"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")
    with pytest.raises(LtxSidecarError) as excinfo:
        await c.generate(
            job_type="text_to_video", prompt="x", model="ltx-2-3-fast",
            width=768, height=512, num_frames=97, fps=24, seed=1,
        )
    assert excinfo.value.status_code == 503
    assert "pipeline_not_loaded" in str(excinfo.value)


@pytest.mark.asyncio
async def test_generate_cuda_oom(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "cuda_oom: CUDA out of memory"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")
    with pytest.raises(LtxSidecarError) as excinfo:
        await c.generate(
            job_type="text_to_video", prompt="x", model="ltx-2-3-fast",
            width=768, height=512, num_frames=97, fps=24, seed=1,
        )
    assert excinfo.value.status_code == 500
    assert "cuda_oom" in str(excinfo.value)


@pytest.mark.asyncio
async def test_generate_optional_fields_included(monkeypatch):
    """Optional fields (lora_path, keyframes, etc.) are included in the payload when provided."""

    captured_body = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, content=b"fake_mp4", headers={"content-type": "video/mp4"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")
    await c.generate(
        job_type="image_to_video",
        prompt="animate this",
        model="ltx-2-3-fast",
        width=768, height=512, num_frames=97, fps=24, seed=42,
        lora_path="/mnt/loras/test.safetensors",
        lora_strength=0.8,
        image_path="/tmp/input.png",
        keyframes=[{"frame_index": 0, "image_path": "/tmp/kf.png", "strength": 1.0}],
    )
    assert captured_body["lora_path"] == "/mnt/loras/test.safetensors"
    assert captured_body["lora_strength"] == 0.8
    assert captured_body["image_path"] == "/tmp/input.png"
    assert len(captured_body["keyframes"]) == 1


@pytest.mark.asyncio
async def test_generate_optional_fields_omitted(monkeypatch):
    """Optional fields are NOT included in payload when not provided (None)."""

    captured_body = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, content=b"fake_mp4", headers={"content-type": "video/mp4"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")
    await c.generate(
        job_type="text_to_video",
        prompt="a cat",
        model="ltx-2-3-fast",
        width=768, height=512, num_frames=97, fps=24, seed=42,
    )
    assert "lora_path" not in captured_body
    assert "keyframes" not in captured_body
    assert "audio_path" not in captured_body
    assert "image_path" not in captured_body
    assert "video_path" not in captured_body
    assert "start_time" not in captured_body
    assert "duration" not in captured_body
    assert "mode" not in captured_body
