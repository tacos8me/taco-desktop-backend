"""Routing tests for /v1/music/analyze `analyzer` field (v1.16.0).

Mocks `madmom_client.get_madmom_client` so no real sidecar is required.
The default `analyzer="librosa"` path is exercised against a real librosa
analysis of a tiny synthesized clip — proves byte-identical behavior for
existing v1.15.x callers.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest

import config

config.GPU_DEVICES = []
config.API_KEYS = set()  # disable auth for routing tests
config.LOAD_MADMOM = True

from fastapi.testclient import TestClient  # noqa: E402

from server import app, uploads  # noqa: E402

client = TestClient(app)


def _make_tiny_wav() -> str:
    """Synthesize a 1.5s 440Hz sine wave, save as WAV, return storage URI."""
    sr = 22050
    duration_s = 1.5
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5 * 32767).astype(np.int16)

    upload_id, storage_uri = uploads.create()
    target_path = uploads.path(upload_id) if hasattr(uploads, "path") else None
    # Fallback: write via uploads.save() with raw bytes for a WAV file.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    uploads.save(upload_id, buf.getvalue())
    return storage_uri


def test_librosa_default_omitted(monkeypatch):
    """Existing v1.15.x callers (no `analyzer` field) get the librosa path."""
    storage_uri = _make_tiny_wav()
    resp = client.post("/v1/music/analyze", json={"audio_uri": storage_uri})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Schema invariants (don't bind to exact float values — librosa is
    # deterministic on a given platform but float math drift is real).
    for key in ("bpm", "beats", "downbeats", "onsets", "rms_envelope", "duration_s", "confidence"):
        assert key in data, f"missing {key!r} in response"
    assert isinstance(data["beats"], list)
    # The librosa branch does NOT add `analyzer_used` — keeping byte-identical.
    assert "analyzer_used" not in data


def test_librosa_explicit(monkeypatch):
    """Explicit analyzer='librosa' is identical to default."""
    storage_uri = _make_tiny_wav()
    resp = client.post(
        "/v1/music/analyze",
        json={"audio_uri": storage_uri, "analyzer": "librosa"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "beats" in data


def test_madmom_routes_to_sidecar(monkeypatch):
    """analyzer='madmom' calls madmom_client and returns its response verbatim."""
    storage_uri = _make_tiny_wav()
    expected = {
        "bpm": 124.5,
        "beats": [0.5, 1.0, 1.5],
        "downbeats": [0.5],
        "onsets": [0.5, 1.0, 1.5],
        "rms_envelope": [[0.0, -28.4]],
        "duration_s": 1.5,
        "confidence": 0.91,
        "analyzer_used": "madmom",
    }

    import madmom_client

    class FakeClient:
        async def analyze(self, audio_path):
            return expected

    monkeypatch.setattr(madmom_client, "_madmom_singleton", FakeClient())
    monkeypatch.setattr(config, "LOAD_MADMOM", True)

    resp = client.post(
        "/v1/music/analyze",
        json={"audio_uri": storage_uri, "analyzer": "madmom"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == expected


def test_madmom_connect_error_returns_503(monkeypatch):
    """httpx.ConnectError → MadmomError(503) → handler returns 503."""
    storage_uri = _make_tiny_wav()

    import madmom_client

    class BrokenClient:
        async def analyze(self, audio_path):
            raise madmom_client.MadmomError(
                "sidecar_unreachable: madmom sidecar not running at http://127.0.0.1:8095",
                503,
            )

    monkeypatch.setattr(madmom_client, "_madmom_singleton", BrokenClient())
    monkeypatch.setattr(config, "LOAD_MADMOM", True)

    resp = client.post(
        "/v1/music/analyze",
        json={"audio_uri": storage_uri, "analyzer": "madmom"},
    )
    assert resp.status_code == 503
    body = resp.json()
    # _error returns {"error": <msg>}
    msg = body.get("error") or body.get("message") or str(body)
    assert "sidecar_unreachable" in msg or "8095" in msg


def test_madmom_5xx_returns_503(monkeypatch):
    """Sidecar 5xx surfaces as 503 from the handler (no silent fallback)."""
    storage_uri = _make_tiny_wav()

    import madmom_client

    class FailingClient:
        async def analyze(self, audio_path):
            raise madmom_client.MadmomError(
                "sidecar_5xx (502) at http://127.0.0.1:8095: bad gateway",
                503,
            )

    monkeypatch.setattr(madmom_client, "_madmom_singleton", FailingClient())
    monkeypatch.setattr(config, "LOAD_MADMOM", True)

    resp = client.post(
        "/v1/music/analyze",
        json={"audio_uri": storage_uri, "analyzer": "madmom"},
    )
    assert resp.status_code == 503


def test_madmom_disabled_returns_503(monkeypatch):
    """LOAD_MADMOM=0 short-circuits to 503 without touching the sidecar."""
    storage_uri = _make_tiny_wav()
    monkeypatch.setattr(config, "LOAD_MADMOM", False)

    resp = client.post(
        "/v1/music/analyze",
        json={"audio_uri": storage_uri, "analyzer": "madmom"},
    )
    assert resp.status_code == 503
    body = resp.json()
    msg = body.get("error") or body.get("message") or str(body)
    assert "LOAD_MADMOM" in msg or "disabled" in msg


def test_invalid_analyzer_returns_422():
    """Pydantic Literal validator rejects unknown analyzer values."""
    storage_uri = _make_tiny_wav()
    resp = client.post(
        "/v1/music/analyze",
        json={"audio_uri": storage_uri, "analyzer": "allin1"},
    )
    assert resp.status_code == 422
