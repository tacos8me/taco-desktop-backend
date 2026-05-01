"""Tests for the LAN dashboard proxy."""

from __future__ import annotations

from fastapi.testclient import TestClient

import dashboard_server


class _DecodedResponse:
    status_code = 200
    content = b'{"ok":true}'
    headers = {
        "content-encoding": "gzip",
        "content-length": "999",
        "content-type": "application/json",
    }

    async def aiter_bytes(self):
        yield self.content


def test_dashboard_proxy_strips_decoded_content_encoding(monkeypatch):
    seen_headers: dict[str, str] = {}

    async def fake_request(method, url, headers=None, content=None):
        nonlocal seen_headers
        seen_headers = {k.lower(): v for k, v in dict(headers or {}).items()}
        return _DecodedResponse()

    monkeypatch.setattr(dashboard_server._http, "request", fake_request)

    client = TestClient(dashboard_server.app)
    resp = client.get(
        "/v1/system/validator-stats?window=24h",
        headers={"accept-encoding": "gzip, deflate, br"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert "accept-encoding" not in seen_headers
    assert "content-encoding" not in resp.headers
    assert resp.headers["content-type"].startswith("application/json")
