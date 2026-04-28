"""HTTP client for the madmom downbeat-detection sidecar (v1.16.0).

madmom needs numpy<1.24 + scipy<1.13 (binary-compat ceiling), incompatible
with the main taco-backend venv. It runs out-of-process at
`config.MADMOM_SIDECAR_URL` (default http://127.0.0.1:8095) inside its own
isolated venv. CPU-only, BSD-licensed.

Mirrors the joyai_client.py pattern: short-lived `httpx.AsyncClient` per
call (the sidecar may restart and stale connections stick), split timeouts
(`analyze_timeout` long, `mgmt_timeout` short).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

import config

logger = logging.getLogger(__name__)


class MadmomError(ValueError):
    """Raised on sidecar errors. Carries `status_code` for HTTP mapping."""

    def __init__(self, msg: str, status_code: int = 503):
        super().__init__(msg)
        self.status_code = status_code


class MadmomClient:
    """HTTP client for the madmom sidecar at MADMOM_SIDECAR_URL."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        analyze_timeout: float = 120.0,
        mgmt_timeout: float = 10.0,
    ):
        self._base_url = base_url if base_url is not None else config.MADMOM_SIDECAR_URL
        self._analyze_timeout = analyze_timeout
        self._mgmt_timeout = mgmt_timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    def _extract_error(self, resp: httpx.Response) -> str:
        try:
            data = resp.json()
            return data.get("error") or data.get("message") or data.get("detail") or resp.text
        except ValueError:
            return resp.text or f"http_{resp.status_code}"

    async def health(self) -> bool:
        """GET /health — True iff sidecar reports `ready: true`. False on any error."""
        url = f"{self._base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=self._mgmt_timeout) as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            logger.info("madmom GET /health unreachable: %s", exc)
            return False
        if resp.status_code != 200:
            return False
        try:
            return bool(resp.json().get("ready"))
        except ValueError:
            return False

    async def analyze(self, audio_path: Path | str) -> dict:
        """POST /v1/analyze — returns the sidecar response body verbatim.

        Raises MadmomError on any transport / HTTP error. Status code on the
        exception is what the server handler should surface (503 for
        unreachable / 5xx, 504 for timeout, etc.).
        """
        url = f"{self._base_url}/v1/analyze"
        payload = {"audio_path": str(audio_path)}
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._analyze_timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            logger.info("madmom POST /v1/analyze TIMEOUT in %.2fs", time.perf_counter() - t0)
            raise MadmomError(f"sidecar_timeout: {exc}", 504) from exc
        except httpx.ConnectError as exc:
            logger.info(
                "madmom POST /v1/analyze UNREACHABLE in %.2fs at %s",
                time.perf_counter() - t0,
                self._base_url,
            )
            raise MadmomError(
                f"sidecar_unreachable: madmom sidecar not running at {self._base_url}",
                503,
            ) from exc
        except httpx.HTTPError as exc:
            logger.info("madmom POST /v1/analyze HTTP error in %.2fs", time.perf_counter() - t0)
            raise MadmomError(f"sidecar_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info("madmom POST /v1/analyze → %d in %.2fs", resp.status_code, dt)

        if resp.status_code >= 500:
            raise MadmomError(
                f"sidecar_5xx ({resp.status_code}) at {self._base_url}: "
                f"{self._extract_error(resp)}",
                503,
            )
        if resp.status_code >= 400:
            raise MadmomError(self._extract_error(resp), resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise MadmomError(f"invalid_json: {exc}", 502) from exc


# Lazy module-level singleton — instantiated on first use to avoid eager
# import-time HTTP setup when LOAD_MADMOM=0.
_madmom_singleton: MadmomClient | None = None


def get_madmom_client() -> MadmomClient:
    global _madmom_singleton
    if _madmom_singleton is None:
        _madmom_singleton = MadmomClient()
    return _madmom_singleton
