"""HTTP client for the sapiens-sidecar pose-temporal-stability service (v1.17.0-rc2).

Mirrors `madmom_client.py`: short-lived `httpx.AsyncClient` per call (the
sidecar may restart and stale connections stick), split timeouts. The sidecar
runs on cuda:1 alongside ACE; it's stopped by `_stop_cuda1_tenants` on turbo
entry like the other cuda:1 tenants.

In v1.17.0-rc2 the sidecar is stub-mode (rc1 ships the sidecar shell; real
inference is rc2-side). The client tolerates `{"stub": true}` payloads and
passes them through verbatim — caller (validator.py) decides how to score.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

import config

logger = logging.getLogger(__name__)


class SapiensError(ValueError):
    """Raised on sidecar errors. Carries `status_code` for HTTP mapping."""

    def __init__(self, msg: str, status_code: int = 503):
        super().__init__(msg)
        self.status_code = status_code


class SapiensClient:
    """HTTP client for the sapiens sidecar at SAPIENS_SIDECAR_URL."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        analyze_timeout: float | None = None,
        mgmt_timeout: float = 10.0,
    ):
        self._base_url = base_url if base_url is not None else config.SAPIENS_SIDECAR_URL
        self._analyze_timeout = (
            analyze_timeout if analyze_timeout is not None else config.SAPIENS_TIMEOUT_S
        )
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
            logger.info("sapiens GET /health unreachable: %s", exc)
            return False
        if resp.status_code != 200:
            return False
        try:
            return bool(resp.json().get("ready"))
        except ValueError:
            return False

    async def analyze_pose(self, video_path: Path | str) -> dict:
        """POST /v1/analyze-pose — returns the sidecar response body verbatim.

        Caller is responsible for resolving any `storage://` URI to an
        absolute path before invoking — the sidecar can't see the local
        uploads/ filesystem and consumes paths directly.

        Stub responses (`{"stub": true, ...}`) are passed through; the
        validator will treat them as tier-2 skipped, not failed.

        Raises SapiensError on any transport / HTTP error. Status code on
        the exception is what the server handler should surface (503 for
        unreachable / 5xx, 504 for timeout, etc.).
        """
        url = f"{self._base_url}/v1/analyze-pose"
        payload = {"video_path": str(video_path)}
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._analyze_timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            logger.info("sapiens POST /v1/analyze-pose TIMEOUT in %.2fs", time.perf_counter() - t0)
            raise SapiensError(f"sidecar_timeout: {exc}", 504) from exc
        except httpx.ConnectError as exc:
            logger.info(
                "sapiens POST /v1/analyze-pose UNREACHABLE in %.2fs at %s",
                time.perf_counter() - t0,
                self._base_url,
            )
            raise SapiensError(
                f"sidecar_unreachable: sapiens sidecar not running at {self._base_url}",
                503,
            ) from exc
        except httpx.HTTPError as exc:
            logger.info("sapiens POST /v1/analyze-pose HTTP error in %.2fs", time.perf_counter() - t0)
            raise SapiensError(f"sidecar_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info("sapiens POST /v1/analyze-pose → %d in %.2fs", resp.status_code, dt)

        if resp.status_code >= 500:
            raise SapiensError(
                f"sidecar_5xx ({resp.status_code}) at {self._base_url}: "
                f"{self._extract_error(resp)}",
                503,
            )
        if resp.status_code >= 400:
            raise SapiensError(self._extract_error(resp), resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise SapiensError(f"invalid_json: {exc}", 502) from exc


# Lazy module-level singleton — instantiated on first use to avoid eager
# import-time HTTP setup when LOAD_SAPIENS=0.
_sapiens_singleton: SapiensClient | None = None


def get_sapiens_client() -> SapiensClient:
    global _sapiens_singleton
    if _sapiens_singleton is None:
        _sapiens_singleton = SapiensClient()
    return _sapiens_singleton
