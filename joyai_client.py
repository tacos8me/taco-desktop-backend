"""HTTP client for the JoyAI image-edit sidecar (v1.1.8).

The JoyAI model needs incompatible diffusers/transformers versions vs. the
main taco-backend process, so it runs out-of-process in a sidecar at
`config.JOYAI_SIDECAR_URL` (default http://127.0.0.1:8092). This module
provides a thin async client that consumers import as:

    from joyai_client import joyai, JoyAIError

Every call opens a short-lived `httpx.AsyncClient` — we intentionally do
NOT reuse a long-lived client because the sidecar may restart and stale
connections can stick. Timeouts are split: `edit_timeout` covers the long
inference call, `mgmt_timeout` covers /health, /load, /unload.
"""

from __future__ import annotations

import logging
import time

import httpx

import config

logger = logging.getLogger(__name__)


class JoyAIError(ValueError):
    """Raised on sidecar errors. Carries `status_code` for HTTP mapping."""

    def __init__(self, msg: str, status_code: int = 500):
        super().__init__(msg)
        self.status_code = status_code


class JoyAIClient:
    """HTTP client for the JoyAI edit sidecar at JOYAI_SIDECAR_URL."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        edit_timeout: float = 180.0,
        mgmt_timeout: float = 60.0,
    ):
        self._base_url = base_url if base_url is not None else config.JOYAI_SIDECAR_URL
        self._edit_timeout = edit_timeout
        self._mgmt_timeout = mgmt_timeout

    # --- internal helpers ---

    def _extract_error(self, resp: httpx.Response) -> str:
        """Pull the error message out of a sidecar error response."""
        try:
            data = resp.json()
            return data.get("error") or data.get("message") or data.get("detail") or resp.text
        except ValueError:
            return resp.text or f"http_{resp.status_code}"

    async def _mgmt_request(self, method: str, path: str) -> dict:
        """Run a management (non-edit) request with the short timeout."""
        url = f"{self._base_url}{path}"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._mgmt_timeout) as client:
                resp = await client.request(method, url)
        except httpx.TimeoutException as exc:
            logger.info("JoyAI %s %s TIMEOUT in %.2fs", method, path, time.perf_counter() - t0)
            raise JoyAIError(f"sidecar_timeout: {exc}", 504) from exc
        except httpx.ConnectError as exc:
            logger.info(
                "JoyAI %s %s UNREACHABLE in %.2fs", method, path, time.perf_counter() - t0
            )
            raise JoyAIError(f"sidecar_unreachable: {exc}", 503) from exc
        except httpx.HTTPError as exc:
            logger.info("JoyAI %s %s HTTP error in %.2fs", method, path, time.perf_counter() - t0)
            raise JoyAIError(f"sidecar_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info("JoyAI %s %s → %d in %.2fs", method, path, resp.status_code, dt)
        if resp.status_code >= 400:
            raise JoyAIError(self._extract_error(resp), resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise JoyAIError(f"invalid_json: {exc}", 502) from exc

    # --- public API ---

    async def health(self) -> dict:
        """GET /health — returns the status dict or raises JoyAIError on network failure."""
        return await self._mgmt_request("GET", "/health")

    async def load(self) -> dict:
        """POST /load — idempotent. Returns status dict. Raises JoyAIError(500, ...) on load_failed."""
        return await self._mgmt_request("POST", "/load")

    async def unload(self) -> dict:
        """POST /unload — idempotent. Returns status dict. Raises JoyAIError(500, ...) on unload_failed."""
        return await self._mgmt_request("POST", "/unload")

    async def edit(
        self,
        *,
        prompt: str,
        image_path: str,
        width: int,
        height: int,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        seed: int | None = None,
    ) -> bytes:
        """POST /edit — returns raw WEBP bytes. Raises JoyAIError on any error response.

        Status code mapping:
          404 → JoyAIError("image_not_found: ...", 404)
          503 → JoyAIError("pipeline_not_loaded", 503)
          500 → JoyAIError("cuda_oom: ..." or "edit_failed: ...", 500)
          timeout → JoyAIError("sidecar_timeout", 504)
          connection refused → JoyAIError("sidecar_unreachable", 503)
        """
        payload = {
            "prompt": prompt,
            "image_path": image_path,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        url = f"{self._base_url}/edit"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._edit_timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            logger.info("JoyAI POST /edit TIMEOUT in %.2fs", time.perf_counter() - t0)
            raise JoyAIError(f"sidecar_timeout: {exc}", 504) from exc
        except httpx.ConnectError as exc:
            logger.info("JoyAI POST /edit UNREACHABLE in %.2fs", time.perf_counter() - t0)
            raise JoyAIError(f"sidecar_unreachable: {exc}", 503) from exc
        except httpx.HTTPError as exc:
            logger.info("JoyAI POST /edit HTTP error in %.2fs", time.perf_counter() - t0)
            raise JoyAIError(f"sidecar_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info(
            "JoyAI POST /edit → %d in %.2fs (%d bytes)",
            resp.status_code,
            dt,
            len(resp.content),
        )
        if resp.status_code == 200:
            return resp.content
        msg = self._extract_error(resp)
        if resp.status_code == 404:
            raise JoyAIError(f"image_not_found: {msg}", 404)
        if resp.status_code == 503:
            raise JoyAIError("pipeline_not_loaded", 503)
        if resp.status_code == 500:
            # Preserve cuda_oom/edit_failed prefix from the sidecar when present.
            raise JoyAIError(msg, 500)
        raise JoyAIError(msg, resp.status_code)


# Module-level singleton. Consumers: `from joyai_client import joyai, JoyAIError`.
joyai = JoyAIClient()
