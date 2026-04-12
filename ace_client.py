"""HTTP client for the ACE-Step music generation sidecar (v1.2).

ACE-Step runs out-of-process on cuda:1 at `config.ACE_SIDECAR_URL`
(default http://127.0.0.1:8001). This module provides a thin async client
that consumers import as:

    from ace_client import ace, AceError

Uses short-lived httpx.AsyncClient instances — the sidecar may restart
and stale connections can stick. ACE wraps all responses in an envelope
``{"data": ..., "code": 200, "error": null, ...}``; we unwrap ``.data``
before returning.

Submit-poll-fetch flow:
  1. POST /release_task → get task_id
  2. Poll POST /query_result every 0.5s → status 0=running, 1=done, 2=failed
  3. GET /v1/audio?path=<url-encoded-path> → raw audio bytes
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.parse
from typing import Callable

import httpx

import config

logger = logging.getLogger(__name__)


class AceError(ValueError):
    """Raised on ACE sidecar errors. Carries ``status_code`` for HTTP mapping."""

    def __init__(self, msg: str, status_code: int = 500):
        super().__init__(msg)
        self.status_code = status_code


class AceClient:
    """HTTP client for the ACE-Step music sidecar."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        generate_timeout: float = 600.0,
        mgmt_timeout: float = 30.0,
    ):
        self._base_url = base_url if base_url is not None else config.ACE_SIDECAR_URL
        self._generate_timeout = generate_timeout
        self._mgmt_timeout = mgmt_timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unwrap(self, data: dict) -> dict:
        """Unwrap ACE envelope ``{"data": ..., "code": ..., "error": ...}``."""
        if data.get("error"):
            raise AceError(str(data["error"]), data.get("code", 500))
        return data.get("data", data)

    async def _post_json(self, path: str, body: dict, timeout: float) -> dict:
        """POST with JSON body, unwrap ACE envelope, return ``.data``."""
        url = f"{self._base_url}{path}"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=body)
        except httpx.ConnectError as exc:
            logger.info("ACE POST %s UNREACHABLE in %.2fs", path, time.perf_counter() - t0)
            raise AceError(f"ace_unreachable: {exc}", 503) from exc
        except httpx.TimeoutException as exc:
            logger.info("ACE POST %s TIMEOUT in %.2fs", path, time.perf_counter() - t0)
            raise AceError(f"ace_timeout: {exc}", 504) from exc
        except httpx.HTTPError as exc:
            logger.info("ACE POST %s HTTP error in %.2fs", path, time.perf_counter() - t0)
            raise AceError(f"ace_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info("ACE POST %s -> %d in %.2fs", path, resp.status_code, dt)
        if resp.status_code >= 400:
            try:
                err_data = resp.json()
                msg = err_data.get("error") or err_data.get("message") or resp.text
            except ValueError:
                msg = resp.text or f"http_{resp.status_code}"
            raise AceError(str(msg), resp.status_code)
        try:
            return self._unwrap(resp.json())
        except ValueError as exc:
            raise AceError(f"invalid_json: {exc}", 502) from exc

    async def _get_json(self, path: str, timeout: float) -> dict:
        """GET JSON, unwrap ACE envelope, return ``.data``."""
        url = f"{self._base_url}{path}"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
        except httpx.ConnectError as exc:
            logger.info("ACE GET %s UNREACHABLE in %.2fs", path, time.perf_counter() - t0)
            raise AceError(f"ace_unreachable: {exc}", 503) from exc
        except httpx.TimeoutException as exc:
            logger.info("ACE GET %s TIMEOUT in %.2fs", path, time.perf_counter() - t0)
            raise AceError(f"ace_timeout: {exc}", 504) from exc
        except httpx.HTTPError as exc:
            logger.info("ACE GET %s HTTP error in %.2fs", path, time.perf_counter() - t0)
            raise AceError(f"ace_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info("ACE GET %s -> %d in %.2fs", path, resp.status_code, dt)
        if resp.status_code >= 400:
            try:
                err_data = resp.json()
                msg = err_data.get("error") or err_data.get("message") or resp.text
            except ValueError:
                msg = resp.text or f"http_{resp.status_code}"
            raise AceError(str(msg), resp.status_code)
        try:
            return self._unwrap(resp.json())
        except ValueError as exc:
            raise AceError(f"invalid_json: {exc}", 502) from exc

    async def _get_bytes(self, path: str, timeout: float) -> bytes:
        """GET raw bytes (for audio download)."""
        url = f"{self._base_url}{path}"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
        except httpx.ConnectError as exc:
            logger.info("ACE GET %s UNREACHABLE in %.2fs", path, time.perf_counter() - t0)
            raise AceError(f"ace_unreachable: {exc}", 503) from exc
        except httpx.TimeoutException as exc:
            logger.info("ACE GET %s TIMEOUT in %.2fs", path, time.perf_counter() - t0)
            raise AceError(f"ace_timeout: {exc}", 504) from exc
        except httpx.HTTPError as exc:
            logger.info("ACE GET %s HTTP error in %.2fs", path, time.perf_counter() - t0)
            raise AceError(f"ace_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info("ACE GET %s -> %d in %.2fs (%d bytes)", path, resp.status_code, dt, len(resp.content))
        if resp.status_code >= 400:
            try:
                err_data = resp.json()
                msg = err_data.get("error") or err_data.get("message") or resp.text
            except ValueError:
                msg = resp.text or f"http_{resp.status_code}"
            raise AceError(str(msg), resp.status_code)
        return resp.content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        """GET /health -- unwrap ACE envelope, return .data dict."""
        return await self._get_json("/health", self._mgmt_timeout)

    async def generate(
        self,
        *,
        params: dict,
        on_progress: Callable[[float], None] | None = None,
    ) -> bytes:
        """Submit -> poll -> fetch. Returns raw audio bytes.

        1. POST /release_task with params -> get task_id
        2. Poll POST /query_result every 0.5s until done or failed
        3. GET /v1/audio?path=<first audio_path> -> raw bytes
        4. Timeout after self._generate_timeout -> AceError("ace_timeout", 504)
        """
        # 1. Submit task
        submit_data = await self._post_json("/release_task", params, self._mgmt_timeout)
        task_id = submit_data.get("task_id")
        if not task_id:
            raise AceError("ace_no_task_id: /release_task returned no task_id", 502)
        logger.info("ACE task submitted: %s", task_id)

        # 2. Poll until done
        deadline = time.monotonic() + self._generate_timeout
        poll_body = {"task_id_list": [task_id]}
        while True:
            if time.monotonic() > deadline:
                raise AceError("ace_timeout", 504)

            await asyncio.sleep(0.5)
            elapsed = time.monotonic() - (deadline - self._generate_timeout)
            if on_progress:
                on_progress(elapsed)

            result_data = await self._post_json("/query_result", poll_body, self._mgmt_timeout)
            # ACE returns data as a list directly, or a dict with "results" key
            if isinstance(result_data, list):
                results = result_data
            else:
                results = result_data.get("results", [])
            if not results:
                continue

            r = results[0]
            status = r.get("status")
            if status == 0:
                # Still running
                continue
            elif status == 2:
                # Failed
                raise AceError(f"ace_generation_failed: task {task_id}", 500)
            elif status == 1:
                # Succeeded -- parse the result JSON string to find audio URLs.
                # ACE returns result as a JSON-encoded string containing an array
                # of objects, each with a "file" key holding the audio download URL
                # (e.g. "/v1/audio?path=%2Fmnt%2F...%2Faudio.mp3").
                result_str = r.get("result", "[]")
                try:
                    import json as _json
                    result_items = _json.loads(result_str) if isinstance(result_str, str) else result_str
                except (ValueError, TypeError):
                    result_items = []
                if not result_items or not isinstance(result_items, list):
                    raise AceError("ace_no_audio: task succeeded but result is empty or unparseable", 502)
                audio_url = result_items[0].get("file")
                if not audio_url:
                    raise AceError("ace_no_audio: result item has no 'file' key", 502)
                logger.info("ACE task %s completed, %d outputs, downloading first", task_id, len(result_items))
                # audio_url is already a full path like "/v1/audio?path=%2Fmnt%2F..."
                audio_bytes = await self._get_bytes(
                    audio_url,
                    self._mgmt_timeout,
                )
                return audio_bytes
            else:
                raise AceError(f"ace_unknown_status: {status}", 502)


# Module-level singleton. Consumers: ``from ace_client import ace, AceError``.
ace = AceClient()
