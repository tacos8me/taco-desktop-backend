"""HTTP client for the LTX video sidecar (turbo mode v1.2).

The LTX sidecar runs an independent LTX pipeline on cuda:1 at
`config.LTX_SIDECAR_URL` (default http://127.0.0.1:8093). This enables
dual-GPU turbo mode — 2 concurrent video jobs without loading a second
SplitModelManager in-process.

    from ltx_sidecar_client import ltx_sidecar, LtxSidecarError

Every call opens a short-lived `httpx.AsyncClient` — we intentionally do
NOT reuse a long-lived client because the sidecar may restart and stale
connections can stick. Timeouts are split: `generate_timeout` covers the
long inference call, `mgmt_timeout` covers /health, /load, /unload.
"""

from __future__ import annotations

import logging
import time

import httpx

import config

logger = logging.getLogger(__name__)


class LtxSidecarError(ValueError):
    """Raised on sidecar errors. Carries `status_code` for HTTP mapping."""

    def __init__(self, msg: str, status_code: int = 500):
        super().__init__(msg)
        self.status_code = status_code


class LtxSidecarClient:
    """HTTP client for the LTX video sidecar at LTX_SIDECAR_URL."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        generate_timeout: float = 300.0,
        mgmt_timeout: float = 60.0,
    ):
        self._base_url = base_url if base_url is not None else config.LTX_SIDECAR_URL
        self._generate_timeout = generate_timeout
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
        """Run a management (non-generate) request with the short timeout."""
        url = f"{self._base_url}{path}"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._mgmt_timeout) as client:
                resp = await client.request(method, url)
        except httpx.TimeoutException as exc:
            logger.info("LTX sidecar %s %s TIMEOUT in %.2fs", method, path, time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_timeout: {exc}", 504) from exc
        except httpx.ConnectError as exc:
            logger.info("LTX sidecar %s %s UNREACHABLE in %.2fs", method, path, time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_unreachable: {exc}", 503) from exc
        except httpx.HTTPError as exc:
            logger.info("LTX sidecar %s %s HTTP error in %.2fs", method, path, time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info("LTX sidecar %s %s → %d in %.2fs", method, path, resp.status_code, dt)
        if resp.status_code >= 400:
            raise LtxSidecarError(self._extract_error(resp), resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise LtxSidecarError(f"invalid_json: {exc}", 502) from exc

    # --- public API ---

    async def health(self) -> dict:
        """GET /health — returns the status dict or raises LtxSidecarError on network failure."""
        return await self._mgmt_request("GET", "/health")

    async def load(self) -> dict:
        """POST /load — idempotent, ensures pipeline loaded on cuda:1."""
        return await self._mgmt_request("POST", "/load")

    async def unload(self) -> dict:
        """POST /unload — free cuda:1 GPU memory."""
        return await self._mgmt_request("POST", "/unload")

    async def generate(
        self,
        *,
        job_type: str,
        prompt: str,
        model: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        generate_audio: bool = False,
        lora_path: str | None = None,
        lora_strength: float = 1.0,
        enhance_prompt: bool = False,
        keyframes: list | None = None,
        audio_path: str | None = None,
        image_path: str | None = None,
        video_path: str | None = None,
        start_time: float | None = None,
        duration: float | None = None,
        mode: str | None = None,
    ) -> bytes:
        """POST /generate — returns raw MP4 bytes.

        Status code mapping:
          503 → LtxSidecarError("pipeline_not_loaded", 503)
          500 → LtxSidecarError("cuda_oom: ..." or "generate_failed: ...", 500)
          timeout → LtxSidecarError("sidecar_timeout", 504)
          connection refused → LtxSidecarError("sidecar_unreachable", 503)
        """
        payload: dict = {
            "job_type": job_type,
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "generate_audio": generate_audio,
            "lora_strength": lora_strength,
            "enhance_prompt": enhance_prompt,
        }
        if lora_path is not None:
            payload["lora_path"] = lora_path
        if keyframes is not None:
            payload["keyframes"] = keyframes
        if audio_path is not None:
            payload["audio_path"] = audio_path
        if image_path is not None:
            payload["image_path"] = image_path
        if video_path is not None:
            payload["video_path"] = video_path
        if start_time is not None:
            payload["start_time"] = start_time
        if duration is not None:
            payload["duration"] = duration
        if mode is not None:
            payload["mode"] = mode

        url = f"{self._base_url}/generate"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._generate_timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            logger.info("LTX sidecar POST /generate TIMEOUT in %.2fs", time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_timeout: {exc}", 504) from exc
        except httpx.ConnectError as exc:
            logger.info("LTX sidecar POST /generate UNREACHABLE in %.2fs", time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_unreachable: {exc}", 503) from exc
        except httpx.HTTPError as exc:
            logger.info("LTX sidecar POST /generate HTTP error in %.2fs", time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info(
            "LTX sidecar POST /generate → %d in %.2fs (%d bytes)",
            resp.status_code,
            dt,
            len(resp.content),
        )
        if resp.status_code == 200:
            return resp.content
        msg = self._extract_error(resp)
        if resp.status_code == 503:
            raise LtxSidecarError("pipeline_not_loaded", 503)
        if resp.status_code == 500:
            raise LtxSidecarError(msg, 500)
        raise LtxSidecarError(msg, resp.status_code)


# Module-level singleton. Consumers: `from ltx_sidecar_client import ltx_sidecar, LtxSidecarError`.
ltx_sidecar = LtxSidecarClient()
