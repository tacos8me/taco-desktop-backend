"""Async HTTP client for the ERNIE-Image sidecar (port 8094).

Mirrors joyai_client.py pattern. The sidecar runs on cuda:1 via
CUDA_VISIBLE_DEVICES=1 and swaps with JoyAI (mutual exclusion).
"""
from __future__ import annotations

import logging
import time

import httpx

import config

logger = logging.getLogger(__name__)


class ErnieError(Exception):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


class ErnieClient:
    """Thin async wrapper around the ERNIE-Image sidecar HTTP API."""

    def __init__(
        self,
        base_url: str = config.ERNIE_SIDECAR_URL,
        generate_timeout: float = 180.0,
        mgmt_timeout: float = 60.0,
    ):
        self._base_url = base_url
        self._generate_timeout = generate_timeout
        self._mgmt_timeout = mgmt_timeout
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def _mgmt(self, method: str, path: str) -> dict:
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._mgmt_timeout) as client:
                resp = await client.request(method, f"{self._base_url}{path}")
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            logger.warning("ERNIE sidecar %s %s TIMEOUT in %.2fs", method, path, time.perf_counter() - t0)
            raise ErnieError(f"sidecar_timeout: {method} {path}", 504)
        except httpx.ConnectError:
            raise ErnieError("sidecar_unreachable: ERNIE-Image sidecar not running", 503)
        except Exception as exc:
            raise ErnieError(f"sidecar_error: {exc}", 502)

    async def health(self) -> dict:
        return await self._mgmt("GET", "/health")

    async def load(self) -> None:
        if self._loaded:
            return
        result = await self._mgmt("POST", "/load")
        self._loaded = result.get("status") == "ready"
        logger.info("ERNIE sidecar load: %s", result.get("status"))

    async def unload(self) -> None:
        if not self._loaded:
            return
        await self._mgmt("POST", "/unload")
        self._loaded = False
        logger.info("ERNIE sidecar unloaded")

    async def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        use_pe: bool = True,
    ) -> bytes:
        """Generate an image and return WEBP bytes."""
        if not self._loaded:
            await self.load()

        body = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "use_pe": use_pe,
        }
        if seed is not None:
            body["seed"] = seed

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._generate_timeout) as client:
                resp = await client.post(f"{self._base_url}/generate", json=body)
            if resp.status_code == 503:
                raise ErnieError("pipeline_not_loaded", 503)
            resp.raise_for_status()
            elapsed = time.perf_counter() - t0
            logger.info("ERNIE generate: %dx%d in %.1fs (%d bytes)",
                        width, height, elapsed, len(resp.content))
            return resp.content
        except httpx.TimeoutException:
            raise ErnieError(f"sidecar_timeout: generate took >{self._generate_timeout}s", 504)
        except ErnieError:
            raise
        except Exception as exc:
            raise ErnieError(f"sidecar_error: {exc}", 502)


# Module singleton
ernie = ErnieClient()
