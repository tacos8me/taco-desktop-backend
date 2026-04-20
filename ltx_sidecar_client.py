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
        generate_timeout: float = 600.0,
        mgmt_timeout: float = 60.0,
        auth_token: str | None = None,
        label: str = "local",
    ):
        self._base_url = base_url if base_url is not None else config.LTX_SIDECAR_URL
        self._generate_timeout = generate_timeout
        self._mgmt_timeout = mgmt_timeout
        self._auth_token = auth_token or None
        self.label = label  # used in logs to distinguish pool members

    # --- internal helpers ---

    def _headers(self) -> dict:
        """Authorization header if token configured, else empty dict."""
        return {"Authorization": f"Bearer {self._auth_token}"} if self._auth_token else {}

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
                resp = await client.request(method, url, headers=self._headers())
        except httpx.TimeoutException as exc:
            logger.info("LTX sidecar[%s] %s %s TIMEOUT in %.2fs", self.label, method, path, time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_timeout: {exc}", 504) from exc
        except httpx.ConnectError as exc:
            logger.info("LTX sidecar[%s] %s %s UNREACHABLE in %.2fs", self.label, method, path, time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_unreachable: {exc}", 503) from exc
        except httpx.HTTPError as exc:
            logger.info("LTX sidecar[%s] %s %s HTTP error in %.2fs", self.label, method, path, time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info("LTX sidecar[%s] %s %s → %d in %.2fs", self.label, method, path, resp.status_code, dt)
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
        # v1.6.1: base64-encoded media bytes for remote sidecars that can't
        # see the caller's filesystem (Modal). Caller SHOULD pass either
        # *_path (local sidecar) OR *_b64 (remote sidecar), not both.
        audio_b64: str | None = None,
        image_b64: str | None = None,
        video_b64: str | None = None,
        # v1.7.0: IC-LoRA outpaint extras
        position: str | None = None,
        conditioning_strength: float | None = None,
        skip_stage_2: bool | None = None,
        # v1.12: chain-segment MP4 for multi-frame latent conditioning.
        segment_path: str | None = None,
        segment_b64: str | None = None,
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
        # v1.6.1 media inline
        if audio_b64 is not None:
            payload["audio_b64"] = audio_b64
        if image_b64 is not None:
            payload["image_b64"] = image_b64
        if video_b64 is not None:
            payload["video_b64"] = video_b64
        # v1.7.0 outpaint extras
        if position is not None:
            payload["position"] = position
        if conditioning_strength is not None:
            payload["conditioning_strength"] = conditioning_strength
        if skip_stage_2 is not None:
            payload["skip_stage_2"] = skip_stage_2
        # v1.12 chain-segment conditioning
        if segment_path is not None:
            payload["segment_path"] = segment_path
        if segment_b64 is not None:
            payload["segment_b64"] = segment_b64

        url = f"{self._base_url}/generate"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._generate_timeout) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            logger.info("LTX sidecar[%s] POST /generate TIMEOUT in %.2fs", self.label, time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_timeout: {exc}", 504) from exc
        except httpx.ConnectError as exc:
            logger.info("LTX sidecar[%s] POST /generate UNREACHABLE in %.2fs", self.label, time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_unreachable: {exc}", 503) from exc
        except httpx.HTTPError as exc:
            logger.info("LTX sidecar[%s] POST /generate HTTP error in %.2fs", self.label, time.perf_counter() - t0)
            raise LtxSidecarError(f"sidecar_http_error: {exc}", 502) from exc

        dt = time.perf_counter() - t0
        logger.info(
            "LTX sidecar[%s] POST /generate → %d in %.2fs (%d bytes)",
            self.label,
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


# Primary sidecar — local cuda:1 systemd ltx-sidecar (port 8093).
# Consumers: `from ltx_sidecar_client import ltx_sidecar, LtxSidecarError`.
ltx_sidecar = LtxSidecarClient(label="local")


# Remote sidecar pool — v1.9.0 multi-provider. Dict keyed by provider name
# ("modal", "runpod"). Providers with empty URL are omitted. Each worker task
# in _scale_remote_pool is bound to one provider, so dispatch is provider-tagged
# by task closure (no per-job routing logic needed).
ltx_remote_sidecars: dict[str, LtxSidecarClient] = {}

if config.LTX_MODAL_SIDECAR_URL:
    ltx_remote_sidecars["modal"] = LtxSidecarClient(
        config.LTX_MODAL_SIDECAR_URL,
        auth_token=config.LTX_MODAL_SIDECAR_TOKEN,
        label="modal",
        # Modal cold-start on scaledown_window expiry can take ~60-90s.
        mgmt_timeout=120.0,
    )

if config.LTX_RUNPOD_SIDECAR_URL:
    ltx_remote_sidecars["runpod"] = LtxSidecarClient(
        config.LTX_RUNPOD_SIDECAR_URL,
        auth_token=config.LTX_RUNPOD_SIDECAR_TOKEN,
        label="runpod",
        # RunPod Load-Balancing Serverless with min_workers=1 keeps one warm;
        # 60s management timeout is enough for the warm path. Cold start on a
        # second concurrent request can take 90-120s but goes through /generate
        # which already has the 600s generate_timeout.
        mgmt_timeout=60.0,
    )

# Legacy alias — v1.6-v1.8 code imports `ltx_remote_sidecar` (singular). Point
# it at the modal entry (the only provider before v1.9) so nothing breaks.
ltx_remote_sidecar: LtxSidecarClient | None = ltx_remote_sidecars.get("modal")
