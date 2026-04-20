"""FastAPI server implementing the LTX-compatible API for taco-desktop."""

from __future__ import annotations

import asyncio
import copy as _copy
import json as _json_mod
import subprocess
import time
import torch
import logging
import random
import secrets as _secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from datetime import datetime, timezone
from email.utils import formatdate, parsedate_to_datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

import config
import split_model_manager
from split_model_manager import SplitModelManager
from flux_manager import FluxManager, FluxLoraError
from joyai_client import joyai, JoyAIError
from ernie_client import ernie, ErnieError
from ace_client import ace, AceError
from ltx_sidecar_client import ltx_sidecar, ltx_remote_sidecar, ltx_remote_sidecars, LtxSidecarError
from chat_manager import ChatManager
from helpers import _duration_to_frames, _resolution_to_dims
from upload_store import UploadStore
from lora_registry import LoRARegistry
from flux_lora_registry import FluxLoRARegistry
from job_queue import (
    Job, JobStatus, JobType, JobStore, make_job_id, make_flux_callback,
    worker_loop, cleanup_loop,
    BatchStatus, BatchItemResult, BatchJob, BatchStore, make_batch_id,
    _MEDIA_TYPES as _JOB_MEDIA_TYPES,
)
from history_store import HistoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

manager = SplitModelManager()
flux = FluxManager()
uploads = UploadStore(config.UPLOAD_DIR)
lora_registry = LoRARegistry(config.LORAS_DIR)
flux_lora_registry = FluxLoRARegistry(config.FLUX_LORAS_DIR)
chat = ChatManager()
history = HistoryStore()

# ---------------------------------------------------------------------------
# Flux generation config (persisted to .flux_config.json)
# ---------------------------------------------------------------------------

_FLUX_CONFIG_PATH = Path(__file__).parent / ".flux_config.json"
_DEFAULT_FLUX_CONFIG = {
    "default_model": "flux2-dev",
    "t2i_steps": 50,
    "edit_steps": 4,
    "guidance_scale": 4.0,
    "turbo": False,
    "turbo_steps": 8,
    "turbo_guidance": 2.5,
}


def _load_flux_config() -> dict:
    if _FLUX_CONFIG_PATH.exists():
        try:
            saved = _json_mod.loads(_FLUX_CONFIG_PATH.read_text())
            merged = _copy.deepcopy(_DEFAULT_FLUX_CONFIG)
            for k in merged:
                if k in saved:
                    merged[k] = saved[k]
            return merged
        except Exception:
            pass
    return _copy.deepcopy(_DEFAULT_FLUX_CONFIG)


def _save_flux_config() -> None:
    try:
        _FLUX_CONFIG_PATH.write_text(_json_mod.dumps(_flux_config, indent=2))
    except Exception:
        pass


_flux_config = _load_flux_config()

# Shared inference lock: FP8 layerwise casting in diffusers causes CUBLAS_STATUS_INTERNAL_ERROR
# when Flux and LTX run CUDA inference concurrently in the same process.
_inference_lock = asyncio.Lock()
_paused = False

# Job queue
job_store = JobStore()
_job_queue: asyncio.Queue[str] = asyncio.Queue()

# Batch queue
batch_store = BatchStore()
_batch_queue: asyncio.Queue[str] = asyncio.Queue()

# Turbo mode — dual-GPU LTX inference (2 concurrent video jobs)
_VIDEO_JOB_TYPES = {JobType.TEXT_TO_VIDEO, JobType.IMAGE_TO_VIDEO, JobType.AUDIO_TO_VIDEO, JobType.RETAKE, JobType.VIDEO_OUTPAINT}
_turbo_active: bool = False
_turbo_worker_task: asyncio.Task | None = None         # local sidecar worker (cuda:1)

# v1.6 / v1.9.0 multi-provider: remote-sidecar pool. User-controlled target
# count per provider persists across turbo toggles. `_remote_worker_tasks[p]`
# is the CURRENT live workers for provider p (scales to match target while
# turbo is active; scales to 0 when turbo is off).
_PROVIDERS: tuple[str, ...] = ("modal", "runpod")
_remote_worker_tasks: dict[str, list[asyncio.Task]] = {p: [] for p in _PROVIDERS}
_remote_worker_targets: dict[str, int] = {
    "modal": 1 if config.LTX_MODAL_SIDECAR_URL else 0,
    "runpod": 0,  # opt-in — leave at 0, operator scales up via dashboard
}
_remote_pool_lock = asyncio.Lock()  # serialize concurrent scale requests


def _provider_max(provider: str) -> int:
    """Upper bound of remote workers for a given provider."""
    return {
        "modal": config.LTX_MODAL_MAX_WORKERS,
        "runpod": config.LTX_RUNPOD_MAX_WORKERS,
    }.get(provider, 0)


def _total_remote_target() -> int:
    """Sum of per-provider targets — the v1.6 flat `_remote_worker_target`."""
    return sum(_remote_worker_targets.values())


def _total_remote_active() -> int:
    """Sum of per-provider live worker counts."""
    return sum(len(v) for v in _remote_worker_tasks.values())

# cuda:1 activity tracker for auto-turbo. Updated on every successful
# JoyAI edit or ACE music completion. Initialized to now so the 30-min
# idle timer doesn't fire immediately at startup.
_last_cuda1_activity: float = time.monotonic()


def _cuda1_idle_seconds() -> float:
    """Seconds since last JoyAI/ACE activity on cuda:1."""
    return time.monotonic() - _last_cuda1_activity


def _touch_cuda1_activity() -> None:
    """Mark cuda:1 as recently active (resets the auto-turbo idle timer)."""
    global _last_cuda1_activity
    _last_cuda1_activity = time.monotonic()


async def _auto_exit_turbo_if_active(reason: str) -> None:
    """If turbo is active, gracefully exit to reclaim cuda:1 for JoyAI/ACE.

    Called by JoyAI-edit and music handlers instead of returning 503.
    Blocks for ~15s while turbo exits, then the caller proceeds normally.
    """
    if _turbo_active:
        logger.info("Auto-turbo exit: %s request reclaiming cuda:1 (idle timer will re-engage later)", reason)
        async with _inference_lock:
            await _exit_turbo_mode()
        _touch_cuda1_activity()  # request arrival = activity

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


# Two-tenant mutual exclusion on cuda:0 (v1.2): LTX, Flux. JoyAI and ACE are
# on cuda:1 sidecars — no swap needed. `_last_gpu_tenant` tracks the current
# holder; `_evict_other_tenants(new)` idempotently flips from any state to `new`.
# Must be called while holding `_inference_lock`.
_last_gpu_tenant: str | None = None  # "ltx" | "flux" | None


async def _evict_other_tenants(new: str) -> None:
    """Evict any tenant that isn't `new`. Caller MUST hold _inference_lock."""
    global _last_gpu_tenant
    if _last_gpu_tenant == "ltx" and new != "ltx":
        logger.info("Auto-swap: evicting LTX from %s for %s", config.LTX_DEVICE, new)
        manager.evict_all()
    if _last_gpu_tenant == "flux" and new != "flux":
        logger.info("Auto-swap: unloading Flux from %s for %s", config.FLUX_DEVICE, new)
        flux.unload()
    _last_gpu_tenant = new


async def _ensure_ltx_resident() -> None:
    """Ensure LTX is loaded on cuda:0. Caller must hold _inference_lock.

    Aggressively unloads Flux even if the tracker already says "ltx" —
    because Flux's pipeline (loaded at startup with enable_model_cpu_offload)
    holds ~0.5-1 GB of CUDA context + offload hook allocations. On peak
    LTX workloads (i2v + audio gen = 85-90 GB), that extra GB causes OOM.
    """
    await _evict_other_tenants("ltx")
    # Kill any lingering Flux pipeline — its CUDA context wastes VRAM
    # that LTX's peak inference needs. Flux will lazy-reload on next image req.
    if flux.is_ready:
        logger.info("Auto-swap: destroying Flux pipeline to free CUDA context for LTX peak")
        flux.unload()
    if not manager.is_ready:
        # If the previous load_all() raised midway, GPU has partial state and
        # a blind retry will OOM again. reset() nulls workers + encoder_ledger
        # + flushes allocator so we start clean.
        if getattr(manager, "_last_load_failed", False):
            logger.warning("Auto-swap: prior LTX load failed — resetting before retry")
            manager.reset()
        logger.info("Auto-swap: loading LTX on %s", config.LTX_DEVICE)
        manager.load_all()


async def _ensure_flux_ready() -> None:
    """Ensure Flux is ready (pipeline exists) on cuda:0. Caller must hold _inference_lock."""
    if config.DUAL_GPU_LTX:
        raise RuntimeError("Flux disabled in dual-GPU LTX mode")
    await _evict_other_tenants("flux")
    if not flux.is_ready:
        logger.info("Auto-swap: loading Flux on %s", config.FLUX_DEVICE)
        flux.load()


async def _ensure_ernie_ready() -> None:
    """Load ERNIE-Image on cuda:1, evicting JoyAI if needed."""
    if not config.LOAD_ERNIE:
        raise ErnieError("ernie_disabled: set LOAD_ERNIE=1 to enable", 503)
    await _auto_exit_turbo_if_active("ernie-image")
    try:
        await joyai.unload()
    except Exception:
        pass
    await ernie.load()


async def _run_music_job(job: Job) -> None:
    """Standalone async task for music on cuda:1. No _inference_lock needed."""
    job.status = JobStatus.PROCESSING
    job.started_at = time.monotonic()
    job.phase = "generating"
    result_bytes: bytes | None = None
    staged_tmp: list[str] = []
    try:
        # Free JoyAI/ERNIE from cuda:1 if loaded — gives ACE more headroom.
        # joyai.load()/ernie.load() before the next request will reload on demand.
        try:
            await joyai.unload()
        except Exception:
            pass
        try:
            await ernie.unload()
        except Exception:
            pass

        p = job.params
        # ACE's validate_audio_path rejects absolute paths outside the system
        # tempdir. Our uploads dir lives on the data volume, not /tmp, so copy
        # any source/reference audio to a tempfile for the duration of the job.
        import shutil, tempfile, os
        for key in ("source_audio_path", "reference_audio_path"):
            src = p.get(key)
            if not src:
                continue
            suffix = os.path.splitext(src)[1] or ".bin"
            fd, staged = tempfile.mkstemp(prefix=f"ace-{key}-{job.id}-", suffix=suffix)
            os.close(fd)
            shutil.copyfile(src, staged)
            p[key] = staged
            staged_tmp.append(staged)

        ace_params = _build_ace_params(p)
        est = _estimate_music_time(p)

        def on_progress(elapsed: float) -> None:
            if job.status == JobStatus.CANCELLED:
                return
            job.progress = min(elapsed / max(est, 1), 0.90)

        result_bytes = await ace.generate(params=ace_params, on_progress=on_progress)
        job.phase = "saving"
        job.progress = 0.99
        upload_id, storage_uri = uploads.create()
        uploads.save(upload_id, result_bytes)
        job.result_uri = storage_uri
        job.result_media_type = _AUDIO_MEDIA_TYPES.get(p.get("audio_format", "mp3"), "audio/mpeg")
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.phase = None
        elapsed = time.monotonic() - (job.started_at or job.created_at)
        _touch_cuda1_activity()
        logger.info("Music job %s completed in %.1fs", job.id, elapsed)
    except AceError as exc:
        job.status = JobStatus.FAILED
        job.phase = None
        job.error = str(exc)[:500]
        job.error_code = "ace_error"
        logger.exception("Music job %s failed", job.id)
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.phase = None
        job.error = str(exc)[:500]
        job.error_code = "generation_failed"
        logger.exception("Music job %s failed", job.id)
    finally:
        # Remove staged /tmp copies of source/reference audio.
        for _p in staged_tmp:
            try:
                import os as _os
                _os.remove(_p)
            except OSError:
                pass
        job.completed_at = time.monotonic()
        # v1.8.2 / SEC P1-3: release music quota slot.
        _decr_key_count(_per_key_music_counts, job.api_key or "")
        if history and job.api_key and result_bytes is not None:
            _params = job.params or {}
            _captured = dict(
                job_id=job.id, api_key=job.api_key, job_type=job.type,
                prompt=_params.get("prompt", ""), model=None,
                width=0, height=0, turbo=False,
                status=job.status, result_uri=job.result_uri,
                result_bytes=result_bytes, created_at=time.time(),
                completed_at=time.time(), error=job.error,
                seed=_params.get("seed"),
                enhanced_prompt=None,
                raw_request=None,
                gen_config_snapshot=None,
                dispatch_params=_params,
            )

            async def _save():
                try:
                    await asyncio.to_thread(history.save, **_captured)
                except Exception:
                    logger.warning("Failed to save music job %s to history", job.id, exc_info=True)

            asyncio.create_task(_save())


async def _dispatch_job(job: Job) -> bytes:
    """Route a job to the correct manager and return result bytes."""
    if _paused:
        raise RuntimeError("System is paused for maintenance")
    p = job.params

    def on_progress(progress: float, phase: str | None = None) -> None:
        # Guard against stale callbacks firing after a DELETE /v2/jobs/{id}
        # cancel — the denoiser runs in an executor and can tick one or two
        # more times before it observes the cancelled flag.
        if job.status == JobStatus.CANCELLED:
            return
        job.progress = progress
        if phase is not None:
            job.phase = phase

    def _capture_enhanced(text: str) -> None:
        job.enhanced_prompt = text

    def _is_cancelled() -> bool:
        # Polled from inside the denoiser loop between steps. Returning True
        # raises GenerationCancelledError and unwinds the sigma loop so we
        # stop burning GPU on a job the user DELETEd.
        return job.status == JobStatus.CANCELLED

    match job.type:
        case JobType.TEXT_TO_VIDEO:
            await _ensure_ltx_resident()
            job.gen_config_snapshot = dict(split_model_manager._gen_config)
            return await manager.generate_text_to_video(
                **p, on_progress=on_progress, on_prompt_enhanced=_capture_enhanced,
                on_cancel_check=_is_cancelled,
            )
        case JobType.IMAGE_TO_VIDEO:
            await _ensure_ltx_resident()
            job.gen_config_snapshot = dict(split_model_manager._gen_config)
            return await manager.generate_image_to_video(
                **p, on_progress=on_progress, on_prompt_enhanced=_capture_enhanced,
                on_cancel_check=_is_cancelled,
            )
        case JobType.AUDIO_TO_VIDEO:
            await _ensure_ltx_resident()
            job.gen_config_snapshot = dict(split_model_manager._gen_config)
            return await manager.generate_audio_to_video(
                **p, on_progress=on_progress, on_prompt_enhanced=_capture_enhanced,
                on_cancel_check=_is_cancelled,
            )
        case JobType.RETAKE:
            await _ensure_ltx_resident()
            job.gen_config_snapshot = dict(split_model_manager._gen_config)
            return await manager.retake(
                **p, on_progress=on_progress, on_prompt_enhanced=_capture_enhanced,
                on_cancel_check=_is_cancelled,
            )
        case JobType.VIDEO_OUTPAINT:
            await _ensure_ltx_resident()
            job.gen_config_snapshot = dict(split_model_manager._gen_config)
            # Drop history-only fields that aren't generate_outpaint kwargs.
            op_params = {k: v for k, v in p.items() if k not in ("width", "height", "model")}
            return await manager.generate_outpaint(
                **op_params, on_progress=on_progress, on_prompt_enhanced=_capture_enhanced,
                on_cancel_check=_is_cancelled,
            )
        case JobType.TEXT_TO_IMAGE:
            model = p.get("model", "flux2-klein")
            if model == "ernie-image":
                # ERNIE-Image: no enhanced-prompt pipeline, no gen_config snapshot.
                await _ensure_ernie_ready()
                on_progress(0.90, phase="encoding")
                result = await ernie.generate(
                    prompt=p["prompt"],
                    width=p.get("width", 1024),
                    height=p.get("height", 1024),
                    num_inference_steps=p.get("num_inference_steps", 50),
                    guidance_scale=p.get("guidance_scale", 4.0),
                    seed=p.get("seed"),
                )
                _touch_cuda1_activity()
                return result
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            if p.get("turbo"):
                job.gen_config_snapshot = {
                    "turbo_steps": _flux_config["turbo_steps"],
                    "turbo_guidance": _flux_config["turbo_guidance"],
                }
            cb = make_flux_callback(job, p.get("num_inference_steps", 50))
            return await flux.generate_text_to_image(**p, callback_on_step_end=cb, phase_sink=on_progress, turbo_steps=_flux_config["turbo_steps"], turbo_guidance=_flux_config["turbo_guidance"])
        case JobType.IMAGE_TO_IMAGE:
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            if p.get("turbo"):
                job.gen_config_snapshot = {
                    "turbo_steps": _flux_config["turbo_steps"],
                    "turbo_guidance": _flux_config["turbo_guidance"],
                }
            cb = make_flux_callback(job, p.get("num_inference_steps", 50))
            return await flux.generate_image_to_image(**p, callback_on_step_end=cb, phase_sink=on_progress, turbo_steps=_flux_config["turbo_steps"], turbo_guidance=_flux_config["turbo_guidance"])
        case JobType.IMAGE_EDIT:
            model = p.get("model", "flux2-klein")
            if model == "joyai-edit":
                # JoyAI is on cuda:1 sidecar -- no GPU swap needed
                await _auto_exit_turbo_if_active("joyai-edit")
                if not config.LOAD_JOYAI:
                    raise JoyAIError("joyai_disabled: set LOAD_JOYAI=1 to enable", 503)
                if ernie.is_loaded:
                    logger.info("Evicting ERNIE-Image from cuda:1 for JoyAI")
                    await ernie.unload()
                await joyai.load()  # idempotent — ensures sidecar pipeline is loaded on cuda:1
                # joyai-edit: exactly one image_path, ignores lora, chat-template
                # prompt wrap is server-side in the sidecar.
                image_paths = p.get("image_paths") or []
                if len(image_paths) != 1:
                    raise ValueError(
                        f"joyai-edit requires exactly one image_uri, got {len(image_paths)}"
                    )
                # Phase callback: JoyAI is a single opaque HTTP call so we just
                # set one phase transition. Client UIs will show "encoding" for
                # the whole duration.
                on_progress(0.90, phase="encoding")
                result = await joyai.edit(
                    prompt=p["prompt"],
                    image_path=image_paths[0],
                    width=p["width"],
                    height=p["height"],
                    num_inference_steps=p.get("num_inference_steps", 30),
                    guidance_scale=p.get("guidance_scale", 4.0),
                    seed=p.get("seed"),
                )
                _touch_cuda1_activity()
                return result
            else:
                await _ensure_flux_ready()
                torch.cuda.set_device(config.FLUX_DEVICE)
                if p.get("turbo"):
                    job.gen_config_snapshot = {
                        "turbo_steps": _flux_config["turbo_steps"],
                        "turbo_guidance": _flux_config["turbo_guidance"],
                    }
                cb = make_flux_callback(job, p.get("num_inference_steps", 4))
                return await flux.generate_image_edit(**p, callback_on_step_end=cb, phase_sink=on_progress)
        case JobType.EXPORT_COMPOSITION:
            from export_handler import export_composition
            audio_uri = p.get("audio_uri")
            return await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: export_composition(p["clips"], p["transitions"], uploads, audio_uri=audio_uri),
            )
        case _:
            raise ValueError(f"Unknown job type: {job.type}")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _last_gpu_tenant
    logger.info("Loading LTX pipelines on %s ...", config.GPU_DEVICES)
    manager.load_all()
    logger.info("LTX pipelines ready.")
    _last_gpu_tenant = "ltx"

    if config.LOAD_FLUX:
        # Don't load Flux eagerly — its pipeline + CUDA context wastes ~0.5-1 GB
        # of GPU that LTX's peak inference needs. _ensure_flux_ready() will
        # lazy-load it on the first image request. The page-cache warmup task
        # below pre-reads Flux's safetensors files for fast cold load.
        logger.info("Flux enabled (lazy load on first image request)")
        # Flux uses enable_model_cpu_offload, so its weights live on pinned CPU
        # at idle. GPU tenancy stays "ltx" until an actual Flux forward pass.
    else:
        logger.info("Flux loading disabled (LOAD_FLUX not set)")

    chat.load()
    logger.info("Chat proxy ready.")

    if config.LOAD_JOYAI:
        try:
            health = await joyai.health()
            logger.info("JoyAI sidecar reachable: %s", health)
        except Exception as exc:
            logger.warning(
                "JoyAI sidecar unreachable at %s — joyai-edit will return 503: %s",
                config.JOYAI_SIDECAR_URL,
                exc,
            )

    if config.LOAD_ACE:
        for attempt in range(3):
            try:
                ace_health = await ace.health()
                logger.info("ACE sidecar reachable: %s", ace_health)
                break
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    logger.warning("ACE sidecar unreachable at %s: %s", config.ACE_SIDECAR_URL, exc)

    try:
        ltx_health = await ltx_sidecar.health()
        logger.info("LTX sidecar reachable: %s", ltx_health)
    except Exception as exc:
        logger.warning(
            "LTX sidecar unreachable at %s — turbo will use sidecar when available: %s",
            config.LTX_SIDECAR_URL,
            exc,
        )

    # Warm the OS page cache for checkpoint files not loaded at startup.
    # The dev checkpoint is already cached (just loaded for the encoder hub +
    # denoiser). The distilled checkpoint, LoRA, and upsamplers are cold until
    # the first fast/hq/pro request — pre-reading them eliminates the 7-30s
    # cold→warm variance on the first model swap.
    async def _warmup_page_cache() -> None:
        import subprocess
        warmup_files = [
            config.DISTILLED_CHECKPOINT,  # 43 GB — used by fast model
            config.DISTILLED_LORA,        # 7 GB — used by pro/hq stage 2
            config.SPATIAL_UPSAMPLER,     # 1 GB — used by all video gens
        ]
        for path in warmup_files:
            if Path(path).exists():
                t0 = time.monotonic()
                await asyncio.to_thread(
                    subprocess.run,
                    ["dd", f"if={path}", "of=/dev/null", "bs=1M"],
                    capture_output=True, timeout=120,
                )
                logger.info("Page cache warm: %s (%.1fs)", Path(path).name, time.monotonic() - t0)
    asyncio.create_task(_warmup_page_cache(), name="page-cache-warmup")

    worker_task = asyncio.create_task(
        worker_loop(job_store, _job_queue, _inference_lock, _dispatch_job, uploads, history,
                    turbo_check=lambda job: _turbo_active and job.type in _VIDEO_JOB_TYPES,
                    on_complete=_decr_queue_on_complete),
        name="queue-worker",
    )
    cleanup_task = asyncio.create_task(
        cleanup_loop(job_store, uploads),
        name="queue-cleanup",
    )
    batch_worker_task = asyncio.create_task(
        batch_worker(),
        name="batch-worker",
    )
    batch_cleanup_task = asyncio.create_task(
        batch_cleanup_loop(),
        name="batch-cleanup",
    )
    # Dual-GPU LTX: use LTX sidecar on cuda:1 for second concurrent worker.
    # Must be a separate process — concurrent CUDA ops on different GPUs in
    # the same process cause illegal memory access.
    dual_worker_task = None
    if config.DUAL_GPU_LTX:
        global _turbo_active
        try:
            await ltx_sidecar.load()
            _turbo_active = True  # bypass _inference_lock, route 2nd worker to sidecar
            dual_worker_task = asyncio.create_task(
                worker_loop(job_store, _job_queue, _inference_lock, _dispatch_job_turbo,
                            uploads, history, turbo_check=lambda job: True,
                            on_complete=_decr_queue_on_complete),
                name="queue-worker-gpu1",
            )
            logger.info("Dual-GPU LTX: sidecar on cuda:1, 2 concurrent video workers active")
        except Exception:
            logger.warning("Dual-GPU LTX: sidecar load failed — running single-GPU", exc_info=True)

    logger.info("Job queue + batch worker started.")

    # v1.8.2 / SEC P0-2: surface admin-auth posture at boot so operators
    # notice degraded mode before a real incident hits.
    if not config.API_KEYS:
        logger.warning("Auth is globally disabled (.api_keys empty) — admin gate is a no-op")
    elif not config.ADMIN_KEYS:
        logger.warning(
            "admin auth disabled: .admin_keys is empty. All API_KEYS entries are treated "
            "as admin as a backwards-compat bridge. Create .admin_keys to lock the 12 "
            "mutation endpoints down to a narrower operator bearer set."
        )
    else:
        logger.info("Admin gate active: %d admin key(s) loaded", len(config.ADMIN_KEYS))

    yield

    worker_task.cancel()
    cleanup_task.cancel()
    batch_worker_task.cancel()
    batch_cleanup_task.cancel()
    if dual_worker_task:
        dual_worker_task.cancel()


app = FastAPI(lifespan=lifespan)

# v1.9.7: gzip JSON + text responses above 1 KB. History list is ~26 KB JSON
# that compresses to ~5 KB (5× reduction). Already-compressed media types
# (image/*, video/*, audio/*) are passed through unchanged by the middleware
# — it respects the response's `Content-Encoding` header and mime type.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|192\.168\.\d+\.\d+)(:\d+)?$",
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# v1.9.7: bounded concurrency for the lazy PyAV preview-extract fallback in
# GET /v2/jobs/{id}/preview. 100+ MB reads + PyAV first-frame decode run
# inside the event-loop's default thread pool; without a cap a burst of
# preview polls on malformed videos could starve every other thread-pool
# task (history.save, sidecar subprocesses, etc.).
_PREVIEW_EXTRACT_SEMAPHORE = asyncio.Semaphore(2)

# v1.10.0 unit B: bounded concurrency for POST /v2/video/extract-frames.
# Up to 16 frames × PyAV decode runs on a thread; capped at 2 concurrent to
# keep the thread pool available for preview extraction and history saves.
# Separate instance from _PREVIEW_EXTRACT_SEMAPHORE so long extracts can't
# starve the preview-polling path.
_FRAME_EXTRACT_SEMAPHORE = asyncio.Semaphore(2)


# v1.9.7: approved-images manifest cache. The manifest.json file was parsed
# fresh on every GET /v1/approved-images + every SSE poll — small file but
# hit every few seconds per tab. Cache keyed by (mtime_ns, size) so any
# out-of-band write (or `write_text` from our own handlers) naturally
# invalidates. Returns the parsed + type-guarded list.
_approved_manifest_cache: tuple[tuple[int, int], list] | None = None


def _load_approved_manifest() -> list:
    """Return the parsed approved-images manifest (cached, mtime-validated).

    Empty list when the file doesn't exist or isn't a list after parsing
    (SEC P2-10 type guard). Zero locking — GIL protects the tuple assign,
    and a brief double-read under contention just parses twice (idempotent).
    """
    global _approved_manifest_cache
    manifest_path = config.APPROVED_IMAGES_DIR / "manifest.json"
    try:
        stat = manifest_path.stat()
    except FileNotFoundError:
        _approved_manifest_cache = None
        return []
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _approved_manifest_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    raw = _json_mod.loads(manifest_path.read_text())
    manifest = raw.get("images", raw) if isinstance(raw, dict) else raw
    if not isinstance(manifest, list):
        logger.warning("approved-images manifest.json was not a list, resetting to empty")
        manifest = []
    _approved_manifest_cache = (key, manifest)
    return manifest


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if not config.API_KEYS:
        return await call_next(request)
    # /health only — the others (dashboard, GPU telemetry) were moved to the
    # LAN-only admin port (see dashboard_server.py). Internet-exposed paths
    # all require Bearer auth now (SEC P1-1 / P1-2, v1.8.1).
    if request.url.path in ("/health", "/v1/approved-images/events"):
        return await call_next(request)
    # SSE job streams: EventSource can't set custom headers, so these endpoints
    # accept a `?token=` query param and do their own auth inside the handler.
    if request.url.path.startswith("/v2/jobs/") and request.url.path.endswith("/stream"):
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""

    # SEC P2-1 (v1.8.2): full-iteration compare instead of any(...) to avoid
    # leaking bearer-set membership via short-circuit timing.
    matched = False
    for key in config.API_KEYS:
        if _secrets.compare_digest(token, key):
            matched = True
    if not token or not matched:
        return _error(401, "Invalid or missing API key")

    return await call_next(request)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


ModelName = Literal["ltx-2-3-fast", "ltx-2-3-pro", "ltx-2-3-hq"]


class LoRAInput(BaseModel):
    id: str = Field(description="LoRA ID from /v1/loras")
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
Resolution = Literal["1920x1080", "1080x1920", "2560x1440", "1440x2560", "3840x2160", "2160x3840"]
RetakeMode = Literal["replace_audio_and_video", "replace_video", "replace_video_only", "replace_audio"]
ImageModelName = Literal["flux2-dev", "flux2-klein", "joyai-edit", "ernie-image"]


class TextToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False
    camera_motion: str | None = Field(default=None, max_length=200)
    lora: LoRAInput | None = None
    enhance_prompt: bool = False


class KeyframeInput(BaseModel):
    image_uri: str
    frame_index: int | Literal["first", "middle", "last"] = Field(default=0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class ImageToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str | None = None
    image_strength: float = Field(default=0.85, ge=0.0, le=1.0)
    keyframes: list[KeyframeInput] | None = None
    model: ModelName
    resolution: Resolution
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    generate_audio: bool = False
    lora: LoRAInput | None = None
    enhance_prompt: bool = False


class AudioToVideoRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    audio_uri: str
    image_uri: str | None = None
    # v1.9.5: was silently stripped by Pydantic (extra="ignore"). Used as the
    # first-keyframe strength when `image_uri` is set. Matches the field on
    # ImageToVideoRequest so clients can tune a2v's intro frame intensity.
    image_strength: float = Field(default=0.85, ge=0.0, le=1.0)
    # v1.10.0: multi-keyframe support for seamless MusicVideo chain mode.
    # Mutually exclusive with image_uri / image_strength in practice — the
    # model_validator below rejects mixed specifications with 422.
    keyframes: list[KeyframeInput] | None = None
    model: ModelName
    resolution: Resolution
    duration: float = Field(default=6.0, gt=0, le=30)
    fps: float = Field(default=24.0, gt=0, le=60)
    lora: LoRAInput | None = None
    enhance_prompt: bool = False

    @model_validator(mode="after")
    def _check_keyframes_exclusive(self) -> "AudioToVideoRequest":
        if self.keyframes is not None:
            if self.image_uri is not None:
                raise ValueError("Cannot specify both image_uri and keyframes")
            # image_strength default is 0.85; reject explicit non-default when
            # keyframes is set (it would be silently ignored otherwise).
            if self.image_strength != 0.85:
                raise ValueError("Cannot specify image_strength together with keyframes")
        return self


class RetakeRequest(BaseModel):
    video_uri: str
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0, le=30)
    mode: RetakeMode
    prompt: str | None = Field(default=None, max_length=10000)
    lora: LoRAInput | None = None


OutpaintPosition = Literal[
    "center", "left", "right", "top", "bottom",
    "top_left", "top_right", "bottom_left", "bottom_right",
]


class VideoOutpaintRequest(BaseModel):
    """IC-LoRA video outpaint: expand source video canvas, LoRA fills black padding.

    Uses `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint` by default (registered as
    LoRA id `ic-lora-outpaint`). Source video is scaled proportionally to
    fit within `target_resolution`, padded with pure black (RGB 0,0,0) at
    `position`; the LoRA was trained to treat black pixels as a fill
    sentinel. Output is silent (no audio).

    Always runs the distilled 2-stage pipeline (stage 1 at half target res
    with IC-LoRA conditioning, stage 2 upsamples + refines). Set
    `skip_stage_2=true` for a faster half-resolution preview.
    """
    video_uri: str
    prompt: str = Field(max_length=10000)
    target_resolution: Resolution
    position: OutpaintPosition = "center"
    duration: float = Field(gt=0, le=30)
    fps: float = Field(gt=0, le=60)
    seed: int = Field(default=0, ge=0)
    enhance_prompt: bool = False
    lora: LoRAInput | None = None
    conditioning_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    skip_stage_2: bool = False


class ExtractFramesRequest(BaseModel):
    """v1.10.0 unit B — body for POST /v2/video/extract-frames.

    Server-side PyAV helper that pulls specific frames out of a stored MP4
    and re-saves them as lossless PNG uploads suitable for feeding as
    keyframes of the next chain clip (see v1.10.0 multi-frame chain
    conditioning design).
    """
    video_uri: str = Field(pattern=r"^storage://[0-9a-f]{32}$")
    frame_indices: list[int] = Field(min_length=1, max_length=16)

    @field_validator("frame_indices")
    @classmethod
    def _validate_indices(cls, v: list[int]) -> list[int]:
        if any(i < 0 for i in v):
            raise ValueError("frame_indices must be non-negative")
        return sorted(set(v))


class TextToImageRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    model: ImageModelName = "flux2-dev"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None
    turbo: bool = False
    lora: LoRAInput | None = None


class ImageToImageRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uri: str
    model: ImageModelName = "flux2-dev"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None
    turbo: bool = False
    lora: LoRAInput | None = None


class ImageEditRequest(BaseModel):
    prompt: str = Field(max_length=10000)
    image_uris: list[str] = Field(min_length=1, max_length=10)
    model: ImageModelName = "flux2-klein"
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_inference_steps: int = Field(default=4, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    seed: int | None = None
    lora: LoRAInput | None = None
    # v1.8.0 — identity preservation (Klein-only, see docs/API.md#preserve-identity)
    preserve_identity: bool = False
    identity_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    identity_mode: Literal["balanced", "faithful", "loose"] = "balanced"


class ChatMessage(BaseModel):
    role: str
    content: str | list  # str for text, list for multimodal


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3-12b-nvfp4"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)


class CharRankRequest(BaseModel):
    rank_image_uri: str
    generated_image_uri: str
    prompt: str = Field(max_length=10000)


class CharRankAnalysis(BaseModel):
    face_match: int = Field(ge=1, le=10)
    eyes: int = Field(ge=1, le=10)
    proportions: int = Field(ge=1, le=10)
    overall_likeness: int = Field(ge=1, le=10)


class CharRankEdits(BaseModel):
    add: list[str] = Field(default_factory=list, max_length=20)
    remove: list[str] = Field(default_factory=list, max_length=20)
    modify: dict[str, str] = Field(default_factory=dict)


class CharRankResponse(BaseModel):
    """SEC P1-5 (v1.8.2): schema the vision model's JSON output is
    validated against before the server echoes it to the client. Rejects
    prompt-injected fields, type confusion, out-of-range scores, and
    malformed edits."""
    score: float = Field(ge=0, le=10)
    analysis: CharRankAnalysis
    edits: CharRankEdits


class MusicGenerationRequest(BaseModel):
    # Core
    prompt: str = Field(max_length=10000, description="Music description / caption")
    lyrics: str = Field(default="[Instrumental]", max_length=50000)
    duration: float = Field(default=60.0, gt=0, le=600)
    audio_format: Literal["mp3", "flac", "wav", "wav32", "opus", "aac"] = "mp3"
    seed: int | None = None
    # Music theory
    bpm: int | None = Field(default=None, ge=30, le=300)
    key_scale: str | None = None
    time_signature: str | None = None
    vocal_language: str | None = None
    # Diffusion (xl-base defaults)
    num_inference_steps: int = Field(default=50, ge=1, le=200)
    guidance_scale: float = Field(default=7.0, ge=0, le=15)
    shift: float = Field(default=3.0, ge=1.0, le=5.0)
    infer_method: Literal["ode", "sde"] = "ode"
    use_adg: bool = False
    cfg_interval_start: float = Field(default=0.0, ge=0, le=1)
    cfg_interval_end: float = Field(default=1.0, ge=0, le=1)
    batch_size: int = Field(default=1, ge=1, le=8)
    # Task type
    task_type: Literal["text2music", "cover", "repaint", "extract", "lego", "complete"] = "text2music"
    source_audio_uri: str | None = None
    reference_audio_uri: str | None = None
    audio_cover_strength: float = Field(default=1.0, ge=0, le=1)
    repainting_start: float = Field(default=0.0, ge=0)
    repainting_end: float | None = None
    repaint_mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
    repaint_strength: float = Field(default=0.5, ge=0, le=1)
    track_name: str | None = None
    # LM / thinking
    thinking: bool = False
    sample_mode: bool = False
    sample_query: str | None = None
    lm_temperature: float = Field(default=0.85, ge=0, le=2)
    lm_top_p: float = Field(default=0.9, ge=0, le=1)


_AUDIO_MEDIA_TYPES = {
    "mp3": "audio/mpeg", "flac": "audio/flac", "wav": "audio/wav",
    "wav32": "audio/wav", "opus": "audio/opus", "aac": "audio/aac",
}


# ---------------------------------------------------------------------------
# Batch request models
# ---------------------------------------------------------------------------

_BATCH_ITEM_TYPES = Literal[
    "text-to-image", "image-to-image", "image-edit",
    "text-to-video", "image-to-video",
]


class BatchItem(BaseModel):
    """One generation request inside a batch."""
    type: _BATCH_ITEM_TYPES
    params: dict[str, Any]


class BatchRequest(BaseModel):
    """Submit a batch of generation jobs."""
    items: list[BatchItem] = Field(..., min_length=1, max_length=50)
    priority: Literal["normal", "high"] = "normal"
    callback_url: str | None = None


# Map batch item type string -> (JobType, Pydantic validator model)
_BATCH_TYPE_MAP: dict[str, tuple[JobType, type[BaseModel]]] = {
    "text-to-image": (JobType.TEXT_TO_IMAGE, TextToImageRequest),
    "image-to-image": (JobType.IMAGE_TO_IMAGE, ImageToImageRequest),
    "image-edit": (JobType.IMAGE_EDIT, ImageEditRequest),
    "text-to-video": (JobType.TEXT_TO_VIDEO, TextToVideoRequest),
    "image-to-video": (JobType.IMAGE_TO_VIDEO, ImageToVideoRequest),
}


def _is_image_type(type_str: str) -> bool:
    """Return True for image generation types."""
    return type_str in ("text-to-image", "image-to-image", "image-edit")


def _batch_item_to_job(item: BatchItem, api_key: str) -> Job:
    """Create a synthetic Job from a BatchItem for dispatch.

    Pre-processes params the same way the v2 endpoint handlers do:
    resolution → width/height, duration → num_frames, camera_motion → prompt,
    lora {id,strength} → lora_path + lora_strength, storage:// URIs → paths.
    Without this, _dispatch_job passes raw pydantic fields (resolution, duration)
    to the manager which expects pre-resolved values (width, height, num_frames).
    """
    job_type, _ = _BATCH_TYPE_MAP[item.type]
    p = dict(item.params)

    # LoRA resolution: convert {id, strength} → lora_path + lora_strength
    if "lora" in p and p["lora"] is not None:
        lora_input = p.pop("lora")
        lora_id = lora_input["id"] if isinstance(lora_input, dict) else lora_input.id
        lora_str = lora_input.get("strength", 1.0) if isinstance(lora_input, dict) else lora_input.strength
        if _is_image_type(item.type):
            info = flux_lora_registry.get(lora_id)
            if info is None:
                raise ValueError(f"Flux LoRA not found: {lora_id}")
            p["lora_path"] = str(flux_lora_registry.resolve_path(lora_id))
        else:
            info = lora_registry.get(lora_id)
            if info is None:
                raise ValueError(f"LoRA not found: {lora_id}")
            p["lora_path"] = str(lora_registry.resolve_path(lora_id))
        p["lora_strength"] = lora_str
    else:
        p.pop("lora", None)

    # Video items: resolve resolution + duration → width/height/num_frames
    if item.type in ("text-to-video", "image-to-video", "audio-to-video"):
        if "resolution" in p:
            w, h = _resolution_to_dims(p.pop("resolution"))
            p["width"] = w
            p["height"] = h
        if "duration" in p and "num_frames" not in p:
            fps = p.get("fps", 24)
            p["num_frames"] = _duration_to_frames(p.pop("duration"), fps)
        if "camera_motion" in p:
            cm = p.pop("camera_motion")
            if cm:
                p["prompt"] = f"{p.get('prompt', '')} [{cm}]"
        if "seed" not in p or p.get("seed") is None:
            p["seed"] = random.randint(0, 2**32 - 1)
        p.setdefault("generate_audio", False)

    # Image items: snap dims to multiples of 16, resolve storage:// URIs
    if item.type in ("text-to-image", "image-to-image", "image-edit"):
        if "width" in p:
            p["width"] = (p["width"] // 16) * 16
        if "height" in p:
            p["height"] = (p["height"] // 16) * 16
        if "seed" not in p or p["seed"] is None:
            p["seed"] = random.randint(0, 2**32 - 1)

    # Resolve storage:// URIs → filesystem paths (same as v2 handlers)
    if item.type == "image-to-image" and "image_uri" in p:
        p["image_path"] = str(uploads.resolve(p.pop("image_uri")))
    if item.type == "image-edit" and "image_uris" in p:
        p["image_paths"] = [str(uploads.resolve(uri)) for uri in p.pop("image_uris")]
    # i2v: convert image_uri/keyframes to the keyframes list format that
    # generate_image_to_video expects: [{"image_path": str, "frame_index": int, "strength": float}]
    if item.type == "image-to-video":
        if "keyframes" in p and p["keyframes"]:
            # Multi-keyframe: resolve each storage:// URI to filesystem path
            kfs = p.pop("keyframes")
            resolved = []
            for kf in kfs:
                kf_dict = kf if isinstance(kf, dict) else kf.model_dump()
                fi = kf_dict.get("frame_index", 0)
                if fi == "first": fi = 0
                elif fi == "middle": fi = p.get("num_frames", 73) // 2
                elif fi == "last": fi = p.get("num_frames", 73) - 1
                elif isinstance(fi, int) and fi < 0: fi = p.get("num_frames", 73) + fi
                resolved.append({
                    "image_path": str(uploads.resolve(kf_dict["image_uri"])),
                    "frame_index": fi,
                    "strength": kf_dict.get("strength", 1.0),
                })
            p["keyframes"] = resolved
        elif "image_uri" in p and p["image_uri"]:
            # Single start frame: convert to keyframe at frame 0
            image_uri = p.pop("image_uri")
            strength = p.pop("image_strength", 0.85)
            p["keyframes"] = [{
                "image_path": str(uploads.resolve(image_uri)),
                "frame_index": 0,
                "strength": strength,
            }]
        p.pop("image_uri", None)
        p.pop("image_strength", None)

    return Job(
        id=make_job_id(),
        type=job_type,
        params=p,
        api_key=api_key,
    )


def _build_ace_params(p: dict) -> dict:
    """Translate MusicGenerationRequest params to ACE /release_task fields."""
    ace_p = {
        "caption": p["prompt"],
        "lyrics": p.get("lyrics", "[Instrumental]"),
        "audio_duration": p.get("duration", 60.0),
        "audio_format": p.get("audio_format", "mp3"),
        "inference_steps": p.get("num_inference_steps", 50),
        "guidance_scale": p.get("guidance_scale", 7.0),
        "shift": p.get("shift", 3.0),
        "infer_method": p.get("infer_method", "ode"),
        "use_adg": p.get("use_adg", False),
        "cfg_interval_start": p.get("cfg_interval_start", 0.0),
        "cfg_interval_end": p.get("cfg_interval_end", 1.0),
        "batch_size": p.get("batch_size", 1),
        "task_type": p.get("task_type", "text2music"),
        "audio_cover_strength": p.get("audio_cover_strength", 1.0),
        "thinking": p.get("thinking", False),
        "sample_mode": p.get("sample_mode", False),
    }
    # Seed handling: ACE uses seed=-1 for random
    if p.get("seed") is not None:
        ace_p["seed"] = p["seed"]
        ace_p["use_random_seed"] = False
    else:
        ace_p["seed"] = -1
        ace_p["use_random_seed"] = True
    # Optional fields -- only send if provided
    for key in ("bpm", "key_scale", "time_signature", "vocal_language",
                "sample_query", "repainting_start", "repainting_end",
                "repaint_mode", "repaint_strength", "track_name",
                "lm_temperature", "lm_top_p"):
        if p.get(key) is not None:
            ace_p[key] = p[key]
    # Resolved audio paths (already converted from storage:// URIs by the handler)
    if p.get("source_audio_path"):
        ace_p["src_audio_path"] = p["source_audio_path"]
    if p.get("reference_audio_path"):
        ace_p["reference_audio_path"] = p["reference_audio_path"]
    return ace_p


def _estimate_music_time(p: dict) -> float:
    """Rough estimate of music generation time in seconds."""
    steps = p.get("num_inference_steps", 50)
    dur = p.get("duration", 60.0)
    if steps <= 8:
        return max(5, dur * 0.1)
    elif steps <= 32:
        return max(15, dur * 0.3)
    else:
        return max(25, dur * 0.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt(prompt: str, camera_motion: str | None) -> str:
    """Append camera-motion tag to prompt if specified."""
    if camera_motion:
        return f"{prompt} [{camera_motion}]"
    return prompt


def _resolve_keyframes(body, num_frames: int, *, allow_neither: bool = False) -> list[dict] | JSONResponse | None:
    """Resolve keyframes, converting symbolic/negative frame indices to absolute values.

    v1.10.0: reused by both ImageToVideoRequest (requires image_uri or keyframes)
    and AudioToVideoRequest (both are optional — audio is the only required input).
    When ``allow_neither`` is True and the body has neither ``image_uri`` nor
    ``keyframes``, returns ``None`` (caller uses the audio-only path).
    """
    if body.keyframes and body.image_uri:
        return _error(422, "Cannot specify both image_uri and keyframes")
    if body.keyframes:
        if len(body.keyframes) == 0:
            return _error(422, "keyframes list must not be empty")
        if len(body.keyframes) > 8:
            return _error(422, "At most 8 keyframes are allowed")
        keyframe_inputs = []
        for kf in body.keyframes:
            fi = kf.frame_index
            if fi == "first":
                fi = 0
            elif fi == "middle":
                fi = num_frames // 2
            elif fi == "last":
                fi = num_frames - 1
            elif isinstance(fi, int) and fi < 0:
                fi = num_frames + fi
            if isinstance(fi, int) and (fi < 0 or (num_frames > 0 and fi >= num_frames)):
                return _error(422, f"Resolved frame_index {fi} is out of range for {num_frames} frames")
            path = str(uploads.resolve(kf.image_uri))
            keyframe_inputs.append({"image_path": path, "frame_index": fi, "strength": kf.strength})
        frame_indices = [kf["frame_index"] for kf in keyframe_inputs]
        if len(frame_indices) != len(set(frame_indices)):
            return _error(422, "Duplicate frame_index values after resolution")
        return keyframe_inputs
    elif body.image_uri:
        path = str(uploads.resolve(body.image_uri))
        return [{"image_path": path, "frame_index": 0, "strength": body.image_strength}]
    elif allow_neither:
        return None
    else:
        return _error(422, "Either image_uri or keyframes is required")


def _resolve_lora(body) -> tuple[str | None, float] | JSONResponse:
    """Resolve optional LoRA from request. Returns (path, strength) or JSONResponse on error."""
    if not getattr(body, "lora", None):
        return None, 1.0
    info = lora_registry.get(body.lora.id)
    if info is None:
        return _error(404, f"LoRA not found: {body.lora.id}")
    return str(lora_registry.resolve_path(body.lora.id)), body.lora.strength


def _resolve_flux_lora(body) -> tuple[str | None, float] | JSONResponse:
    """Resolve optional Flux LoRA from request. Returns (path, strength) or JSONResponse on error."""
    if not getattr(body, "lora", None):
        return None, 1.0
    info = flux_lora_registry.get(body.lora.id)
    if info is None:
        return _error(404, f"Flux LoRA not found: {body.lora.id}")
    return str(flux_lora_registry.resolve_path(body.lora.id)), body.lora.strength


def _error(status: int, msg: str) -> JSONResponse:
    # Avoid leaking internal filesystem paths in error responses
    text = msg[:500]
    if "/mnt/" in text or "/home/" in text or "/tmp/" in text:
        text = "Internal server error"
    return JSONResponse(status_code=status, content={"error": text, "message": text, "detail": text})


def _serve_with_http_cache(
    path: "Path", media_type: str, request: Request, max_age: int, *, immutable: bool = False,
) -> Response:
    """Serve a static file with HTTP caching headers + conditional-GET (304).

    v1.9.7. FastAPI ``FileResponse`` sets ``ETag`` + ``Last-Modified`` from file
    mtime/size but does NOT honor ``If-None-Match`` or ``If-Modified-Since``
    — it always returns 200 with the full body. This wrapper adds:
      - ``Cache-Control: public, max-age=N[, immutable]``
      - Manual conditional-GET check using Starlette's ETag formula so
        client-cached ETags from a prior FileResponse match our 304 path
      - On miss, falls back to ``FileResponse`` with sendfile (zero-copy)

    Call sites: ``/v2/history/{id}/thumbnail``, ``/v2/history/{id}/image``.
    """
    import hashlib as _hashlib
    stat = path.stat()
    # Mirror starlette.responses.FileResponse.set_stat_headers exactly so
    # an ETag cached from a prior FileResponse request matches on 304:
    #   md5(f"{st_mtime}-{st_size}").hexdigest()
    _etag_base = f"{stat.st_mtime}-{stat.st_size}"
    etag = '"' + _hashlib.md5(_etag_base.encode(), usedforsecurity=False).hexdigest() + '"'
    last_mod = formatdate(stat.st_mtime, usegmt=True)
    cc = f"public, max-age={max_age}" + (", immutable" if immutable else "")

    inm = request.headers.get("if-none-match", "")
    not_modified = False
    if inm:
        # Handle comma-separated ETag lists + weak-prefix `W/`.
        for tag in inm.split(","):
            t = tag.strip().removeprefix("W/")
            if t == etag:
                not_modified = True
                break

    if not not_modified:
        ims = request.headers.get("if-modified-since")
        if ims:
            try:
                client_time = parsedate_to_datetime(ims)
                file_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0)
                if file_time <= client_time:
                    not_modified = True
            except (ValueError, TypeError):
                pass  # malformed If-Modified-Since → fall through to full response

    if not_modified:
        return Response(status_code=304, headers={
            "ETag": etag,
            "Cache-Control": cc,
            "Last-Modified": last_mod,
        })

    return FileResponse(
        path=str(path), media_type=media_type,
        headers={"Cache-Control": cc},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# v1.8.1 / SEC P1-1+P1-2: dashboard moved to the LAN-only admin server on
# port 8099 (dashboard_server.py). Keeping a stub route that 404s avoids
# misleading crawlers / cached links into thinking they hit something real.
@app.get("/dashboard", include_in_schema=False)
async def dashboard_moved():
    return _error(404, "Not found")


_gpu_cache: dict | None = None
_gpu_cache_time: float = 0.0


async def _query_gpu_info() -> list[dict]:
    """Run nvidia-smi and parse GPU info."""
    import subprocess
    result = await asyncio.to_thread(
        subprocess.run,
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, timeout=5,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "temperature_c": int(parts[4]) if parts[4] not in ("[N/A]", "[Not Supported]") and parts[4].isdigit() else None,
                "utilization_pct": int(parts[5]) if parts[5] not in ("[N/A]", "[Not Supported]") and parts[5].isdigit() else None,
                "power_draw_w": float(parts[6]) if parts[6] not in ("[N/A]", "[Not Supported]") else None,
            })
        elif len(parts) >= 5:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "temperature_c": int(parts[4]) if parts[4].isdigit() else None,
            })
    return gpus


@app.get("/v1/system/gpu")
async def system_gpu() -> dict:
    """nvidia-smi GPU telemetry for the dashboard (2s cache)."""
    global _gpu_cache, _gpu_cache_time
    now = time.monotonic()
    if _gpu_cache is None or (now - _gpu_cache_time) > 2.0:
        try:
            gpus = await _query_gpu_info()
            from split_model_manager import _gen_config
            _gpu_cache = {
                "gpus": gpus,
                "turbo": _turbo_active,
                "sampler": _gen_config["sampler"],
                "gen_config": dict(_gen_config),
                "flux_config": dict(_flux_config),
                "gpu0_tenant": _last_gpu_tenant or "idle",
                "gpu1_tenant": (
                    "ltx-sidecar" if config.DUAL_GPU_LTX else
                    ("ltx-sidecar" if _turbo_active else
                     ("ace+joyai" if (config.LOAD_ACE or config.LOAD_JOYAI) else "idle"))
                ),
            }
            _gpu_cache_time = now
        except Exception as exc:
            logger.warning("nvidia-smi failed: %s", exc)
            return {"gpus": [], "turbo": _turbo_active, "error": str(exc)}
    return _gpu_cache


@app.get("/health")
async def health() -> dict:
    if _paused:
        return {
            "status": "paused",
            "ltx": "paused",
            "flux": "paused",
            "ace": "paused" if config.LOAD_ACE else "disabled",
            "ernie": "paused" if config.LOAD_ERNIE else "disabled",
            "chat": "ready" if chat.is_ready else "not_loaded",
            "queue": job_store.stats(),
        }
    return {
        "status": "ok",
        "ltx": "ready" if manager.is_ready else "not_loaded",
        "flux": "ready" if flux.is_ready else "not_loaded",
        "ace": "enabled" if config.LOAD_ACE else "disabled",
        "ernie": "ready" if ernie.is_loaded else ("enabled" if config.LOAD_ERNIE else "disabled"),
        "chat": "ready" if chat.is_ready else "not_loaded",
        "queue": job_store.stats(),
    }


@app.post("/v1/system/pause")
async def system_pause(request: Request):
    """Evict all models from GPU to free VRAM for training."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _paused, _last_gpu_tenant
    if _paused:
        return {"status": "already_paused"}
    try:
        async with _inference_lock:
            # Exit turbo mode first if active
            if _turbo_active:
                await _exit_turbo_mode()
            # Cancel queued (not yet processing) jobs
            while not _job_queue.empty():
                try:
                    job_id = _job_queue.get_nowait()
                    job = job_store.get(job_id)
                    if job and job.status == JobStatus.QUEUED:
                        job.status = JobStatus.FAILED
                        job.error = "System paused"
                except asyncio.QueueEmpty:
                    break
            # Cancel queued batches
            while not _batch_queue.empty():
                try:
                    bid = _batch_queue.get_nowait()
                    b = batch_store.get(bid)
                    if b and b.status == BatchStatus.QUEUED:
                        b.status = BatchStatus.CANCELLED
                except asyncio.QueueEmpty:
                    break
            _paused = True
            manager.evict_all()
            flux.unload()
            if config.LOAD_JOYAI:
                try:
                    await joyai.unload()
                except Exception:
                    logger.warning("JoyAI unload during pause failed", exc_info=True)
            if config.LOAD_ERNIE:
                try:
                    await ernie.unload()
                except Exception:
                    logger.warning("ERNIE unload during pause failed", exc_info=True)
            _last_gpu_tenant = None
        logger.info("System paused — all GPU memory freed")
        return {"status": "paused"}
    except Exception:
        logger.exception("Pause failed")
        _paused = True
        return JSONResponse(status_code=500, content={"error": "pause_failed", "status": "paused"})


@app.post("/v1/system/resume")
async def system_resume(request: Request):
    """Reload all models after training."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _paused, _last_gpu_tenant
    if not _paused:
        return {"status": "already_running"}
    try:
        async with _inference_lock:
            manager.load_all()
            _last_gpu_tenant = "ltx"
            if config.LOAD_FLUX:
                flux.load()
            _paused = False
        logger.info("System resumed — all models reloaded")
        return {"status": "ready"}
    except Exception:
        logger.exception("Resume failed — system remains paused")
        return JSONResponse(status_code=500, content={"error": "resume_failed", "status": "paused"})


@app.post("/v1/flux/unload")
async def flux_unload(request: Request):
    """Unload Flux model from the Flux device to free VRAM for training / vision models."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _last_gpu_tenant
    if not flux.is_ready:
        return {"status": "already_unloaded"}
    try:
        async with _inference_lock:
            flux.unload()
            _last_gpu_tenant = None
        logger.info("Flux unloaded from %s", config.FLUX_DEVICE)
        return {"status": "unloaded"}
    except Exception:
        logger.exception("Flux unload failed")
        return JSONResponse(status_code=500, content={"error": "flux_unload_failed"})


@app.post("/v1/ltx/unload")
async def ltx_unload(request: Request):
    """Unload LTX from the LTX device to free VRAM (e.g. for a training run).

    Unlike /v1/system/pause this touches ONLY LTX — the Flux pipeline stays
    available for image generation. In single-GPU swap mode (LTX_DEVICE ==
    FLUX_DEVICE), this also makes room on the GPU for a subsequent Flux
    forward pass; the next video request will auto-swap LTX back in.
    """
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _last_gpu_tenant
    if not manager.is_ready:
        return {"status": "already_unloaded"}
    try:
        async with _inference_lock:
            manager.evict_all()
            _last_gpu_tenant = None
        logger.info("LTX unloaded from %s", config.LTX_DEVICE)
        return {"status": "unloaded"}
    except Exception:
        logger.exception("LTX unload failed")
        return JSONResponse(status_code=500, content={"error": "ltx_unload_failed"})


@app.post("/v1/ltx/reload")
async def ltx_reload(request: Request):
    """Reload LTX to the LTX device."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _last_gpu_tenant
    if manager.is_ready:
        return {"status": "already_loaded"}
    try:
        async with _inference_lock:
            manager.load_all()
            _last_gpu_tenant = "ltx"
        logger.info("LTX reloaded to %s", config.LTX_DEVICE)
        return {"status": "loaded"}
    except Exception:
        logger.exception("LTX reload failed")
        return JSONResponse(status_code=500, content={"error": "ltx_reload_failed"})


@app.post("/v1/flux/reload")
async def flux_reload(request: Request):
    """Reload Flux model to the Flux device."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _last_gpu_tenant
    if flux.is_ready:
        return {"status": "already_loaded"}
    try:
        async with _inference_lock:
            flux.load()
            _last_gpu_tenant = "flux"
        logger.info("Flux reloaded to %s", config.FLUX_DEVICE)
        return {"status": "loaded"}
    except Exception:
        logger.exception("Flux reload failed")
        return JSONResponse(status_code=500, content={"error": "flux_reload_failed"})


# ---------------------------------------------------------------------------
# Turbo Mode (dual-GPU) — Phase 3
# ---------------------------------------------------------------------------


def _get_gpu_free_mb(gpu_index: int) -> int:
    """Quick check of free VRAM on a GPU via torch.cuda."""
    try:
        free, total = torch.cuda.mem_get_info(gpu_index)
        return free // (1024 * 1024)
    except Exception:
        return 0


async def _systemctl_unit(unit: str, action: str) -> None:
    """Run `systemctl --user <action> <unit>` in a thread.

    Raises RuntimeError with the stderr on non-zero exit. Distinguishes
    from `_ace_systemctl`'s earlier implementation by reporting failure
    explicitly — callers who want "best effort, don't raise" should wrap.
    """
    result = await asyncio.to_thread(
        subprocess.run,
        ["systemctl", "--user", action, unit],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"systemctl {action} {unit}: exit={result.returncode} stderr={result.stderr.strip()!r}"
        )


async def _ace_systemctl(action: str) -> None:
    """Back-compat alias for pre-v1.5 callers — delegates to _systemctl_unit."""
    try:
        await _systemctl_unit("ace-step", action)
    except RuntimeError as exc:
        # Preserve earlier silent-on-error semantics
        logger.warning("ace_systemctl %s: %s", action, exc)


# ---------------------------------------------------------------------------
# cuda:1 drain verification — v1.5 turbo entry hardening
# ---------------------------------------------------------------------------


async def _cuda1_memory_used_mib() -> int:
    """Return cuda:1's used VRAM in MiB via nvidia-smi. Returns -1 on failure."""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        lines = [line.strip() for line in result.stdout.strip().splitlines()]
        if len(lines) >= 2:
            return int(lines[1])
        return -1
    except Exception:
        logger.exception("nvidia-smi memory query failed")
        return -1


async def _list_cuda1_processes() -> list[dict]:
    """Enumerate processes with memory allocations on cuda:1.

    Uses bus-id matching — cuda:1 on this host is PCI bus E1:00.0 (verified
    via `nvidia-smi -L` which also lists GPU UUIDs). Fallback: any compute
    app whose bus_id is NOT `01:00.0` is assumed to be cuda:1 (2-GPU box).
    """
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        procs = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            pid_str, name, mem_str, bus_id = parts[:4]
            # Heuristic: cuda:0 is bus 01:00.0, anything else = cuda:1
            if "01:00.0" in bus_id:
                continue
            try:
                procs.append({"pid": int(pid_str), "name": name, "mem_mib": int(mem_str), "bus": bus_id})
            except ValueError:
                continue
        return procs
    except Exception:
        logger.exception("nvidia-smi compute-apps query failed")
        return []


async def _wait_cuda1_free(threshold_mib: int = 2000, timeout_s: float = 20.0) -> bool:
    """Poll cuda:1 memory until below `threshold_mib`. Returns True if clear,
    False if timeout expired with cuda:1 still crowded. Polls every 1s.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used = await _cuda1_memory_used_mib()
        if 0 <= used < threshold_mib:
            return True
        logger.info("Turbo: cuda:1 still has %d MiB used, waiting…", used)
        await asyncio.sleep(1)
    return False


async def _stop_cuda1_tenants() -> None:
    """Stop all cuda:1 systemd tenants (ACE, JoyAI, ERNIE, ltx-sidecar).

    Always best-effort — if a unit isn't running, systemctl returns 0 or a
    harmless non-zero; we log and continue. The subsequent _wait_cuda1_free
    check is the actual correctness gate.
    """
    units = [
        ("ace-step",              config.LOAD_ACE),
        ("joyai-sidecar",         config.LOAD_JOYAI),
        ("ernie-image-sidecar",   config.LOAD_ERNIE),
        ("ltx-sidecar",           True),  # always stop — may be stale from prior turbo
    ]
    for unit, _ in units:
        try:
            await _systemctl_unit(unit, "stop")
            logger.info("Turbo: systemctl stop %s", unit)
        except RuntimeError as exc:
            # "not running" is fine; only log
            logger.info("Turbo: systemctl stop %s — %s", unit, exc)


async def _restore_cuda1_tenants() -> None:
    """Restart the cuda:1 systemd tenants (ACE/JoyAI/ERNIE) that were
    configured LOAD_*=1. Inverse of `_stop_cuda1_tenants` (minus ltx-sidecar,
    which is only started during turbo).
    """
    for unit, cfg_flag in [
        ("ace-step",            config.LOAD_ACE),
        ("joyai-sidecar",       config.LOAD_JOYAI),
        ("ernie-image-sidecar", config.LOAD_ERNIE),
    ]:
        if not cfg_flag:
            continue
        try:
            await _systemctl_unit(unit, "start")
            logger.info("Turbo exit: systemctl start %s", unit)
        except RuntimeError as exc:
            logger.warning("Turbo exit: systemctl start %s failed: %s", unit, exc)


async def _auto_exit_turbo_on_sidecar_failure() -> None:
    """Exit turbo mode after a sidecar transport failure. Runs in a background
    task so we don't block the dispatching worker while holding the inference
    lock. Best-effort; logs but doesn't raise.
    """
    try:
        async with _inference_lock:
            await _exit_turbo_mode()
        logger.warning("Auto-exited turbo mode after sidecar transport failure")
    except Exception:
        logger.exception("Auto-exit turbo after sidecar failure: failed")


async def _dispatch_job_turbo_remote(job: Job, *, provider: str = "modal") -> bytes:
    """Route a video job to a REMOTE LTX sidecar for the given provider.

    v1.5: single Modal sidecar. v1.6: 0..MAX_WORKERS pool. v1.9.0: multi-provider
    (``provider`` selects the LtxSidecarClient from ``ltx_remote_sidecars``).
    Each worker task in ``_scale_remote_pool`` binds a specific provider via
    functools.partial, so dispatch is task-tagged with no per-job routing.

    Unlike the local sidecar path, remote transport failures do NOT auto-exit
    turbo — remotes are optional extra capacity. If one is broken, its workers
    fail their jobs; main + local-sidecar + other-provider workers keep serving.

    v1.6.1: media files (audio for a2v, image for i2v keyframes, video for
    retake) get inlined as base64 in the request body — remote sidecars can't
    see our uploads/ directory.
    """
    if job.type not in _VIDEO_JOB_TYPES:
        raise ValueError(f"Remote turbo worker cannot handle {job.type} — only video jobs supported")
    client = ltx_remote_sidecars.get(provider)
    if client is None:
        raise RuntimeError(f"remote_sidecar_not_configured: {provider}")
    p = job.params

    import base64

    def _read_b64(path: str | None) -> str | None:
        if not path:
            return None
        try:
            return base64.b64encode(Path(path).read_bytes()).decode("ascii")
        except FileNotFoundError:
            raise ValueError(f"remote_dispatch: media file not found: {path}")

    audio_b64 = _read_b64(p.get("audio_path"))
    image_b64 = _read_b64(p.get("image_path"))
    video_b64 = _read_b64(p.get("video_path"))

    # Keyframes: list of {image_path, frame_index, strength}. Inline each image.
    keyframes = p.get("keyframes")
    remote_keyframes = None
    if keyframes:
        remote_keyframes = []
        for kf in keyframes:
            new_kf = dict(kf)
            if new_kf.get("image_path"):
                new_kf["image_b64"] = _read_b64(new_kf.pop("image_path"))
            remote_keyframes.append(new_kf)

    # v1.7.0 + v1.9.0: for outpaint, each remote provider has the IC-LoRA
    # pre-staged on its own network volume at a provider-specific mount.
    # Rewrite the local LORAS_DIR prefix to the provider's mount so fused-
    # transformer cache keys match across requests on that provider (avoids
    # per-request LoRA file re-staging).
    remote_lora_path = p.get("lora_path")
    if job.type == JobType.VIDEO_OUTPAINT and remote_lora_path:
        local_loras_dir = str(config.LORAS_DIR).rstrip("/") + "/"
        if remote_lora_path.startswith(local_loras_dir):
            provider_mount = config.LTX_PROVIDER_LORAS_MOUNT.get(provider)
            if provider_mount:
                remote_lora_path = remote_lora_path.replace(local_loras_dir, provider_mount)

    return await client.generate(
        job_type=job.type,
        prompt=p["prompt"], model=p.get("model", "ltx-2-3-fast"),
        width=p["width"], height=p["height"],
        num_frames=p["num_frames"], fps=p.get("fps", 24),
        seed=p["seed"], generate_audio=p.get("generate_audio", False),
        lora_path=remote_lora_path, lora_strength=p.get("lora_strength", 1.0),
        enhance_prompt=p.get("enhance_prompt", False),
        keyframes=remote_keyframes,
        # v1.6.1: drop local paths, send base64 bytes instead
        audio_path=None, audio_b64=audio_b64,
        image_path=None, image_b64=image_b64,
        video_path=None, video_b64=video_b64,
        start_time=p.get("start_time"),
        duration=p.get("duration"),
        mode=p.get("mode"),
        # v1.7.0 outpaint extras — None for non-outpaint job types
        position=p.get("position"),
        conditioning_strength=p.get("conditioning_strength"),
        skip_stage_2=p.get("skip_stage_2"),
    )


# ---------------------------------------------------------------------------
# Remote-sidecar pool (v1.6)
# ---------------------------------------------------------------------------


async def _scale_remote_pool() -> None:
    """Reconcile ``_remote_worker_tasks[provider]`` to the per-provider target.

    Desired per-provider count is ``_remote_worker_targets[p]`` IF turbo is
    active (remote workers are turbo-scoped — outside turbo, Flux/ACE/JoyAI/
    ERNIE submit heterogeneous job types and only the main worker can safely
    dispatch them). Otherwise desired = 0 for every provider.

    v1.9.0: iterates every configured provider in ``ltx_remote_sidecars``.
    Each worker task is bound to its provider via functools.partial so the
    task pulls from the shared queue but dispatches to its own URL.

    Safe to call concurrently — guarded by ``_remote_pool_lock``.
    """
    from functools import partial

    if not ltx_remote_sidecars:
        return

    async with _remote_pool_lock:
        for provider, client in ltx_remote_sidecars.items():
            max_workers = _provider_max(provider)
            target = _remote_worker_targets.get(provider, 0) if _turbo_active else 0
            desired = max(0, min(target, max_workers))
            tasks = _remote_worker_tasks.setdefault(provider, [])
            current = len(tasks)

            if desired > current:
                # Warming once keeps cold-start out of the first job's latency.
                if current == 0:
                    try:
                        await asyncio.wait_for(client.health(), timeout=90.0)
                    except Exception as exc:
                        logger.warning(
                            "Remote pool[%s]: health probe failed (will try anyway): %s",
                            provider, exc,
                        )
                for i in range(current, desired):
                    task = asyncio.create_task(
                        worker_loop(
                            job_store, _job_queue, _inference_lock,
                            partial(_dispatch_job_turbo_remote, provider=provider),
                            uploads, history,
                            # Always skip inference_lock — remote workers never
                            # touch local GPUs. This is safe because remote
                            # workers are video-only (accept_check below) and
                            # the remote dispatcher raises on non-video types.
                            turbo_check=lambda job: True,
                            # v1.9.6: re-queue non-video jobs for the main
                            # worker (export-composition, char-rank, etc.).
                            accept_check=lambda job: job.type in _VIDEO_JOB_TYPES,
                            on_complete=_decr_queue_on_complete,
                        ),
                        name=f"remote-worker-{provider}-{i}",
                    )
                    tasks.append(task)
                logger.info("Remote pool[%s]: scaled %d -> %d workers", provider, current, desired)

            elif desired < current:
                while len(tasks) > desired:
                    task = tasks.pop()
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                logger.info("Remote pool[%s]: scaled %d -> %d workers", provider, current, desired)
                # NOTE: deliberately not calling client.unload() here.
                # On Modal our /unload path calls SplitModelManager.evict_all(),
                # which clears the worker list. A future /generate against a
                # still-warm container then fails with "no LTX workers available"
                # because @modal.enter only runs once per container lifetime.
                # Modal's scaledown_window (5 min) / RunPod's idle_timeout
                # reclaim the GPU naturally when the pool stays at 0; that's
                # the authoritative path for credit-saving.


async def _dispatch_job_turbo(job: Job) -> bytes:
    """Route a video job to the LTX sidecar on cuda:1.

    If the sidecar is unreachable / times out / returns a transport-level
    error (not a job-level failure), we schedule an automatic turbo exit so
    the next queued job doesn't immediately fail the same way.
    """
    if job.type not in _VIDEO_JOB_TYPES:
        raise ValueError(f"Turbo worker cannot handle {job.type} — only video jobs supported")
    p = job.params
    try:
        return await ltx_sidecar.generate(
            job_type=job.type,
            prompt=p["prompt"], model=p.get("model", "ltx-2-3-fast"),
            width=p["width"], height=p["height"],
            num_frames=p["num_frames"], fps=p.get("fps", 24),
            seed=p["seed"], generate_audio=p.get("generate_audio", False),
            lora_path=p.get("lora_path"), lora_strength=p.get("lora_strength", 1.0),
            enhance_prompt=p.get("enhance_prompt", False),
            keyframes=p.get("keyframes"),
            audio_path=p.get("audio_path"),
            image_path=p.get("image_path"),
            video_path=p.get("video_path"),
            start_time=p.get("start_time"),
            duration=p.get("duration"),
            mode=p.get("mode"),
            # v1.7.0 outpaint extras — None for non-outpaint job types
            position=p.get("position"),
            conditioning_strength=p.get("conditioning_strength"),
            skip_stage_2=p.get("skip_stage_2"),
        )
    except LtxSidecarError as exc:
        # Transport-level failures (status 502/503/504) mean the sidecar is
        # dead or unreachable. Schedule a turbo exit so the next job doesn't
        # just fail the same way — systemd will restart the sidecar and the
        # main worker resumes on cuda:0. Job-level errors (4xx, validation)
        # don't trigger exit.
        status = getattr(exc, "status", None) or getattr(exc, "args", [None, None])[1]
        if status in (502, 503, 504) or any(
            tag in str(exc).lower() for tag in ("unreachable", "timeout", "http_error")
        ):
            logger.error("Turbo dispatch: sidecar transport failure (%s) — scheduling auto-exit", exc)
            # SEC P2-8 (v1.8.2): dedup — no-op if an exit is already scheduled.
            _schedule_auto_exit_turbo()
        raise


async def _enter_turbo_mode() -> None:
    """Enable turbo: claim cuda:1 for LTX, start 2–3 concurrent workers.

    v1.5 hardening — this used to trust HTTP /unload calls on JoyAI/ERNIE,
    which could succeed on the wire while leaving GPU memory resident
    (client reports "unloaded" but the Python process still holds tensors).
    That caused a classic OOM on ltx-sidecar /load: cuda:1 already 97%
    full, allocator hits the ceiling. We now use `systemctl stop` as the
    hammer for every cuda:1 tenant + poll `nvidia-smi` to verify the GPU
    is actually free before attempting the ltx-sidecar load. If it isn't,
    we abort with a detailed error instead of hitting OOM.

    Flux, ACE, and JoyAI are unavailable during turbo.
    Caller must hold _inference_lock.
    """
    global _turbo_active, _turbo_worker_task, _last_gpu_tenant

    if _turbo_active:
        return  # idempotent

    # Step 1: Evict Flux from cuda:0 first (different GPU — independent of cuda:1)
    flux.unload()

    # Step 2: Stop ALL cuda:1 systemd tenants (ACE + JoyAI + ERNIE + any
    # stale ltx-sidecar). systemctl stop is the authority; we no longer
    # trust the HTTP /unload path.
    await _stop_cuda1_tenants()

    # Step 3: Verify cuda:1 actually dropped to <2 GB. If a process is
    # lingering (systemd unit reports stopped but PID kept around, or a
    # non-managed orphan from a prior crash), abort with a clear error so
    # the operator can clean up manually rather than auto-kill (destructive).
    if not await _wait_cuda1_free(threshold_mib=2000, timeout_s=20.0):
        offenders = await _list_cuda1_processes()
        await _restore_cuda1_tenants()  # restart services we stopped
        raise RuntimeError(
            "turbo_entry_aborted: cuda:1 still crowded after systemctl-stop. "
            f"Offenders: {offenders}. Investigate with `nvidia-smi`, SIGKILL "
            "stale PIDs, then retry."
        )
    logger.info("Turbo: cuda:1 drained, starting ltx-sidecar")

    # Step 4: Start the LTX sidecar service (we stopped it in Step 2 to
    # clear stale state), wait for it to bind, then /load the pipeline.
    try:
        await _systemctl_unit("ltx-sidecar", "start")
    except RuntimeError as exc:
        await _restore_cuda1_tenants()
        raise RuntimeError(f"turbo_entry_failed: ltx-sidecar start: {exc}")
    # Poll /health until the FastAPI server is up (usually <10s).
    for _ in range(30):
        try:
            await ltx_sidecar.health()
            break
        except LtxSidecarError:
            await asyncio.sleep(1)
    else:
        await _restore_cuda1_tenants()
        raise RuntimeError("turbo_entry_failed: ltx-sidecar /health never came up")
    await ltx_sidecar.load()
    _last_gpu_tenant = "ltx"

    # Step 5: Start the local-sidecar turbo worker. Both workers pull from
    # _job_queue and skip _inference_lock because the sidecar runs on a
    # separate GPU.
    _turbo_active = True
    _turbo_worker_task = asyncio.create_task(
        worker_loop(job_store, _job_queue, _inference_lock, _dispatch_job_turbo, uploads, history,
                    turbo_check=lambda job: _turbo_active,
                    # v1.9.6: video-only — export-composition, char-rank, etc.
                    # get re-queued for the main worker to pick up.
                    accept_check=lambda job: job.type in _VIDEO_JOB_TYPES,
                    on_complete=_decr_queue_on_complete),
        name="turbo-worker",
    )

    # Step 6 (v1.6): bring up the remote-sidecar pool to match the user's
    # current target (0..MAX). See `_scale_remote_pool` + the pool control
    # endpoints. Scales to 0 automatically in `_exit_turbo_mode`.
    await _scale_remote_pool()
    remote_active = _total_remote_active()
    breakdown = ", ".join(
        f"{p}={len(t)}" for p, t in _remote_worker_tasks.items() if t
    ) or "none"
    logger.info(
        "TURBO MODE ON: %d concurrent video workers (2 local + %d remote [%s])",
        2 + remote_active, remote_active, breakdown,
    )


async def _exit_turbo_mode() -> None:
    """Unload the LTX sidecar pipeline, restore cuda:1 for ACE+JoyAI.

    Caller must hold _inference_lock.
    """
    global _turbo_active, _turbo_worker_task, _last_gpu_tenant

    if not _turbo_active:
        return  # idempotent

    # Flip the flag first so `_scale_remote_pool` treats desired=0.
    _turbo_active = False

    # Step 1a: scale remote pool to zero. Preserves `_remote_worker_targets`
    # so the user's per-provider settings persist across turbo toggles.
    await _scale_remote_pool()

    # Step 1b: cancel the local-sidecar worker loop.
    if _turbo_worker_task is not None:
        _turbo_worker_task.cancel()
        try:
            await _turbo_worker_task
        except asyncio.CancelledError:
            pass
        _turbo_worker_task = None

    # Step 2: Best-effort HTTP /unload for local sidecar (graceful GPU
    # memory free before the systemctl-stop hammer). The remote pool's
    # /unload already fired inside `_scale_remote_pool()` when it went to 0.
    try:
        await ltx_sidecar.unload()
        logger.info("Turbo exit: local sidecar /unload ok")
    except LtxSidecarError as exc:
        logger.warning("Turbo exit: local sidecar /unload failed — %s", exc)

    # Step 3: Stop the local ltx-sidecar systemd unit. Frees its Python
    # process + any lingering cuda:1 allocator blocks.
    try:
        await _systemctl_unit("ltx-sidecar", "stop")
        logger.info("Turbo exit: systemctl stop ltx-sidecar")
    except RuntimeError as exc:
        logger.warning("Turbo exit: ltx-sidecar stop: %s", exc)
    _last_gpu_tenant = "ltx"

    # Step 4: Restore cuda:1 tenants (ACE/JoyAI/ERNIE) via systemctl — symmetric
    # with _enter_turbo_mode's systemctl-stop. Services cold-boot in 5–15s;
    # the first real request to each may pay a load penalty, but correctness
    # is guaranteed.
    await _restore_cuda1_tenants()

    logger.info("TURBO MODE OFF: cuda:1 released, sidecar services restarting")


class TurboRequest(BaseModel):
    enable: bool


@app.post("/v1/system/turbo")
async def system_turbo(body: TurboRequest, request: Request) -> JSONResponse:
    """Toggle turbo mode (claim/release cuda:1 for dual-GPU inference)."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    if _paused:
        return JSONResponse(
            status_code=503,
            content={"error": "system_paused", "message": "Cannot toggle turbo while paused."},
            headers={"Retry-After": "300"},
        )
    if body.enable and _turbo_active:
        return JSONResponse(status_code=409, content={"error": "already_enabled", "turbo": True})
    if not body.enable and not _turbo_active:
        return JSONResponse(status_code=409, content={"error": "already_disabled", "turbo": False})

    try:
        async with _inference_lock:
            if body.enable:
                await _enter_turbo_mode()
            else:
                await _exit_turbo_mode()
        return JSONResponse(content={
            "turbo": _turbo_active,
            "flux_device": config.FLUX_DEVICE,
            "ltx_device": config.LTX_DEVICE,
            "ace_status": "unloaded" if _turbo_active else ("loaded" if config.LOAD_ACE else "disabled"),
            "joyai_status": "unloaded" if _turbo_active else ("loaded" if config.LOAD_JOYAI else "disabled"),
        })
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    except Exception:
        logger.exception("Turbo toggle failed")
        return JSONResponse(status_code=500, content={"error": "turbo_toggle_failed"})


# ---------------------------------------------------------------------------
# Remote-sidecar pool controls (v1.6)
# ---------------------------------------------------------------------------


class PoolCountRequest(BaseModel):
    count: int = Field(ge=0)


def _pool_state_payload() -> dict:
    """Per-provider pool state + legacy flat-field aliases.

    v1.9.0 response shape. ``providers`` is the authoritative per-provider map;
    the flat ``remote_*`` fields are kept as aliases to the modal provider so
    pre-v1.9 dashboards and scripts don't break.
    """
    providers: dict[str, dict] = {}
    for provider in _PROVIDERS:
        client = ltx_remote_sidecars.get(provider)
        providers[provider] = {
            "configured": client is not None,
            "url": (getattr(client, "_base_url", None) if client else None) or None,
            "target": _remote_worker_targets.get(provider, 0),
            "active": len(_remote_worker_tasks.get(provider, [])),
            "max": _provider_max(provider),
        }
    modal = providers.get("modal", {})
    return {
        "turbo_active": _turbo_active,
        "providers": providers,
        # Legacy flat fields — aliased to modal for backwards compat with v1.6-v1.8.
        "remote_sidecar_configured": bool(modal.get("configured")),
        "remote_sidecar_url": modal.get("url"),
        "remote_worker_target": int(modal.get("target", 0)),
        "remote_worker_active": int(modal.get("active", 0)),
        "remote_worker_max": int(modal.get("max", 0)),
    }


@app.get("/v1/system/pool")
async def get_pool_state() -> JSONResponse:
    """Current state of the LTX remote-sidecar pool (v1.9.0 multi-provider).

    Response:
      - ``turbo_active``: pool is turbo-scoped; active counts are 0 when off
      - ``providers``: per-provider dict keyed by name (``modal``, ``runpod``)
        with ``configured``, ``url``, ``target``, ``active``, ``max``
      - Legacy flat ``remote_*`` fields aliased to the modal provider
    """
    return JSONResponse(content=_pool_state_payload())


@app.post("/v1/system/pool/remote-workers")
async def set_pool_remote_workers(request: Request) -> JSONResponse:
    """Set target remote-sidecar worker counts (0..MAX).

    Accepts two body shapes:
      - ``{"count": N}`` — legacy v1.6 shape, scales the ``modal`` provider only
      - ``{"modal": N, "runpod": M}`` — per-provider targets (v1.9.0)

    Takes effect immediately if turbo is active; otherwise targets are stored
    and applied at next turbo-on.
    """
    deny = _require_admin(request)
    if deny is not None:
        return deny
    if not ltx_remote_sidecars:
        return _error(400, "remote_sidecar_not_configured: set LTX_MODAL_SIDECAR_URL or LTX_RUNPOD_SIDECAR_URL in .env")
    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid_json")
    if not isinstance(body, dict):
        return _error(400, "body_must_be_object")

    # Accept either legacy {"count": N} (modal-only) or per-provider keys.
    if "count" in body:
        if "modal" not in ltx_remote_sidecars:
            return _error(400, "modal_provider_not_configured")
        try:
            count = int(body["count"])
        except (TypeError, ValueError):
            return _error(400, "count_must_be_int")
        if count < 0:
            return _error(400, "count_must_be_nonneg")
        _remote_worker_targets["modal"] = max(0, min(count, _provider_max("modal")))
    else:
        unknown = set(body.keys()) - set(_PROVIDERS)
        if unknown:
            return _error(400, f"unknown_provider: {sorted(unknown)}")
        for provider, raw in body.items():
            try:
                n = int(raw)
            except (TypeError, ValueError):
                return _error(400, f"count_must_be_int: {provider}")
            if n < 0:
                return _error(400, f"count_must_be_nonneg: {provider}")
            if provider not in ltx_remote_sidecars:
                if n == 0:
                    continue  # no-op — caller explicitly leaving this provider disabled
                return _error(400, f"provider_not_configured: {provider}")
            _remote_worker_targets[provider] = max(0, min(n, _provider_max(provider)))

    try:
        await _scale_remote_pool()
    except Exception as exc:
        logger.exception("Remote pool scale failed")
        return _error(500, f"pool_scale_failed: {exc}")
    payload = _pool_state_payload()
    payload["applied_now"] = _turbo_active
    return JSONResponse(content=payload)


@app.post("/v1/system/pool/remote-workers/{provider}")
async def set_pool_remote_workers_provider(
    provider: str, body: PoolCountRequest, request: Request
) -> JSONResponse:
    """Set the target worker count for a specific provider (v1.9.0)."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    if provider not in _PROVIDERS:
        return _error(400, f"unknown_provider: {provider}")
    if provider not in ltx_remote_sidecars:
        return _error(400, f"provider_not_configured: {provider}")
    _remote_worker_targets[provider] = max(0, min(body.count, _provider_max(provider)))
    try:
        await _scale_remote_pool()
    except Exception as exc:
        logger.exception("Remote pool scale failed")
        return _error(500, f"pool_scale_failed: {exc}")
    payload = _pool_state_payload()
    payload["applied_now"] = _turbo_active
    return JSONResponse(content=payload)


@app.get("/v1/system/config")
async def get_gen_config() -> JSONResponse:
    """Get current generation configuration."""
    from split_model_manager import _gen_config
    return JSONResponse(content=dict(_gen_config))


@app.post("/v1/system/config")
async def set_gen_config(request: Request) -> JSONResponse:
    """Update generation configuration. Merges body into current config."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _gpu_cache
    import split_model_manager
    body = await request.json()
    for key, value in body.items():
        if key in split_model_manager._gen_config:
            split_model_manager._gen_config[key] = value
    _gpu_cache = None  # invalidate so next poll returns fresh config
    split_model_manager._save_gen_config()
    return JSONResponse(content={"status": "ok", **dict(split_model_manager._gen_config)})


@app.post("/v1/system/config/reset")
async def reset_gen_config(request: Request) -> JSONResponse:
    """Reset generation config to defaults."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _gpu_cache
    import split_model_manager, copy
    split_model_manager._gen_config.update(copy.deepcopy(split_model_manager._DEFAULT_GEN_CONFIG))
    _gpu_cache = None  # invalidate so next poll returns fresh config
    split_model_manager._save_gen_config()
    return JSONResponse(content={"status": "reset", **dict(split_model_manager._gen_config)})


@app.get("/v1/system/flux-config")
async def get_flux_config() -> JSONResponse:
    """Get current Flux generation configuration."""
    return JSONResponse(content=dict(_flux_config))


@app.post("/v1/system/flux-config")
async def set_flux_config(request: Request) -> JSONResponse:
    """Update Flux generation configuration. Merges body into current config."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _gpu_cache
    body = await request.json()
    for key, value in body.items():
        if key in _flux_config:
            _flux_config[key] = value
    _gpu_cache = None
    _save_flux_config()
    return JSONResponse(content={"status": "ok", **dict(_flux_config)})


@app.post("/v1/system/flux-config/reset")
async def reset_flux_config(request: Request) -> JSONResponse:
    """Reset Flux generation config to defaults."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    global _gpu_cache
    _flux_config.update(_copy.deepcopy(_DEFAULT_FLUX_CONFIG))
    _gpu_cache = None
    _save_flux_config()
    return JSONResponse(content={"status": "reset", **dict(_flux_config)})


@app.get("/v1/system/sampler")
async def get_sampler_config() -> JSONResponse:
    """Get current sampler configuration (alias for config endpoint)."""
    from split_model_manager import _gen_config
    return JSONResponse(content={
        "sampler": _gen_config["sampler"],
        "eta_stage1": _gen_config["eta_stage1"],
        "eta_default": _gen_config["eta_default"],
        "stage2_sigmas": _gen_config["stage2_sigmas"],
    })


@app.post("/v1/system/sampler")
async def set_sampler_config(request: Request) -> JSONResponse:
    """Toggle sampler between Euler and CFG++ (alias — writes to gen_config)."""
    deny = _require_admin(request)
    if deny is not None:
        return deny
    import split_model_manager
    body = await request.json()
    sampler = body.get("sampler", "euler")
    split_model_manager._gen_config["sampler"] = sampler
    if "eta_stage1" in body:
        split_model_manager._gen_config["eta_stage1"] = float(body["eta_stage1"])
    if "eta_default" in body:
        split_model_manager._gen_config["eta_default"] = float(body["eta_default"])
    if "stage2_sigmas" in body:
        split_model_manager._gen_config["stage2_sigmas"] = body["stage2_sigmas"]
    elif sampler == "cfg_pp":
        split_model_manager._gen_config["stage2_sigmas"] = [0.85, 0.725, 0.4219, 0.0]
    return JSONResponse(content={"status": "ok", "sampler": sampler})


@app.post("/v1/text-to-video")
async def text_to_video(body: TextToVideoRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    # Auto-swap handles manager.is_ready lazily inside the lock
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    try:
        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        prompt = _build_prompt(body.prompt, body.camera_motion)
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            video_bytes = await manager.generate_text_to_video(
                prompt=prompt,
                model=body.model,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=body.fps,
                seed=seed,
                generate_audio=body.generate_audio,
                lora_path=lora_path,
                lora_strength=lora_strength,
                enhance_prompt=body.enhance_prompt,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except Exception as exc:
        logger.exception("text-to-video failed")
        return _error(500, str(exc))


@app.post("/v1/image-to-video")
async def image_to_video(body: ImageToVideoRequest) -> Response:
    num_frames = _duration_to_frames(body.duration, body.fps)
    keyframe_inputs = _resolve_keyframes(body, num_frames)
    if isinstance(keyframe_inputs, JSONResponse):
        return keyframe_inputs
    if _paused:
        return _error(503, "System is paused for maintenance")
    # Auto-swap handles manager.is_ready lazily inside the lock
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    try:
        width, height = _resolution_to_dims(body.resolution)
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            video_bytes = await manager.generate_image_to_video(
                prompt=body.prompt,
                keyframes=keyframe_inputs,
                model=body.model,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=body.fps,
                seed=seed,
                generate_audio=body.generate_audio,
                lora_path=lora_path,
                lora_strength=lora_strength,
                enhance_prompt=body.enhance_prompt,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("image-to-video failed")
        return _error(500, str(exc))


@app.post("/v1/audio-to-video")
async def audio_to_video(body: AudioToVideoRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    # Auto-swap handles manager.is_ready lazily inside the lock
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    try:
        audio_path = str(uploads.resolve(body.audio_uri))
        width, height = _resolution_to_dims(body.resolution)
        num_frames = _duration_to_frames(body.duration, body.fps)
        keyframe_inputs = _resolve_keyframes(body, num_frames, allow_neither=True)
        if isinstance(keyframe_inputs, JSONResponse):
            return keyframe_inputs
        # v1.10.0: image_uri path is now folded into keyframe_inputs by
        # _resolve_keyframes (single-keyframe list). _run_a2v receives only
        # `keyframes` from here.
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            video_bytes = await manager.generate_audio_to_video(
                prompt=body.prompt,
                audio_path=audio_path,
                image_path=None,
                keyframes=keyframe_inputs,
                model=body.model,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=body.fps,
                seed=seed,
                lora_path=lora_path,
                lora_strength=lora_strength,
                enhance_prompt=body.enhance_prompt,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("audio-to-video failed")
        return _error(500, str(exc))


@app.post("/v1/retake")
async def retake(body: RetakeRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    # Auto-swap handles manager.is_ready lazily inside the lock
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    try:
        video_path = str(uploads.resolve(body.video_uri))
        prompt = body.prompt or ""
        seed = random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_ltx_resident()
            torch.cuda.set_device(config.LTX_DEVICE)
            video_bytes = await manager.retake(
                video_path=video_path,
                start_time=body.start_time,
                duration=body.duration,
                mode=body.mode,
                prompt=prompt,
                seed=seed,
                lora_path=lora_path,
                lora_strength=lora_strength,
            )
        return Response(content=video_bytes, media_type="video/mp4")
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("retake failed")
        return _error(422, f"Content rejected or generation failed: {exc}")


@app.post("/v1/text-to-image")
async def text_to_image(body: TextToImageRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    if body.model == "ernie-image":
        try:
            await _ensure_ernie_ready()
            width = (body.width // 16) * 16
            height = (body.height // 16) * 16
            seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
            image_bytes = await ernie.generate(
                prompt=body.prompt,
                width=width,
                height=height,
                num_inference_steps=body.num_inference_steps,
                guidance_scale=body.guidance_scale,
                seed=seed,
            )
            return Response(content=image_bytes, media_type="image/webp")
        except ErnieError as exc:
            return _error(exc.status_code, str(exc))
        except Exception as exc:
            logger.exception("text-to-image (ernie) failed")
            return _error(500, str(exc))
    if not config.LOAD_FLUX:
        return _error(500, "Flux not enabled")
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    try:
        width = (body.width // 16) * 16
        height = (body.height // 16) * 16
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            image_bytes = await flux.generate_text_to_image(
                prompt=body.prompt,
                width=width,
                height=height,
                num_inference_steps=body.num_inference_steps,
                guidance_scale=body.guidance_scale,
                seed=seed,
                model=body.model,
                turbo=body.turbo,
                lora_path=lora_path,
                lora_strength=lora_strength,
                turbo_steps=_flux_config["turbo_steps"],
                turbo_guidance=_flux_config["turbo_guidance"],
            )
        return Response(content=image_bytes, media_type="image/webp")
    except FluxLoraError as exc:
        return _error(422, str(exc))
    except Exception as exc:
        logger.exception("text-to-image failed")
        return _error(500, str(exc))


@app.post("/v1/image-to-image")
async def image_to_image(body: ImageToImageRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    if not config.LOAD_FLUX:
        return _error(500, "Flux not enabled")
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    try:
        image_path = str(uploads.resolve(body.image_uri))
        width = (body.width // 16) * 16
        height = (body.height // 16) * 16
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            image_bytes = await flux.generate_image_to_image(
                prompt=body.prompt,
                image_path=image_path,
                width=width,
                height=height,
                num_inference_steps=body.num_inference_steps,
                guidance_scale=body.guidance_scale,
                seed=seed,
                model=body.model,
                turbo=body.turbo,
                lora_path=lora_path,
                lora_strength=lora_strength,
                turbo_steps=_flux_config["turbo_steps"],
                turbo_guidance=_flux_config["turbo_guidance"],
            )
        return Response(content=image_bytes, media_type="image/webp")
    except FluxLoraError as exc:
        return _error(422, str(exc))
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("image-to-image failed")
        return _error(500, str(exc))


@app.post("/v1/image-edit")
async def image_edit(body: ImageEditRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    # joyai-edit routes to the out-of-process sidecar; the flux-backed
    # models still require LOAD_FLUX.
    if body.model == "joyai-edit":
        await _auto_exit_turbo_if_active("joyai-edit (v1)")
        if len(body.image_uris) != 1:
            return _error(422, "joyai-edit requires exactly one image_uri")
        if not config.LOAD_JOYAI:
            return _error(503, "JoyAI not enabled (LOAD_JOYAI=0)")
        try:
            if ernie.is_loaded:
                logger.info("Evicting ERNIE-Image from cuda:1 for JoyAI")
                await ernie.unload()
            await joyai.load()  # idempotent — ensures sidecar pipeline is loaded on cuda:1
            image_path = str(uploads.resolve(body.image_uris[0]))
            width = (body.width // 16) * 16
            height = (body.height // 16) * 16
            seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
            # JoyAI is on cuda:1 sidecar -- no inference lock needed
            image_bytes = await joyai.edit(
                prompt=body.prompt,
                image_path=image_path,
                width=width,
                height=height,
                num_inference_steps=body.num_inference_steps,
                guidance_scale=body.guidance_scale,
                seed=seed,
            )
            return Response(content=image_bytes, media_type="image/webp")
        except JoyAIError as exc:
            return _error(exc.status_code, str(exc))
        except FileNotFoundError as exc:
            return _error(404, str(exc))
        except Exception as exc:
            logger.exception("image-edit (joyai) failed")
            return _error(500, str(exc))

    if not config.LOAD_FLUX:
        return _error(500, "Flux pipeline not loaded")
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    # v1.8.0: preserve_identity is Klein-only (hooks target Flux2KleinKVPipeline)
    if body.preserve_identity and body.model != "flux2-klein":
        return _error(422, "preserve_identity_klein_only")
    # Zero-strength is just a no-op — downgrade quietly so the hook path is skipped.
    effective_preserve = body.preserve_identity and body.identity_strength > 0.0
    try:
        image_paths = [str(uploads.resolve(uri)) for uri in body.image_uris]
        width = (body.width // 16) * 16
        height = (body.height // 16) * 16
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

        async with _inference_lock:
            await _ensure_flux_ready()
            torch.cuda.set_device(config.FLUX_DEVICE)
            image_bytes = await flux.generate_image_edit(
                prompt=body.prompt,
                image_paths=image_paths,
                width=width,
                height=height,
                num_inference_steps=body.num_inference_steps,
                guidance_scale=body.guidance_scale,
                seed=seed,
                model=body.model,
                lora_path=lora_path,
                lora_strength=lora_strength,
                preserve_identity=effective_preserve,
                identity_strength=body.identity_strength,
                identity_mode=body.identity_mode,
            )
        return Response(content=image_bytes, media_type="image/webp")
    except FluxLoraError as exc:
        return _error(422, str(exc))
    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("image-edit failed")
        return _error(500, str(exc))


@app.post("/v1/music")
async def generate_music(body: MusicGenerationRequest) -> Response:
    if _paused:
        return _error(503, "System is paused for maintenance")
    await _auto_exit_turbo_if_active("music (v1)")
    if not config.LOAD_ACE:
        return _error(503, "Music generation not enabled (LOAD_ACE=0)")
    # Validate task-type requirements
    if body.task_type in ("cover", "repaint", "extract", "lego", "complete") and not body.source_audio_uri:
        return _error(422, f"task_type '{body.task_type}' requires source_audio_uri")
    if body.task_type in ("extract", "lego", "complete") and not body.track_name:
        return _error(422, f"task_type '{body.task_type}' requires track_name")
    # Resolve URIs
    params = body.model_dump(exclude_none=True)
    if body.source_audio_uri:
        try:
            params["source_audio_path"] = str(uploads.resolve(body.source_audio_uri))
        except FileNotFoundError:
            return _error(404, "source_audio_uri not found")
    if body.reference_audio_uri:
        try:
            params["reference_audio_path"] = str(uploads.resolve(body.reference_audio_uri))
        except FileNotFoundError:
            return _error(404, "reference_audio_uri not found")
    params.pop("source_audio_uri", None)
    params.pop("reference_audio_uri", None)
    try:
        # Free JoyAI's 50 GB on cuda:1 — reloads on next joyai-edit request
        try:
            await joyai.unload()
        except Exception:
            pass
        ace_params = _build_ace_params(params)
        audio_bytes = await ace.generate(params=ace_params)
        media_type = _AUDIO_MEDIA_TYPES.get(body.audio_format, "audio/mpeg")
        return Response(content=audio_bytes, media_type=media_type)
    except AceError as exc:
        return _error(exc.status_code, str(exc))
    except Exception as exc:
        logger.exception("music generation failed")
        return _error(500, str(exc))


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> JSONResponse:
    if not chat.is_ready:
        return _error(500, "Chat model not loaded")
    if not body.messages:
        return _error(422, "Messages list cannot be empty")
    try:
        messages = [m.model_dump() for m in body.messages]
        result = await chat.generate_chat_completion(
            messages=messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("chat completion failed")
        return _error(500, str(exc))


@app.post("/v1/upload")
async def upload(request: Request) -> JSONResponse:
    upload_id, storage_uri = uploads.create()

    # Build upload_url from the incoming request's host
    base = str(request.base_url).rstrip("/")
    upload_url = f"{base}/uploads/put/{upload_id}"

    return JSONResponse(
        content={
            "upload_url": upload_url,
            "storage_uri": storage_uri,
            "required_headers": {},
        }
    )


@app.put("/uploads/put/{upload_id}")
async def upload_put(upload_id: str, request: Request) -> Response:
    from upload_store import MAX_UPLOAD_BYTES
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        return _error(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit")
    # v1.8.2 / SEC P2-3: early quota peek using Content-Length so we reject
    # BEFORE reading the whole body into memory.
    api_key = _extract_api_key(request) or ""
    if content_length:
        try:
            proposed = int(content_length)
        except ValueError:
            proposed = 0
        quota = _upload_quota_error(api_key, proposed)
        if quota is not None:
            return quota
    data = await request.body()
    if len(data) > MAX_UPLOAD_BYTES:
        return _error(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit")
    # SEC P2-3 final check after read (in case Content-Length was absent).
    quota = _upload_quota_error(api_key, len(data))
    if quota is not None:
        return quota
    # SEC P2-7 (v1.8.2): magic-byte check against declared Content-Type.
    declared = request.headers.get("content-type", "")
    if not _content_type_matches_magic(declared, data[:16]):
        return JSONResponse(
            status_code=422,
            content={
                "error": "content_type_mismatch",
                "message": "body does not match declared Content-Type",
                "detail": f"declared={declared or '(none)'}",
            },
        )
    uploads.save(upload_id, data)
    _record_upload_bytes(api_key, len(data))
    return Response(status_code=201)


@app.get("/uploads/get/{upload_id}")
async def upload_get(upload_id: str, request: Request) -> Response:
    """Read back a previously-uploaded file (v1.9.1).

    The upload_id is a 128-bit uuid4 hex, unforgeable. Any caller with the ID
    and a valid bearer token can fetch the file — the ID itself is the
    capability. See plan doc for deferred scoping policy (TTL, per-key
    ownership, signed URLs, range requests — all explicitly out of scope).
    """
    try:
        path = uploads.resolve(f"storage://{upload_id}")
    except ValueError:
        return _error(400, "invalid_upload_id")
    except FileNotFoundError:
        return _error(404, "upload_not_found")
    # Sniff only the first 16 bytes — don't read the whole file into memory.
    try:
        with path.open("rb") as f:
            head = f.read(16)
    except OSError:
        return _error(500, "upload_read_failed")
    media_type = _infer_media_type_from_magic(head)
    return FileResponse(path=str(path), media_type=media_type)


# ---------------------------------------------------------------------------
# LoRA Management Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/loras")
async def list_loras() -> JSONResponse:
    loras = lora_registry.list_all()
    return JSONResponse(content={
        "loras": [
            {"id": l.id, "name": l.name, "filename": l.filename, "base_model": l.base_model,
             "size_bytes": l.size_bytes, "uploaded_at": l.uploaded_at, "description": l.description,
             "trigger_word": l.trigger_word, "strategy": l.strategy}
            for l in loras
        ],
        "count": len(loras),
    })


@app.post("/v1/loras", status_code=201)
async def upload_lora(request: Request) -> Response:
    from fastapi import UploadFile
    import io

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        return _error(400, "Expected multipart/form-data")

    api_key = _extract_api_key(request) or ""

    # v1.8.2 / SEC P2-4: total-active-LoRA cap per key. Stored per-key count
    # is derived from the in-memory counter map populated on successful
    # registry.add (+decrement on delete_lora). Unlike upload bytes (rolling
    # 24h), LoRAs stick — each add occupies a slot until the user deletes it.
    if api_key and config.API_KEYS and _get_key_count(_per_key_lora_counts, api_key) >= config.PER_KEY_LORA_COUNT:
        return JSONResponse(
            status_code=429,
            content={
                "error": "per_key_lora_count_exceeded",
                "message": f"LoRA limit of {config.PER_KEY_LORA_COUNT} per key reached — delete some first.",
            },
            headers={"Retry-After": "3600"},
        )

    form = await request.form(max_part_size=config.MAX_LORA_SIZE_BYTES)
    file = form.get("file")
    name = form.get("name")
    description = str(form.get("description", ""))
    base_model = str(form.get("base_model", "ltx-2.3"))
    trigger_word = form.get("trigger_word")
    strategy = form.get("strategy")
    trigger_word = str(trigger_word) if trigger_word else None
    strategy = str(strategy) if strategy else None

    if not file or not hasattr(file, "read"):
        return _error(400, "Missing 'file' field")
    if not name:
        return _error(422, "Missing 'name' field")

    filename = getattr(file, "filename", "unknown.safetensors") or "unknown.safetensors"
    if not filename.endswith(".safetensors"):
        return _error(400, "File must be a .safetensors file")

    data = await file.read()
    if len(data) > config.MAX_LORA_SIZE_BYTES:
        return _error(413, f"File exceeds {config.MAX_LORA_SIZE_BYTES // (1024*1024)}MB limit")

    # v1.8.2 / SEC P2-3: LoRA upload bytes count toward the 24h rolling
    # byte quota too — they're just user-provided bytes on disk.
    quota = _upload_quota_error(api_key, len(data))
    if quota is not None:
        return quota

    try:
        info = lora_registry.add(name=str(name), filename=filename, data=data, description=description,
                                base_model=base_model, trigger_word=trigger_word, strategy=strategy)
    except ValueError as exc:
        return _error(400, str(exc))

    _record_upload_bytes(api_key, len(data))
    _incr_key_count(_per_key_lora_counts, api_key)

    return JSONResponse(
        status_code=201,
        content={"id": info.id, "name": info.name, "filename": info.filename,
                 "base_model": info.base_model, "size_bytes": info.size_bytes,
                 "uploaded_at": info.uploaded_at, "description": info.description,
                 "trigger_word": info.trigger_word, "strategy": info.strategy},
    )


@app.delete("/v1/loras/{lora_id}")
async def delete_lora(lora_id: str, request: Request) -> JSONResponse:
    if not lora_registry.delete(lora_id):
        return _error(404, f"LoRA not found: {lora_id}")
    # SEC P2-4: free a slot for the caller. The registry doesn't record
    # ownership today, so we decrement the caller's counter regardless of
    # who originally uploaded. Worst case: a heavy user deletes someone
    # else's LoRA and claws back a slot for themselves. Acceptable because
    # the caller still had to auth and get here; MAX_LORA_SIZE_BYTES and
    # PER_KEY_UPLOAD_BYTES_PER_DAY still rate-limit the downstream add.
    api_key = _extract_api_key(request) or ""
    _decr_key_count(_per_key_lora_counts, api_key)
    return JSONResponse(content={"deleted": True, "id": lora_id})


# ---------------------------------------------------------------------------
# Flux LoRA endpoints (folder-drop — no upload, no delete)
# ---------------------------------------------------------------------------


@app.get("/v1/flux-loras")
async def list_flux_loras() -> JSONResponse:
    loras = flux_lora_registry.list_all()
    return JSONResponse(content={
        "loras": [
            {"id": l.id, "name": l.name, "filename": l.filename,
             "size_bytes": l.size_bytes, "model_compat": l.model_compat,
             "description": l.description, "trigger_word": l.trigger_word}
            for l in loras
        ],
        "count": len(loras),
    })


@app.post("/v1/flux-loras/rescan")
async def rescan_flux_loras() -> JSONResponse:
    count = flux_lora_registry.rescan()
    return JSONResponse(content={"rescanned": True, "count": count})


# ---------------------------------------------------------------------------
# V2 Async Job Endpoints
# ---------------------------------------------------------------------------


def _decr_queue_on_complete(job: Job) -> None:
    """v1.8.2 / SEC P1-3: worker_loop callback — decrement the per-key
    queue counter when a submitted-via-_submit_job job reaches a terminal
    state. Music and batch items decrement themselves (they don't go
    through worker_loop).
    """
    _decr_key_count(_per_key_queue_counts, job.api_key or "")


def _submit_job(job_type: JobType, params: dict, request: Request, raw: dict | None = None) -> JSONResponse:
    """Create a job, enqueue it, return 202.

    ``raw`` captures the client-facing request body (via Pydantic
    ``model_dump(mode="json")``) so history_store can persist the exact user
    intent — storage:// URIs, keyframe sub-models, etc. — rather than the
    lowered dispatch params.
    """
    if _paused:
        return JSONResponse(
            status_code=503,
            content={"error": "system_paused", "message": "System is paused for maintenance."},
            headers={"Retry-After": "300"},
        )

    auth = request.headers.get("Authorization", "")
    api_key = auth[7:] if auth.startswith("Bearer ") else ""

    # v1.8.2 / SEC P1-3: per-key cap enforced BEFORE the global depth check
    # so one bearer can't claim the whole queue.
    if api_key and config.API_KEYS and _get_key_count(_per_key_queue_counts, api_key) >= config.PER_KEY_QUEUE_CAP:
        return JSONResponse(
            status_code=429,
            content={
                "error": "per_key_queue_full",
                "message": f"You have {config.PER_KEY_QUEUE_CAP} jobs in flight; wait for one to finish.",
            },
            headers={"Retry-After": "30"},
        )

    if job_store.pending_count() >= config.MAX_QUEUE_DEPTH:
        return JSONResponse(
            status_code=429,
            content={"error": "queue_full", "message": "Job queue is full. Try again later."},
            headers={"Retry-After": "30"},
        )

    job = Job(id=make_job_id(), type=job_type, params=params, api_key=api_key)
    job.raw_request = raw
    _incr_key_count(_per_key_queue_counts, api_key)
    job_store.add(job)
    _job_queue.put_nowait(job.id)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.id,
            "status": "queued",
            "poll_url": f"/v2/jobs/{job.id}",
            "stream_url": f"/v2/jobs/{job.id}/stream",
        },
    )


@app.post("/v2/text-to-video")
async def v2_text_to_video(body: TextToVideoRequest, request: Request) -> JSONResponse:
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    width, height = _resolution_to_dims(body.resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    prompt = _build_prompt(body.prompt, body.camera_motion)
    seed = random.randint(0, 2**32 - 1)
    params = dict(prompt=prompt, model=body.model, width=width, height=height,
                  num_frames=num_frames, fps=body.fps, seed=seed, generate_audio=body.generate_audio,
                  lora_path=lora_path, lora_strength=lora_strength,
                  enhance_prompt=body.enhance_prompt)
    return _submit_job(JobType.TEXT_TO_VIDEO, params, request, raw=body.model_dump(mode="json"))


@app.post("/v2/image-to-video")
async def v2_image_to_video(body: ImageToVideoRequest, request: Request) -> JSONResponse:
    width, height = _resolution_to_dims(body.resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    keyframe_inputs = _resolve_keyframes(body, num_frames)
    if isinstance(keyframe_inputs, JSONResponse):
        return keyframe_inputs
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    seed = random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, keyframes=keyframe_inputs, model=body.model,
                  width=width, height=height, num_frames=num_frames, fps=body.fps,
                  seed=seed, generate_audio=body.generate_audio,
                  lora_path=lora_path, lora_strength=lora_strength,
                  enhance_prompt=body.enhance_prompt)
    return _submit_job(JobType.IMAGE_TO_VIDEO, params, request, raw=body.model_dump(mode="json"))


@app.post("/v2/audio-to-video")
async def v2_audio_to_video(body: AudioToVideoRequest, request: Request) -> JSONResponse:
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    audio_path = str(uploads.resolve(body.audio_uri))
    width, height = _resolution_to_dims(body.resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    keyframe_inputs = _resolve_keyframes(body, num_frames, allow_neither=True)
    if isinstance(keyframe_inputs, JSONResponse):
        return keyframe_inputs
    seed = random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, audio_path=audio_path, image_path=None,
                  keyframes=keyframe_inputs,
                  model=body.model, width=width, height=height, num_frames=num_frames,
                  fps=body.fps, seed=seed,
                  lora_path=lora_path, lora_strength=lora_strength,
                  enhance_prompt=body.enhance_prompt)
    return _submit_job(JobType.AUDIO_TO_VIDEO, params, request, raw=body.model_dump(mode="json"))


@app.post("/v2/retake")
async def v2_retake(body: RetakeRequest, request: Request) -> JSONResponse:
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    video_path = str(uploads.resolve(body.video_uri))
    prompt = body.prompt or ""
    seed = random.randint(0, 2**32 - 1)
    params = dict(video_path=video_path, start_time=body.start_time,
                  duration=body.duration, mode=body.mode, prompt=prompt, seed=seed,
                  lora_path=lora_path, lora_strength=lora_strength)
    return _submit_job(JobType.RETAKE, params, request, raw=body.model_dump(mode="json"))


DEFAULT_OUTPAINT_LORA_ID = "ic-lora-outpaint"


@app.post("/v2/video-outpaint")
async def v2_video_outpaint(body: VideoOutpaintRequest, request: Request) -> JSONResponse:
    # Default to the registered outpaint IC-LoRA when the client omits one.
    if body.lora is None:
        body = body.model_copy(update={"lora": LoRAInput(id=DEFAULT_OUTPAINT_LORA_ID, strength=1.0)})
    lora_result = _resolve_lora(body)
    if isinstance(lora_result, JSONResponse):
        return lora_result
    lora_path, lora_strength = lora_result
    if lora_path is None:
        return _error(500, "outpaint LoRA resolve returned None — registry misconfigured")
    video_path = str(uploads.resolve(body.video_uri))
    width, height = _resolution_to_dims(body.target_resolution)
    num_frames = _duration_to_frames(body.duration, body.fps)
    seed = body.seed if body.seed else random.randint(0, 2**32 - 1)
    params = dict(
        video_path=video_path, prompt=body.prompt,
        target_width=width, target_height=height, position=body.position,
        num_frames=num_frames, fps=body.fps, seed=seed,
        conditioning_strength=body.conditioning_strength,
        skip_stage_2=body.skip_stage_2,
        lora_path=lora_path, lora_strength=lora_strength,
        enhance_prompt=body.enhance_prompt,
        # History-only fields (stripped before generate_outpaint call):
        width=width, height=height, model="ic-lora-outpaint",
    )
    return _submit_job(JobType.VIDEO_OUTPAINT, params, request, raw=body.model_dump(mode="json"))


@app.post("/v2/video/extract-frames")
async def v2_video_extract_frames(body: ExtractFramesRequest, request: Request) -> JSONResponse:
    """v1.10.0 unit B — server-side PyAV frame extractor for chain conditioning.

    Decodes the requested frame indices out of a stored MP4 in a single pass,
    re-saves each as a lossless PNG upload, and returns the resulting
    storage:// URIs. Powers the multi-frame chain conditioning flow in
    noodle-v (composition export seam elimination).

    Security model: capability URL — bearer unlocks the endpoint, then any
    caller with the upload id can fetch the bytes. Same shape as
    /uploads/get/{id} (v1.9.1). Output bytes count against
    PER_KEY_UPLOAD_BYTES_PER_DAY.
    """
    api_key = _extract_api_key(request) or ""
    try:
        video_path = uploads.resolve(body.video_uri)
    except (ValueError, FileNotFoundError):
        return _error(404, "video_not_found")

    indices = list(body.frame_indices)  # already sorted+deduped by validator

    def _decode_and_encode() -> tuple[list[bytes], list[tuple[int, int]]]:
        """Returns (per-frame PNG bytes, per-frame (width, height))."""
        from history_store import _extract_frames_as_pils
        import io as _io
        video_bytes = video_path.read_bytes()
        pils = _extract_frames_as_pils(video_bytes, indices)
        png_blobs: list[bytes] = []
        dims: list[tuple[int, int]] = []
        for img in pils:
            buf = _io.BytesIO()
            img.save(buf, format="PNG", compress_level=6)
            png_blobs.append(buf.getvalue())
            dims.append((img.width, img.height))
        return png_blobs, dims

    try:
        async with _FRAME_EXTRACT_SEMAPHORE:
            png_blobs, dims = await asyncio.wait_for(
                asyncio.to_thread(_decode_and_encode), timeout=30.0,
            )
    except asyncio.TimeoutError:
        logger.warning("extract-frames timed out for %s indices=%s", body.video_uri, indices)
        return _error(504, "pyav_timeout")
    except IndexError as exc:
        return _error(422, f"frame_index_out_of_range: {exc}")
    except RuntimeError as exc:
        logger.warning("extract-frames decode failure for %s: %s", body.video_uri, exc)
        return _error(500, "extract_failed")
    except Exception:
        logger.exception("extract-frames unexpected failure for %s", body.video_uri)
        return _error(500, "extract_failed")

    total_bytes = sum(len(b) for b in png_blobs)
    quota = _upload_quota_error(api_key, total_bytes)
    if quota is not None:
        return quota

    frames_out = []
    for idx, png, (w, h) in zip(indices, png_blobs, dims):
        upload_id, storage_uri = uploads.create()
        uploads.save(upload_id, png)
        frames_out.append({
            "frame_index": idx,
            "storage_uri": storage_uri,
            "width": w,
            "height": h,
        })
    _record_upload_bytes(api_key, total_bytes)
    return JSONResponse(content={"frames": frames_out})


@app.post("/v2/text-to-image")
async def v2_text_to_image(body: TextToImageRequest, request: Request) -> JSONResponse:
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    width = (body.width // 16) * 16
    height = (body.height // 16) * 16
    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, width=width, height=height,
                  num_inference_steps=body.num_inference_steps,
                  guidance_scale=body.guidance_scale, seed=seed,
                  model=body.model, turbo=body.turbo,
                  lora_path=lora_path, lora_strength=lora_strength)
    return _submit_job(JobType.TEXT_TO_IMAGE, params, request, raw=body.model_dump(mode="json"))


@app.post("/v2/image-to-image")
async def v2_image_to_image(body: ImageToImageRequest, request: Request) -> JSONResponse:
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    image_path = str(uploads.resolve(body.image_uri))
    width = (body.width // 16) * 16
    height = (body.height // 16) * 16
    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, image_path=image_path, width=width, height=height,
                  num_inference_steps=body.num_inference_steps,
                  guidance_scale=body.guidance_scale, seed=seed,
                  model=body.model, turbo=body.turbo,
                  lora_path=lora_path, lora_strength=lora_strength)
    return _submit_job(JobType.IMAGE_TO_IMAGE, params, request, raw=body.model_dump(mode="json"))


@app.post("/v2/image-edit")
async def v2_image_edit(body: ImageEditRequest, request: Request) -> JSONResponse:
    flux_lora_result = _resolve_flux_lora(body)
    if isinstance(flux_lora_result, JSONResponse):
        return flux_lora_result
    lora_path, lora_strength = flux_lora_result
    # v1.8.0: preserve_identity is Klein-only (hooks target Flux2KleinKVPipeline)
    if body.preserve_identity and body.model != "flux2-klein":
        return _error(422, "preserve_identity_klein_only")
    effective_preserve = body.preserve_identity and body.identity_strength > 0.0
    image_paths = [str(uploads.resolve(uri)) for uri in body.image_uris]
    width = (body.width // 16) * 16
    height = (body.height // 16) * 16
    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
    params = dict(prompt=body.prompt, image_paths=image_paths, width=width, height=height,
                  num_inference_steps=body.num_inference_steps,
                  guidance_scale=body.guidance_scale, seed=seed,
                  model=body.model,
                  lora_path=lora_path, lora_strength=lora_strength,
                  preserve_identity=effective_preserve,
                  identity_strength=body.identity_strength,
                  identity_mode=body.identity_mode)
    return _submit_job(JobType.IMAGE_EDIT, params, request, raw=body.model_dump(mode="json"))


@app.post("/v2/music")
async def v2_music(body: MusicGenerationRequest, request: Request) -> JSONResponse:
    if _paused:
        return JSONResponse(status_code=503, content={"error": "system_paused"}, headers={"Retry-After": "300"})
    if _turbo_active:
        return JSONResponse(
            status_code=503,
            content={"error": "turbo_mode_active: ACE/JoyAI unavailable while turbo mode is enabled. Disable turbo first."},
            headers={"Retry-After": "10"},
        )
    if not config.LOAD_ACE:
        return _error(503, "Music generation not enabled (LOAD_ACE=0)")
    # Validate task-type requirements
    if body.task_type in ("cover", "repaint", "extract", "lego", "complete") and not body.source_audio_uri:
        return _error(422, f"task_type '{body.task_type}' requires source_audio_uri")
    if body.task_type in ("extract", "lego", "complete") and not body.track_name:
        return _error(422, f"task_type '{body.task_type}' requires track_name")
    auth = request.headers.get("Authorization", "")
    api_key_check = auth[7:] if auth.startswith("Bearer ") else ""
    # v1.8.2 / SEC P1-3: per-key music cap BEFORE global cap.
    if api_key_check and config.API_KEYS and _get_key_count(_per_key_music_counts, api_key_check) >= config.PER_KEY_MUSIC_CAP:
        return JSONResponse(
            status_code=429,
            content={
                "error": "per_key_queue_full",
                "message": f"You have {config.PER_KEY_MUSIC_CAP} music jobs in flight.",
            },
            headers={"Retry-After": "30"},
        )
    # Queue depth check for music
    music_pending = sum(
        1 for j in job_store._jobs.values()
        if j.type == JobType.MUSIC_GENERATION and j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
    )
    if music_pending >= config.MAX_MUSIC_PENDING:
        return JSONResponse(status_code=429, content={"error": "music_queue_full"}, headers={"Retry-After": "30"})
    # Resolve URIs
    params = body.model_dump(exclude_none=True)
    if body.source_audio_uri:
        try:
            params["source_audio_path"] = str(uploads.resolve(body.source_audio_uri))
        except FileNotFoundError:
            return _error(404, "source_audio_uri not found")
    if body.reference_audio_uri:
        try:
            params["reference_audio_path"] = str(uploads.resolve(body.reference_audio_uri))
        except FileNotFoundError:
            return _error(404, "reference_audio_uri not found")
    params.pop("source_audio_uri", None)
    params.pop("reference_audio_uri", None)
    api_key = api_key_check
    job = Job(id=make_job_id(), type=JobType.MUSIC_GENERATION, params=params, api_key=api_key)
    _incr_key_count(_per_key_music_counts, api_key)
    job_store.add(job)
    asyncio.create_task(_run_music_job(job))
    return JSONResponse(status_code=202, content={
        "job_id": job.id, "status": "queued",
        "poll_url": f"/v2/jobs/{job.id}",
        "stream_url": f"/v2/jobs/{job.id}/stream",
    })


@app.get("/v2/jobs/{job_id}")
async def v2_job_status(job_id: str, request: Request) -> JSONResponse:
    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")
    deny = _require_owner(job.api_key, request)
    if deny is not None:
        return deny
    return JSONResponse(content={
        "job_id": job.id,
        "status": job.status,
        "type": job.type,
        "progress": job.progress if job.status == JobStatus.PROCESSING else (1.0 if job.status == JobStatus.COMPLETED else None),
        "phase": job.phase if job.status == JobStatus.PROCESSING else None,
        "queue_position": job_store.queue_position(job.id) if job.status == JobStatus.QUEUED else None,
        "error": {"code": job.error_code or "generation_failed", "message": job.error} if job.error else None,
        "result_url": f"/v2/jobs/{job.id}/result" if job.status == JobStatus.COMPLETED else None,
        "result_storage_uri": job.result_uri if job.status == JobStatus.COMPLETED else None,
        "result_media_type": job.result_media_type,
    })


@app.get("/v2/jobs/{job_id}/preview")
async def v2_job_preview(job_id: str, request: Request) -> Response:
    """Return a low-res preview JPEG for a job in progress or completed.

    Four paths:
    1. Fast path — `job.preview_bytes` already populated (Flux step-end callback,
       or history-save backfill on the worker). Serve the cached JPEG.
    2. On-disk thumbnail path — `history.save()` has already written a thumbnail
       to `config.THUMBNAIL_DIR / thumb_{upload_id}`. Serve it as a zero-copy
       `FileResponse` (sendfile). Survives restart because it's on disk.
    3. Fallback lazy extraction — completed video job with no thumbnail on disk
       (e.g. no api_key so history.save was skipped, or the save task failed).
       Read + decode the first frame via PyAV, but offload the 100+ MB read and
       the PyAV call to a thread so the event loop isn't blocked.
    4. 204 — queued / processing / failed without result bytes. Frontends should
       keep polling; `404` shows up red in browser dev tools and confuses users
       into thinking the job is broken.
    """
    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")
    deny = _require_owner(job.api_key, request)
    if deny is not None:
        return deny

    # Path 1: cached preview in RAM (Flux step-end, or worker backfill)
    if job.preview_bytes:
        return Response(content=job.preview_bytes, media_type="image/jpeg")

    # Path 2: on-disk thumbnail from history.save() — zero-copy sendfile
    if job.result_uri:
        upload_id = job.result_uri.removeprefix("storage://")
        thumb_path = config.THUMBNAIL_DIR / f"thumb_{upload_id}"
        if thumb_path.exists():
            return FileResponse(path=str(thumb_path), media_type="image/jpeg")

    # Path 3: fallback lazy extraction for completed video jobs without an
    # on-disk thumbnail. Offloaded to a thread so the 100+ MB read + PyAV
    # decode don't block the event loop. v1.9.7: bounded concurrency +
    # timeout so a malformed MP4 can't starve the thread pool.
    if (
        job.status == JobStatus.COMPLETED
        and job.result_uri
        and (job.result_media_type or "").startswith("video/")
    ):
        try:
            result_path = uploads.resolve(job.result_uri)
            if result_path.exists():
                from history_store import _first_video_frame_as_pil

                def _extract() -> bytes | None:
                    video_bytes = result_path.read_bytes()
                    frame = _first_video_frame_as_pil(video_bytes)
                    if frame is None:
                        return None
                    import io as _io
                    buf = _io.BytesIO()
                    frame.convert("RGB").save(buf, format="JPEG", quality=80)
                    return buf.getvalue()

                async with _PREVIEW_EXTRACT_SEMAPHORE:
                    try:
                        preview = await asyncio.wait_for(
                            asyncio.to_thread(_extract), timeout=8.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Preview extract timed out for job %s — likely malformed MP4",
                            job_id,
                        )
                        preview = None
                if preview is not None:
                    job.preview_bytes = preview  # cache for subsequent polls
                    return Response(content=preview, media_type="image/jpeg")
        except Exception:
            logger.exception("Failed to extract video preview for job %s", job_id)

    # Path 4: no preview available — 204, not 404
    return Response(status_code=204)


@app.get("/v2/jobs/{job_id}/result")
async def v2_job_result(job_id: str, request: Request) -> Response:
    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")
    deny = _require_owner(job.api_key, request)
    if deny is not None:
        return deny
    if job.status != JobStatus.COMPLETED or not job.result_uri:
        return _error(409, "Job result not ready")
    try:
        path = uploads.resolve(job.result_uri)
        if not path.exists():
            raise FileNotFoundError()
        return FileResponse(
            path=str(path),
            media_type=job.result_media_type or "application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )
    except FileNotFoundError:
        return _error(404, "Result file expired or not found")


@app.delete("/v2/jobs/{job_id}")
async def v2_cancel_job(job_id: str, request: Request) -> JSONResponse:
    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")
    deny = _require_owner(job.api_key, request)
    if deny is not None:
        return deny
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return _error(409, "Cannot cancel a finished job")
    job.status = JobStatus.CANCELLED
    return JSONResponse(content={"job_id": job.id, "status": "cancelled"})


@app.get("/v2/jobs/{job_id}/stream")
async def v2_job_stream(
    job_id: str, request: Request, token: str | None = None,
) -> Response:
    """SSE stream for live job status + progress + phase updates.

    Eliminates client-side polling: instead of hitting `/v2/jobs/{id}` every
    500 ms for two minutes (240 GETs per video job), the client opens one
    EventSource and receives push updates whenever (status, progress, phase)
    changes. The stream closes itself with one final event on terminal state
    (completed / failed / cancelled).

    **Auth**: EventSource cannot set custom headers, so pass either a bearer
    `Authorization` header (programmatic clients) or `?token=<sse-token>`
    query param (browsers — get one via `POST /v1/sse-token`).

    **Event format**: each event is a single `data:` line containing the same
    JSON shape as `GET /v2/jobs/{id}`, so clients can reuse their existing
    polling parser. Idle-period keepalive comments (`: keepalive`) are emitted
    every 15 s to prevent intermediate proxies from closing the connection
    during long queue waits.
    """
    # Match the middleware's "no keys configured = auth disabled" mode, since
    # this endpoint bypasses the middleware to allow `?token=` query-param auth
    # (EventSource can't set headers).
    api_key: str | None = None
    if config.API_KEYS:
        api_key = _resolve_sse_token(token) or _extract_api_key(request)
        if not api_key:
            return _error(401, "Missing API key")

    job = job_store.get(job_id)
    if job is None:
        return _error(404, "Job not found")
    # SEC P0-1: ownership gate. Reuse the api_key we already resolved above
    # rather than re-resolving via _require_owner. Constant-time compare;
    # 404 on mismatch (same shape as job-missing, no existence oracle).
    if config.API_KEYS and job.api_key and api_key and not _secrets.compare_digest(job.api_key, api_key):
        return _error(404, "Job not found")

    import json as _json
    import time as _time

    def _snapshot(j: Job) -> dict:
        return {
            "job_id": j.id,
            "status": j.status,
            "type": j.type,
            "progress": j.progress,
            "phase": j.phase,
            "queue_position": (
                job_store.queue_position(j.id) if j.status == JobStatus.QUEUED else None
            ),
            "error": (
                {"code": j.error_code or "generation_failed", "message": j.error}
                if j.error else None
            ),
            "result_url": f"/v2/jobs/{j.id}/result" if j.status == JobStatus.COMPLETED else None,
            "result_storage_uri": j.result_uri if j.status == JobStatus.COMPLETED else None,
            "result_media_type": j.result_media_type,
        }

    async def event_stream():
        # Dedup key — only emit when something observable changes.
        # Round progress to 3 decimals so micro-fluctuations don't flood the stream.
        last_key: tuple | None = None
        last_keepalive = _time.monotonic()
        terminal = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        try:
            while True:
                if await request.is_disconnected():
                    return
                j = job_store.get(job_id)
                if j is None:
                    # Job expired from the store (cleanup TTL). Surface it and close.
                    yield f"event: error\ndata: {_json.dumps({'error': 'job_expired'})}\n\n"
                    return
                key = (j.status, round(j.progress, 3), j.phase, j.error_code)
                if key != last_key:
                    yield f"data: {_json.dumps(_snapshot(j))}\n\n"
                    last_key = key
                    last_keepalive = _time.monotonic()
                if j.status in terminal:
                    return
                now = _time.monotonic()
                if now - last_keepalive > 15:
                    yield ": keepalive\n\n"
                    last_keepalive = now
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if ever proxied
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# SSE session tokens (short-lived, keeps API key out of URLs)
# ---------------------------------------------------------------------------

_sse_tokens: dict[str, tuple[str, float]] = {}  # token → (api_key, expires_at)


@app.post("/v1/sse-token")
async def create_sse_token(request: Request) -> JSONResponse:
    """Issue a 5-minute disposable token for SSE EventSource connections."""
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    import time as _time
    token = _secrets.token_urlsafe(32)
    _sse_tokens[token] = (api_key, _time.time() + 300)
    # Prune expired tokens
    now = _time.time()
    expired = [t for t, (_, exp) in _sse_tokens.items() if exp < now]
    for t in expired:
        del _sse_tokens[t]
    return JSONResponse(content={"token": token, "expires_in": 300})


def _resolve_sse_token(token: str | None) -> str | None:
    """Resolve an SSE token to its API key, or None if invalid/expired."""
    if not token:
        return None
    import time as _time
    entry = _sse_tokens.get(token)
    if not entry:
        return None
    api_key, expires_at = entry
    if _time.time() > expires_at:
        del _sse_tokens[token]
        return None
    return api_key


# ---------------------------------------------------------------------------
# Approved Images (noodle-i → noodle-v pipeline)
# ---------------------------------------------------------------------------


@app.post("/v1/approved-images")
async def approve_image(request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    body = await request.json()
    image_uri = body.get("image_uri")
    if not image_uri:
        return _error(400, "Missing image_uri")

    import json, hashlib, time as _time
    config.APPROVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = config.APPROVED_IMAGES_DIR / "manifest.json"
    # v1.9.7: cached load; write below invalidates via mtime change.
    # The cache handles SEC P2-10 type guard internally.
    manifest = list(_load_approved_manifest())  # copy before mutation

    entry = {
        "id": hashlib.sha256(f"{image_uri}{_time.time()}".encode()).hexdigest()[:16],
        "image_uri": image_uri,
        "prompt": body.get("prompt", ""),
        "model": body.get("model", ""),
        "width": body.get("width", 0),
        "height": body.get("height", 0),
        "api_key_hash": hashlib.sha256(api_key.encode()).hexdigest(),
        "created_at": _time.time(),
    }
    manifest.insert(0, entry)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return JSONResponse(content={"id": entry["id"], "status": "approved"}, status_code=201)


@app.get("/v1/approved-images")
async def list_approved_images(request: Request, limit: int = 50, offset: int = 0) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")

    import hashlib
    # v1.9.7: cached load, mtime-invalidated. Empty list when no file.
    manifest = _load_approved_manifest()
    if not manifest:
        return JSONResponse(content=[])

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    filtered = [e for e in manifest if e.get("api_key_hash") == key_hash]
    page = filtered[offset : offset + limit]

    results = []
    for e in page:
        r = {k: v for k, v in e.items() if k != "api_key_hash"}
        r["image_url"] = f"/v1/approved-images/{e['id']}/file"
        results.append(r)
    return JSONResponse(content=results)


@app.get("/v1/approved-images/events")
async def approved_images_events(request: Request, token: str | None = None) -> StreamingResponse:
    """SSE stream — emits new approved images as they arrive."""
    api_key = _resolve_sse_token(token) or _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")

    import json, hashlib

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    manifest_path = config.APPROVED_IMAGES_DIR / "manifest.json"

    async def event_stream():
        seen_ids: set[str] = set()
        last_mtime: float = 0

        while True:
            if await request.is_disconnected():
                break

            try:
                if manifest_path.exists():
                    mtime = manifest_path.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        raw = json.loads(manifest_path.read_text())
                        manifest = raw.get("images", raw) if isinstance(raw, dict) else raw
                        if not isinstance(manifest, list):
                            manifest = []
                        filtered = [e for e in manifest if e.get("api_key_hash") == key_hash]

                        for entry in filtered:
                            if entry["id"] not in seen_ids:
                                seen_ids.add(entry["id"])
                                r = {k: v for k, v in entry.items() if k != "api_key_hash"}
                                r["image_url"] = f"/v1/approved-images/{entry['id']}/file"
                                yield f"data: {json.dumps(r)}\n\n"
            except Exception:
                pass

            await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/v1/approved-images/{image_id}/file")
async def get_approved_image_file(image_id: str, request: Request) -> Response:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")

    import hashlib
    # v1.9.7: cached manifest, mtime-invalidated.
    manifest = _load_approved_manifest()
    if not manifest:
        return _error(404, "Not found")
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    entry = next((e for e in manifest if e["id"] == image_id and e.get("api_key_hash") == key_hash), None)
    if not entry:
        return _error(404, "Not found")

    path = uploads.resolve(entry["image_uri"])
    if not path.exists():
        return _error(404, "Image file not found")
    # Approved images are also immutable once approved — cache for 30 days.
    return _serve_with_http_cache(
        path, "image/webp", request, max_age=2_592_000, immutable=True,
    )


# ---------------------------------------------------------------------------
# Char Mode — Vision Ranking
# ---------------------------------------------------------------------------

CHAR_RANKING_PROMPT = """You are a character consistency evaluator for AI image generation. You compare a REFERENCE character image against a GENERATED image to assess how well the generated image preserves the character's identity.

Focus specifically on:
- Facial structure: jaw shape, cheekbones, chin, nose bridge
- Eyes: shape, size, spacing, color, eyelid characteristics
- Head proportions: forehead height, face width-to-height ratio, head size relative to body
- Overall likeness: would someone recognize this as the same person?

Rate each criterion from 1-10 where:
- 1-3: Poor match, clearly different person
- 4-6: Some resemblance but noticeable differences
- 7-8: Good match with minor discrepancies
- 9-10: Excellent match, clearly the same character

Return ONLY valid JSON with no additional text:
{
  "score": <float, average of all criteria>,
  "analysis": {
    "face_match": <int 1-10>,
    "eyes": <int 1-10>,
    "proportions": <int 1-10>,
    "overall_likeness": <int 1-10>
  },
  "edits": {
    "add": ["specific description to append to the generation prompt"],
    "remove": ["specific words/phrases to remove from the prompt"],
    "modify": {"aspect_name": "improved description"}
  }
}

Rules for edits:
- Only suggest edits if score < 9
- Keep edits SMALL and SPECIFIC (1-3 items max)
- Focus on the biggest discrepancy between reference and generated
- Use concrete terms: "narrower jawline" not "better face"
- If score >= 9, return empty edits: {"add": [], "remove": [], "modify": {}}"""


@app.post("/v2/char/rank")
async def v2_char_rank(body: CharRankRequest, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    if not chat.is_ready:
        return _error(500, "Chat model not loaded")

    try:
        import base64

        rank_path = uploads.resolve(body.rank_image_uri)
        gen_path = uploads.resolve(body.generated_image_uri)

        rank_b64 = base64.b64encode(rank_path.read_bytes()).decode()
        gen_b64 = base64.b64encode(gen_path.read_bytes()).decode()

        messages = [
            {"role": "system", "content": CHAR_RANKING_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f'Original prompt: "{body.prompt}"\n\nFirst image is the REFERENCE character. Second image is the GENERATED result. Evaluate character consistency.'},
                {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{rank_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{gen_b64}"}},
            ]}
        ]

        result = await chat.generate_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            model=config.CHAR_VISION_MODEL,
        )

        import re as _re
        text = result["choices"][0]["message"]["content"]
        json_match = _re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return _error(500, "Vision model did not return valid JSON")

        # v1.8.2 / SEC P1-5: validate the LLM's structured output against
        # the documented schema before handing to the client. Guards against
        # prompt-injection that alters fields, missing keys, wrong types,
        # score-out-of-range, etc.
        from pydantic import ValidationError as _VErr
        try:
            ranking = CharRankResponse.model_validate_json(json_match.group())
        except _VErr as exc:
            detail = str(exc)[:500]
            return JSONResponse(
                status_code=502,
                content={
                    "error": "char_rank_schema_violation",
                    "message": "Vision model output failed schema validation",
                    "detail": detail,
                },
            )
        return JSONResponse(content=ranking.model_dump())

    except FileNotFoundError as exc:
        return _error(404, str(exc))
    except Exception as exc:
        logger.exception("char/rank failed")
        return _error(500, str(exc))


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@app.get("/v2/history")
async def v2_history(request: Request, limit: int = 50, offset: int = 0, type: str | None = None) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    items = history.list(api_key, limit=min(limit, 200), offset=offset, job_type=type)
    results = []
    for item in items:
        r = {
            "id": item["id"],
            "prompt": item["prompt"],
            "model": item["model"],
            "width": item["width"],
            "height": item["height"],
            "turbo": bool(item["turbo"]),
            "status": item["status"],
            "created_at": item["created_at"],
            "error": item["error"],
        }
        if item["thumbnail_uri"]:
            r["thumbnail_url"] = f"/v2/history/{item['id']}/thumbnail"
        if item["result_uri"]:
            r["image_url"] = f"/v2/history/{item['id']}/image"
        results.append(r)
    return JSONResponse(content=results)


@app.get("/v2/history/{generation_id}/image")
async def v2_history_image(generation_id: str, request: Request) -> Response:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    item = history.get(generation_id, api_key)
    if not item or not item["result_uri"]:
        return _error(404, "Not found")
    path = uploads.resolve(item["result_uri"])
    if not path.exists():
        return _error(404, "Result file not found")
    media_type = "video/mp4" if "video" in item.get("job_type", "") else "image/webp"
    # v1.9.7: result files are immutable for their 30-day retention window.
    # `immutable` is truthful here — once a job completes, its result bytes
    # never change.
    return _serve_with_http_cache(
        path, media_type, request, max_age=2_592_000, immutable=True,
    )


@app.get("/v2/history/{generation_id}/thumbnail")
async def v2_history_thumbnail(generation_id: str, request: Request) -> Response:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    item = history.get(generation_id, api_key)
    if not item or not item["thumbnail_uri"]:
        return _error(404, "Not found")
    thumb_id = item["thumbnail_uri"].removeprefix("thumb://")
    path = config.THUMBNAIL_DIR / thumb_id
    if not path.exists():
        return _error(404, "Thumbnail file not found")
    # v1.9.7: thumb_id is content-addressed (sha/derived). 1-year max-age +
    # immutable is truthful — a new thumbnail gets a new id.
    return _serve_with_http_cache(
        path, "image/jpeg", request, max_age=31_536_000, immutable=True,
    )


@app.delete("/v2/history/{generation_id}")
async def v2_history_delete(generation_id: str, request: Request) -> JSONResponse:
    """Remove a history entry + its result/thumbnail files. Scoped to the
    caller's API key — returns 404 both when the entry doesn't exist and
    when it belongs to another key, so the ID space can't be probed."""
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    if not history.delete(generation_id, api_key):
        return _error(404, "Not found")
    return JSONResponse(content={"ok": True})


@app.get("/v2/history/{generation_id}")
async def v2_history_get(generation_id: str, request: Request) -> JSONResponse:
    """Return the full history record for a single generation, including the
    raw request body (``params``) and generation-config snapshot
    (``gen_config``). Scoped to the caller's API key — returns 404 both when
    the entry doesn't exist and when it belongs to another key."""
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    item = history.get(generation_id, api_key)
    if not item:
        return _error(404, "Not found")

    params_json = item.get("params_json")
    try:
        params = _json_mod.loads(params_json) if params_json else {}
    except (ValueError, TypeError):
        params = {}

    gen_config_json = item.get("gen_config_json")
    try:
        gen_config = _json_mod.loads(gen_config_json) if gen_config_json else None
    except (ValueError, TypeError):
        gen_config = None

    response: dict = {
        "id": item["id"],
        "job_type": item.get("job_type"),
        "prompt": item.get("prompt"),
        "enhanced_prompt": item.get("enhanced_prompt"),
        "model": item.get("model"),
        "width": item.get("width"),
        "height": item.get("height"),
        "seed": item.get("seed"),
        "turbo": bool(item.get("turbo")),
        "status": item.get("status"),
        "created_at": item.get("created_at"),
        "completed_at": item.get("completed_at"),
        "error": item.get("error"),
        "params": params,
        "gen_config": gen_config,
    }
    if item.get("result_uri"):
        response["result_url"] = f"/v2/history/{item['id']}/image"
    else:
        response["result_url"] = None
    if item.get("thumbnail_uri"):
        response["thumbnail_url"] = f"/v2/history/{item['id']}/thumbnail"
    else:
        response["thumbnail_url"] = None
    return JSONResponse(content=response)


def _extract_api_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


# ---------------------------------------------------------------------------
# v1.8.2 security helpers — admin gate, per-key quotas, magic-byte upload
# check. Placed here (after _extract_api_key) because every handler lower in
# the file already forward-references helpers defined in this zone.
# ---------------------------------------------------------------------------


def _sha256_key(api_key: str) -> str:
    import hashlib
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _constant_time_match(token: str, keyset: set[str]) -> bool:
    """Full-iteration bearer compare. SEC P2-1 (v1.8.2).

    ``any(compare_digest ...)`` short-circuits on the first match, which
    leaks set membership via wall-clock timing (caller observes ~N*t for a
    miss, ~k*t for a hit where k ≤ N is the match position). Iterating the
    entire set eliminates the side channel.
    """
    if not token:
        return False
    matched = False
    for key in keyset:
        if _secrets.compare_digest(token, key):
            matched = True
    return matched


def _require_admin(request: Request) -> JSONResponse | None:
    """Admin gate. SEC P0-2 (v1.8.2).

    Enforcement matrix:
      - ``config.API_KEYS`` empty → auth globally off, allow through (parity
        with middleware bypass)
      - ``config.ADMIN_KEYS`` empty → backwards-compat bridge: allow iff the
        caller's bearer is in ``API_KEYS``. A WARN is logged once at startup
        so operators know admin auth is degraded until they create
        ``.admin_keys``.
      - Otherwise → caller's bearer must be in ``ADMIN_KEYS`` via full-
        iteration ``compare_digest``. 403 on mismatch (not 404: admin
        endpoints are known-existent, no existence oracle to protect).
    """
    if not config.API_KEYS:
        return None
    token = _extract_api_key(request) or ""
    if not config.ADMIN_KEYS:
        if _constant_time_match(token, config.API_KEYS):
            return None
        return _error(401, "Invalid or missing API key")
    if _constant_time_match(token, config.ADMIN_KEYS):
        return None
    return _error(403, "admin_required")


# v1.8.2 / SEC P1-3 — in-memory per-key queue counters. Keyed by
# sha256(api_key) so raw bearers never land in the map (debuggers, heap
# dumps, thread-local inspection). Decremented from job_queue.worker_loop
# via the on_complete callback and from _run_music_job's finally block.
_per_key_queue_counts: dict[str, int] = {}
_per_key_music_counts: dict[str, int] = {}
_per_key_batch_counts: dict[str, int] = {}

# SEC P2-4: total active LoRAs per key. Best-effort counter — restarts
# reset it to 0 (until v1.8.3 we don't stamp the registry with api_key).
# After restart, a heavy user can briefly burst up to PER_KEY_LORA_COUNT
# more LoRAs; this is acceptable because MAX_LORA_SIZE_BYTES still caps
# single-file size and PER_KEY_UPLOAD_BYTES_PER_DAY caps aggregate bytes.
_per_key_lora_counts: dict[str, int] = {}


def _incr_key_count(bucket: dict[str, int], api_key: str) -> None:
    if not api_key:
        return
    h = _sha256_key(api_key)
    bucket[h] = bucket.get(h, 0) + 1


def _decr_key_count(bucket: dict[str, int], api_key: str) -> None:
    if not api_key:
        return
    h = _sha256_key(api_key)
    cur = bucket.get(h, 0) - 1
    if cur <= 0:
        bucket.pop(h, None)
    else:
        bucket[h] = cur


def _get_key_count(bucket: dict[str, int], api_key: str) -> int:
    if not api_key:
        return 0
    return bucket.get(_sha256_key(api_key), 0)


# v1.8.2 / SEC P2-3 — rolling 24h upload byte counter. Each bucket is
# sha256(api_key) → list of (epoch_seconds, bytes) tuples pruned on access.
_per_key_upload_bytes: dict[str, list[tuple[float, int]]] = {}


def _upload_window_total(api_key: str) -> int:
    if not api_key:
        return 0
    h = _sha256_key(api_key)
    window = _per_key_upload_bytes.get(h)
    if not window:
        return 0
    cutoff = time.time() - 86400
    pruned = [(t, n) for (t, n) in window if t >= cutoff]
    if pruned:
        _per_key_upload_bytes[h] = pruned
    else:
        _per_key_upload_bytes.pop(h, None)
    return sum(n for _, n in pruned)


def _record_upload_bytes(api_key: str, n_bytes: int) -> None:
    if not api_key or n_bytes <= 0:
        return
    h = _sha256_key(api_key)
    _per_key_upload_bytes.setdefault(h, []).append((time.time(), n_bytes))


def _upload_quota_error(api_key: str, proposed_bytes: int) -> JSONResponse | None:
    """SEC P2-3. Return 429 JSONResponse if adding ``proposed_bytes`` would
    exceed PER_KEY_UPLOAD_BYTES_PER_DAY in the 24h rolling window. None =
    allowed. No-op when auth is off or caller is anonymous (api_key='').
    """
    if not api_key or not config.API_KEYS:
        return None
    cap = config.PER_KEY_UPLOAD_BYTES_PER_DAY
    if cap <= 0:
        return None
    current = _upload_window_total(api_key)
    if current + max(0, proposed_bytes) > cap:
        return JSONResponse(
            status_code=429,
            content={
                "error": "per_key_upload_quota_exceeded",
                "message": f"24h upload quota of {cap // (1024*1024)} MiB reached",
                "detail": f"used={current}, requested={proposed_bytes}, cap={cap}",
            },
            headers={"Retry-After": "3600"},
        )
    return None


# v1.8.2 / SEC P2-7 — Content-Type magic-byte verification. Only applied
# when the client declared a content-type we recognize; octet-stream and
# unknown types are passed through unchanged.
def _infer_media_type_from_magic(first_bytes: bytes) -> str:
    """Infer a canonical MIME type from the first few bytes of a file.

    Used by ``GET /uploads/get/{id}`` (v1.9.0) to set a sensible Content-Type
    on upload reads so browsers (especially Safari) don't reject media.
    Returns ``application/octet-stream`` on unknown/empty input — the browser
    will sniff the stream in most cases.
    """
    b = first_bytes or b""
    if b.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if b.startswith(b"\x89PNG"):
        return "image/png"
    if len(b) >= 12 and b[:4] == b"RIFF":
        if b[8:12] == b"WEBP":
            return "image/webp"
        if b[8:12] == b"WAVE":
            return "audio/wav"
    if b.startswith(b"ID3") or b.startswith(b"\xff\xfb") or b.startswith(b"\xff\xf3") or b.startswith(b"\xff\xf2"):
        return "audio/mpeg"
    if b.startswith(b"fLaC"):
        return "audio/flac"
    if b.startswith(b"OggS"):
        return "audio/ogg"
    if len(b) >= 8 and b[4:8] == b"ftyp":
        return "video/mp4"
    return "application/octet-stream"


def _content_type_matches_magic(declared: str, first_bytes: bytes) -> bool:
    """Return True if ``first_bytes`` (≥ 16 bytes recommended) matches the
    declared content-type's magic signature. Lenient on unrecognized /
    missing / application/octet-stream — those return True.
    """
    declared = (declared or "").lower().split(";")[0].strip()
    if not declared or declared == "application/octet-stream":
        return True
    b = first_bytes or b""
    if declared == "image/jpeg":
        return b.startswith(b"\xff\xd8")
    if declared == "image/png":
        return b.startswith(b"\x89PNG")
    if declared == "image/webp":
        return len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP"
    if declared == "video/mp4":
        return len(b) >= 8 and b[4:8] == b"ftyp"
    if declared == "audio/mpeg":
        return b.startswith(b"ID3") or b.startswith(b"\xff\xfb") or b.startswith(b"\xff\xf3")
    if declared == "audio/wav" or declared == "audio/x-wav":
        return len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE"
    if declared == "audio/flac":
        return b.startswith(b"fLaC")
    if declared == "audio/ogg":
        return b.startswith(b"OggS")
    return True


# v1.8.2 / SEC P2-8 — dedup concurrent _auto_exit_turbo_on_sidecar_failure
# invocations. If the Modal sidecar is flapping, every failed job would
# schedule its own exit task; we only need one.
_exit_turbo_scheduled: bool = False


def _schedule_auto_exit_turbo() -> None:
    """Idempotent scheduler: creates at most one exit task at a time.

    Subsequent sidecar failures while an exit is pending log a WARN and
    return without creating another task. The flag is cleared in the task's
    finally so a future flap can schedule a fresh exit.
    """
    global _exit_turbo_scheduled
    if _exit_turbo_scheduled:
        logger.warning("Turbo auto-exit already scheduled; suppressing duplicate trigger")
        return
    _exit_turbo_scheduled = True

    async def _run_once() -> None:
        global _exit_turbo_scheduled
        try:
            await _auto_exit_turbo_on_sidecar_failure()
        finally:
            _exit_turbo_scheduled = False

    asyncio.create_task(_run_once())


def _require_owner(
    owner_key: str, request: Request, *, sse_token: str | None = None,
) -> JSONResponse | None:
    """Ownership gate for in-memory job/batch endpoints. SEC P0-1 (v1.8.2).

    Returns a 404 JSONResponse when the caller's Bearer (or SSE ``?token=``)
    does not match the resource's ``api_key``. Returns ``None`` — caller may
    proceed — when:

      - ``owner_key`` is empty/falsy (legacy resources or auth-disabled-mode
        submissions — preserves backwards-compat so pre-fix jobs keep working)
      - ``config.API_KEYS`` is empty (auth globally disabled — parity with the
        middleware's bypass)
      - the caller's key equals ``owner_key`` via constant-time compare

    Returns 404 (not 403) on mismatch to avoid an existence oracle: an attacker
    probing random job IDs sees the same 404 shape whether the ID is unknown or
    belongs to another tenant.

    Job/batch endpoints that previously returned 404 for "not found" still do;
    this helper only ADDS a tenancy filter on top.
    """
    if not owner_key or not config.API_KEYS:
        return None
    caller: str | None = None
    if sse_token:
        caller = _resolve_sse_token(sse_token)
    if not caller:
        caller = _extract_api_key(request)
    if not caller or not _secrets.compare_digest(owner_key, caller):
        return _error(404, "Not found")
    return None


# ---------------------------------------------------------------------------
# Batch Endpoints — Phase 3
# ---------------------------------------------------------------------------


@app.post("/v2/batch")
async def v2_batch_submit(body: BatchRequest, request: Request) -> JSONResponse:
    """Submit a batch of generation jobs. Returns 202 with batch_id."""
    if _paused:
        return JSONResponse(
            status_code=503,
            content={"error": "system_paused", "message": "System is paused for maintenance."},
            headers={"Retry-After": "300"},
        )

    auth = request.headers.get("Authorization", "")
    api_key = auth[7:] if auth.startswith("Bearer ") else ""
    # v1.8.2 / SEC P1-3: per-key batch cap BEFORE global cap.
    if api_key and config.API_KEYS and _get_key_count(_per_key_batch_counts, api_key) >= config.PER_KEY_BATCH_CAP:
        return JSONResponse(
            status_code=429,
            content={
                "error": "per_key_queue_full",
                "message": f"You have {config.PER_KEY_BATCH_CAP} batches in flight.",
            },
            headers={"Retry-After": "30"},
        )

    if batch_store.active_count() >= config.MAX_BATCH_QUEUE_DEPTH:
        return JSONResponse(
            status_code=429,
            content={"error": "batch_queue_full", "message": "Batch queue is full. Try again later."},
            headers={"Retry-After": "30"},
        )

    # Validate all items eagerly — parse through corresponding Pydantic model
    for i, item in enumerate(body.items):
        if item.type not in _BATCH_TYPE_MAP:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid item type at index {i}: {item.type}"},
            )
        _, validator_model = _BATCH_TYPE_MAP[item.type]
        try:
            validator_model(**item.params)
        except Exception as exc:
            return JSONResponse(
                status_code=422,
                content={"error": f"Validation failed at item {i} ({item.type}): {exc}"},
            )

    # Swap optimization: sort all images before all videos, group by model within images
    image_items = [it for it in body.items if _is_image_type(it.type)]
    video_items = [it for it in body.items if not _is_image_type(it.type)]
    # Within image items, sort klein before dev to minimize Flux model reloads
    image_items.sort(key=lambda it: it.params.get("model", "flux2-klein"))
    sorted_items = image_items + video_items

    batch = BatchJob(
        id=make_batch_id(),
        items=sorted_items,
        api_key=api_key,
        total=len(sorted_items),
        turbo=_turbo_active,
        priority=body.priority,
        callback_url=body.callback_url,
    )
    _incr_key_count(_per_key_batch_counts, api_key)
    batch_store.add(batch)
    _batch_queue.put_nowait(batch.id)

    return JSONResponse(
        status_code=202,
        content={
            "batch_id": batch.id,
            "status": "queued",
            "total": batch.total,
            "queue_position": max(0, batch_store.active_count() - 1),
        },
    )


@app.get("/v2/batch/{batch_id}")
async def v2_batch_status(batch_id: str, request: Request) -> JSONResponse:
    """Poll batch status + partial results."""
    batch = batch_store.get(batch_id)
    if batch is None:
        return _error(404, "Batch not found")
    deny = _require_owner(batch.api_key, request)
    if deny is not None:
        return deny
    return JSONResponse(content={
        "batch_id": batch.id,
        "status": batch.status,
        "total": batch.total,
        "completed_count": batch.completed_count,
        "failed_count": batch.failed_count,
        "current_index": batch.current_index,
        "turbo": batch.turbo,
        "results": [
            {
                "index": r.index,
                "type": r.type,
                "status": r.status,
                "result_uri": r.result_uri,
                "result_url": f"/v2/batch/{batch.id}/result/{r.index}" if r.status == "completed" and r.result_uri else None,
                "media_type": r.media_type,
                "error": r.error,
                "elapsed_s": r.elapsed_s,
            }
            for r in batch.results
        ],
        "created_at": batch.created_at,
        "started_at": batch.started_at,
        "completed_at": batch.completed_at,
    })


@app.get("/v2/batch/{batch_id}/result/{index}")
async def v2_batch_item_result(batch_id: str, index: int, request: Request) -> Response:
    """Download the result file for a completed batch item."""
    batch = batch_store.get(batch_id)
    if batch is None:
        return _error(404, "Batch not found")
    deny = _require_owner(batch.api_key, request)
    if deny is not None:
        return deny
    for r in batch.results:
        if r.index == index and r.status == "completed" and r.result_uri:
            try:
                path = uploads.resolve(r.result_uri)
                if not path.exists():
                    raise FileNotFoundError()
                return FileResponse(
                    path=str(path),
                    media_type=r.media_type or "application/octet-stream",
                    headers={"Cache-Control": "no-store"},
                )
            except FileNotFoundError:
                return _error(404, "Result file expired or not found")
    return _error(404, "Batch item not found or not completed")


@app.delete("/v2/batch/{batch_id}")
async def v2_batch_cancel(batch_id: str, request: Request) -> JSONResponse:
    """Cancel remaining items in a batch."""
    batch = batch_store.get(batch_id)
    if batch is None:
        return _error(404, "Batch not found")
    deny = _require_owner(batch.api_key, request)
    if deny is not None:
        return deny
    if batch.status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED):
        return JSONResponse(status_code=409, content={
            "error": "batch_already_finished",
            "batch_id": batch.id,
            "status": batch.status,
        })
    cancelled_count = batch.total - batch.completed_count - batch.failed_count
    if batch.status == BatchStatus.PROCESSING:
        # Currently running item will finish, then batch_worker sees cancelled status
        cancelled_count = max(0, cancelled_count - 1)
    batch.status = BatchStatus.CANCELLED
    return JSONResponse(content={
        "batch_id": batch.id,
        "status": "cancelled",
        "completed_count": batch.completed_count,
        "cancelled_count": cancelled_count,
    })


# ---------------------------------------------------------------------------
# Batch worker coroutine
# ---------------------------------------------------------------------------


async def _save_batch_item_history(
    hist: "HistoryStore", job: Job, result_bytes: bytes, storage_uri: str,
) -> None:
    """Fire-and-forget history save for a single batch item."""
    try:
        p = job.params or {}
        await asyncio.to_thread(hist.save,
            job_id=job.id, api_key=job.api_key, job_type=job.type,
            prompt=p.get("prompt", ""), model=p.get("model"),
            width=p.get("width", 0), height=p.get("height", 0),
            turbo=p.get("turbo", False), status=JobStatus.COMPLETED,
            result_uri=storage_uri, result_bytes=result_bytes,
            created_at=time.time(), completed_at=time.time(), error=None,
        )
    except Exception:
        logger.warning("Failed to save batch item %s to history", job.id, exc_info=True)


async def _fire_batch_webhook(batch: BatchJob) -> None:
    """Fire-and-forget webhook with 3 retries."""
    import httpx
    payload = {
        "batch_id": batch.id,
        "status": batch.status,
        "total": batch.total,
        "completed_count": batch.completed_count,
        "failed_count": batch.failed_count,
        "results": [
            {"index": r.index, "type": r.type, "status": r.status,
             "result_uri": r.result_uri, "media_type": r.media_type,
             "error": r.error, "elapsed_s": r.elapsed_s}
            for r in batch.results
        ],
    }
    backoff = [1, 5, 15]
    for attempt, delay in enumerate(backoff):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(batch.callback_url, json=payload)
                if resp.is_success:
                    logger.info("Batch %s webhook delivered on attempt %d", batch.id, attempt + 1)
                    return
        except Exception:
            pass
        if attempt < len(backoff) - 1:
            await asyncio.sleep(delay)
    logger.warning("Batch %s webhook failed after 3 retries to %s", batch.id, batch.callback_url)


async def _process_batch_item(batch, i: int, item, dispatch_fn=None) -> None:
    """Process a single batch item. Used by both turbo (no lock) and normal mode.

    In turbo mode, `dispatch_fn` alternates between _dispatch_job (cuda:0)
    and _dispatch_job_turbo (cuda:1) so 2 items run on different GPUs.
    In normal mode, the caller holds _inference_lock before calling this.
    """
    t0 = time.monotonic()
    try:
        job = _batch_item_to_job(item, batch.api_key)
        if not _turbo_active:
            # Single-GPU: ensure correct tenant before dispatch
            if _is_image_type(item.type):
                await _ensure_flux_ready()
            else:
                await _ensure_ltx_resident()
        fn = dispatch_fn or _dispatch_job
        result_bytes = await fn(job)

        upload_id, storage_uri = uploads.create()
        uploads.save(upload_id, result_bytes)
        elapsed = time.monotonic() - t0

        batch.results.append(BatchItemResult(
            index=i, type=item.type, status="completed",
            result_uri=storage_uri,
            media_type=_JOB_MEDIA_TYPES.get(JobType(item.type), "application/octet-stream"),
            elapsed_s=round(elapsed, 2),
        ))
        batch.completed_count += 1

        if history and batch.api_key:
            asyncio.create_task(_save_batch_item_history(
                history, job, result_bytes, storage_uri))

    except Exception as exc:
        elapsed = time.monotonic() - t0
        batch.results.append(BatchItemResult(
            index=i, type=item.type, status="failed",
            error=str(exc)[:500], elapsed_s=round(elapsed, 2),
        ))
        batch.failed_count += 1
        logger.exception("Batch %s item %d failed", batch.id, i)


async def batch_worker() -> None:
    """Background worker that processes batches from the batch queue.

    In turbo mode: dispatches 2 items concurrently via asyncio.gather
    (one per GPU worker). SplitModelManager._acquire_worker() assigns
    each job to a different GPU.

    In normal mode: dispatches 1 item at a time, holding _inference_lock,
    with swap logic (_ensure_ltx_resident / _ensure_flux_ready).
    """
    logger.info("Batch worker started")

    while True:
        batch_id = await _batch_queue.get()
        batch = batch_store.get(batch_id)
        if batch is None or batch.status == BatchStatus.CANCELLED:
            # v1.8.2 / SEC P1-3: release per-key slot even on pre-dequeue
            # cancel (e.g. via /v1/system/pause).
            if batch is not None:
                _decr_key_count(_per_key_batch_counts, batch.api_key or "")
            _batch_queue.task_done()
            continue

        batch.status = BatchStatus.PROCESSING
        batch.started_at = time.monotonic()

        # Auto-turbo: if cuda:1 has been idle long enough and the batch is
        # large enough to benefit, engage dual-GPU mode automatically.
        # Turbo persists until a JoyAI/ACE request reclaims cuda:1.
        idle_min = _cuda1_idle_seconds() / 60
        if (
            not _turbo_active
            and idle_min >= config.AUTO_TURBO_IDLE_MINUTES
            and len(batch.items) >= 2
        ):
            logger.info(
                "Auto-turbo: cuda:1 idle %.0f min, engaging for batch %s (%d items)",
                idle_min, batch.id, len(batch.items),
            )
            try:
                async with _inference_lock:
                    await _enter_turbo_mode()
            except Exception:
                logger.warning("Auto-turbo entry failed — processing batch in single-GPU mode", exc_info=True)

        logger.info("Processing batch %s (%d items, turbo=%s)", batch.id, batch.total, _turbo_active)

        items = list(enumerate(batch.items))
        idx = 0
        while idx < len(items):
            if batch.status == BatchStatus.CANCELLED:
                for j in range(idx, len(items)):
                    batch.results.append(BatchItemResult(
                        index=items[j][0], type=items[j][1].type,
                        status="cancelled",
                    ))
                break

            if _turbo_active:
                # Dual-GPU / turbo: dispatch 2 items concurrently.
                # In DUAL_GPU_LTX mode, both go through _dispatch_job — the
                # in-process _acquire_worker() routes each to a different GPU.
                # In sidecar turbo, first → _dispatch_job, second → _dispatch_job_turbo.
                chunk = items[idx:idx+2]
                batch.current_index = chunk[0][0]
                dispatchers = [_dispatch_job, _dispatch_job_turbo]
                tasks = [
                    _process_batch_item(batch, i, item, dispatch_fn=dispatchers[ci])
                    for ci, (i, item) in enumerate(chunk)
                ]
                await asyncio.gather(*tasks)
                idx += len(chunk)
            else:
                # Normal: 1 at a time with inference lock
                i, item = items[idx]
                batch.current_index = i
                async with _inference_lock:
                    await _process_batch_item(batch, i, item)
                idx += 1

        # Final status
        batch.completed_at = time.monotonic()
        if batch.status != BatchStatus.CANCELLED:
            if batch.failed_count == 0:
                batch.status = BatchStatus.COMPLETED
            elif batch.completed_count == 0:
                batch.status = BatchStatus.FAILED
            else:
                batch.status = BatchStatus.PARTIAL

        total_elapsed = batch.completed_at - (batch.started_at or batch.created_at)
        logger.info("Batch %s finished: %d/%d completed in %.1fs",
                     batch.id, batch.completed_count, batch.total, total_elapsed)

        if batch.callback_url:
            asyncio.create_task(_fire_batch_webhook(batch))

        # v1.8.2 / SEC P1-3: release per-key batch quota slot.
        _decr_key_count(_per_key_batch_counts, batch.api_key or "")

        _batch_queue.task_done()


async def batch_cleanup_loop() -> None:
    """Periodically remove expired batch metadata."""
    ttl = config.BATCH_RESULT_TTL_SECONDS
    logger.info("Batch cleanup loop started (TTL=%ds)", ttl)
    while True:
        await asyncio.sleep(60)
        now = time.monotonic()
        to_remove: list[str] = []
        for batch in batch_store.all_batches():
            if batch.status in (BatchStatus.COMPLETED, BatchStatus.FAILED,
                                BatchStatus.CANCELLED, BatchStatus.PARTIAL):
                if batch.completed_at and (now - batch.completed_at) > ttl:
                    to_remove.append(batch.id)
        for bid in to_remove:
            batch_store.remove(bid)
        if to_remove:
            logger.info("Cleaned up %d expired batches", len(to_remove))


# ---------------------------------------------------------------------------
# Compositions
# ---------------------------------------------------------------------------

from composition_store import CompositionStore
compositions = CompositionStore()


def _composition_data_from_body(body: dict) -> dict:
    """Split a composition-save body into persisted ``data`` (without ``name``).

    v1.9.5: previously we hardcoded ``{"clips", "transitions"}`` which silently
    dropped any other top-level field (notably ``audio_uri`` for MusicVideo
    mode — reload → re-export → silent MP4). Now pass the whole body through
    minus ``name`` so future frontend additions don't require server changes.
    ``clips`` / ``transitions`` default to empty lists when absent.
    """
    data = {k: v for k, v in body.items() if k != "name"}
    data.setdefault("clips", [])
    data.setdefault("transitions", [])
    return data


@app.post("/v2/compositions")
async def v2_compositions_create(request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    body = await request.json()
    if not isinstance(body, dict):
        return _error(400, "body_must_be_object")
    name = body.get("name", "Untitled")
    data = _composition_data_from_body(body)
    result = compositions.create(api_key, name, data)
    return JSONResponse(content=result, status_code=201)


@app.get("/v2/compositions")
async def v2_compositions_list(request: Request, limit: int = 50, offset: int = 0) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    items = compositions.list(api_key, limit=min(limit, 200), offset=offset)
    return JSONResponse(content=items)


@app.get("/v2/compositions/{comp_id}")
async def v2_compositions_get(comp_id: str, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    item = compositions.get(comp_id, api_key)
    if not item:
        return _error(404, "Composition not found")
    return JSONResponse(content=item)


@app.put("/v2/compositions/{comp_id}")
async def v2_compositions_update(comp_id: str, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    body = await request.json()
    if not isinstance(body, dict):
        return _error(400, "body_must_be_object")
    name = body.get("name", "Untitled")
    data = _composition_data_from_body(body)
    updated = compositions.update(comp_id, api_key, name, data)
    if not updated:
        return _error(404, "Composition not found")
    return JSONResponse(content={"status": "updated"})


@app.delete("/v2/compositions/{comp_id}")
async def v2_compositions_delete(comp_id: str, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    deleted = compositions.delete(comp_id, api_key)
    if not deleted:
        return _error(404, "Composition not found")
    return JSONResponse(content={"status": "deleted"})


@app.post("/v2/compositions/{comp_id}/export")
async def v2_compositions_export(comp_id: str, request: Request) -> JSONResponse:
    api_key = _extract_api_key(request)
    if not api_key:
        return _error(401, "Missing API key")
    comp = compositions.get(comp_id, api_key)
    if not comp:
        return _error(404, "Composition not found")
    # Optional body: {"audio_uri": "storage://<id>"} to overlay audio on export.
    # Body is optional for backwards compat with existing clients that POST empty.
    # Precedence (v1.9.5): request body > stored comp["data"]["audio_uri"].
    # Falling back to the stored value means a reload-then-export produces the
    # same MP4 as the original export (MusicVideo comps persist audio_uri via
    # v1.9.5's _composition_data_from_body).
    audio_uri: str | None = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            raw = body.get("audio_uri")
            if isinstance(raw, str) and raw:
                audio_uri = raw
    except Exception:
        pass  # no body / invalid JSON → fall through to stored lookup
    if audio_uri is None:
        stored = comp["data"].get("audio_uri")
        if isinstance(stored, str) and stored:
            audio_uri = stored
    params = {
        "composition_id": comp_id,
        "clips": comp["data"].get("clips", []),
        "transitions": comp["data"].get("transitions", []),
        "audio_uri": audio_uri,
    }
    return _submit_job(JobType.EXPORT_COMPOSITION, params, request)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )
