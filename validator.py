"""Validator pipeline (v1.17.0-rc2) — tier orchestration + composite scoring.

Three tiers, each independently degradable:

  - **Tier 1 (RAFT)**: optical-flow magnitude + smoothness, in-process on
    cuda:0 (lazy-loaded `raft_small` from torchvision). Evicted after each
    call so LTX can reclaim cuda:0 for the next video request.
  - **Tier 2 (Sapiens)**: pose temporal stability via the sidecar at
    :8096. Stub-tolerant: a `{"stub": true}` response is treated as
    skipped (not failed); composite scoring uses 1.0 in that slot.
  - **Tier 3 (Gemma judge)**: vision LLM verdict via the existing
    `chat_manager` proxy (CHAR_VISION_MODEL, mirrors `/v2/char/rank`).
    Strict-JSON output, Pydantic-validated.

Caching: `validator_runs` table keyed by `(video_sha256, validator_version)`.
A version bump forces re-runs; same-version hits return the cached row.

Composite recommendation:
  composite = 0.4·tier1_dynamic + 0.2·tier2_stability + 0.4·tier3_score
  pass     iff composite ≥ 0.65
  warn     iff 0.45 ≤ composite < 0.65
  retake   iff composite < 0.45  OR  tier3.verdict == "retake"

When tier2 is skipped (stub) or unreachable, the tier2 slot contributes
0.2·1.0 (= no penalty), preserving the same composite arithmetic. Tier1
and tier3 are required — failure of either makes the validator return
None for the offending tier and the composite collapses to a partial
score with a clear `tier1_failed` / `tier3_failed` flag.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

import config
from history_store import HistoryStore, _extract_frames_as_pils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier-3 judge response schema
# ---------------------------------------------------------------------------


class JudgeResponseV1(BaseModel):
    """Schema for Gemma judge responses (config.JUDGE_PROMPT_V1).

    Mirrors the v1.8.2 / SEC P1-5 pattern from CharRankResponse — the LLM's
    output is validated against this before the validator trusts it.
    """

    verdict: Literal["pass", "warn", "retake"]
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=4000)
    retake_hint: str | None = None


# ---------------------------------------------------------------------------
# Tier 1 — RAFT optical flow (in-process, cuda:0)
# ---------------------------------------------------------------------------

# RAFT model is loaded lazily on the first tier-1 call and evicted after each
# call (LTX reclaims cuda:0 between requests). raft_small chosen over
# raft_large because the validator runs on every completed video; the +1 GB
# transient VRAM and ~50 ms/frame difference matter when 28 jobs are in
# flight. Documented choice — flip to raft_large if accuracy regresses.
_RAFT_DEVICE = "cuda:0"
_RAFT_FLOW_TARGET_WIDTH = 256


def _decode_video_frames_for_flow(video_path: str, target_width: int = _RAFT_FLOW_TARGET_WIDTH) -> Any:
    """Decode video frames at 24fps, downsample to target_width, return tensor.

    Returns a uint8 tensor of shape (T, 3, H, W) on CPU, ready to ship to
    the GPU in chunks. PyAV is already a dep via history_store.
    """
    import av
    import torch

    frames = []
    h_target = None
    with av.open(video_path, mode="r") as container:
        if not container.streams.video:
            raise ValueError("no_video_stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")  # (H, W, 3)
            if h_target is None:
                h, w = arr.shape[:2]
                ratio = target_width / max(w, 1)
                h_target = max(1, int(h * ratio))
            # Lazy resize via PIL — avoids a hard dep on cv2 / torchvision.transforms.
            from PIL import Image
            img = Image.fromarray(arr).resize((target_width, h_target), Image.BILINEAR)
            import numpy as np
            arr = np.asarray(img)  # (h_target, target_width, 3)
            frames.append(arr)
    if len(frames) < 2:
        raise ValueError("insufficient_frames_for_flow")
    import numpy as np
    stacked = np.stack(frames, axis=0)  # (T, H, W, 3)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)
    return tensor


async def _run_tier1_raft(video_path: str) -> dict:
    """RAFT optical flow → (dynamic_degree, flow_windows[4], motion_smoothness).

    Runs on cuda:0 in `torch.inference_mode()`. After completion, the model
    is dropped + cuda cache emptied so LTX's next request gets a clean
    allocator. Pure-CPU fallback isn't implemented — RAFT on CPU is
    >30s/clip; we'd rather degrade tier1 to None and let composite collapse.
    """
    import torch
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()

    def _compute() -> dict:
        device = torch.device(_RAFT_DEVICE if torch.cuda.is_available() else "cpu")
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False).to(device).eval()
        try:
            transforms = weights.transforms()
            frames = _decode_video_frames_for_flow(video_path)
            T = frames.shape[0]
            if T < 2:
                raise ValueError("insufficient_frames_for_flow")
            mean_flow_per_pair: list[float] = []
            with torch.inference_mode():
                for i in range(T - 1):
                    pair_a = frames[i:i + 1].to(device)
                    pair_b = frames[i + 1:i + 2].to(device)
                    a, b = transforms(pair_a, pair_b)
                    flow_predictions = model(a, b)
                    flow = flow_predictions[-1]  # (1, 2, H, W) — final refinement
                    mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
                    mean_flow_per_pair.append(float(mag.mean().item()))
            import numpy as np
            arr = np.asarray(mean_flow_per_pair, dtype=np.float64)
            # dynamic_degree: top-5% percentile of mean flow magnitudes.
            dynamic_degree = float(np.percentile(arr, 95)) if arr.size else 0.0
            # flow_windows: 4-window mean across time.
            windows = np.array_split(arr, 4)
            flow_windows = [float(w.mean()) if len(w) else 0.0 for w in windows]
            # motion_smoothness: 1 / (1 + var(diff(arr))).
            if arr.size >= 3:
                accel = np.diff(arr, n=1)
                smoothness = 1.0 / (1.0 + float(np.var(accel)))
            else:
                smoothness = 1.0
            return {
                "dynamic_degree": dynamic_degree,
                "flow_windows": flow_windows,
                "motion_smoothness": smoothness,
            }
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = await loop.run_in_executor(None, _compute)
    payload["latency_s"] = time.perf_counter() - t0
    return payload


# ---------------------------------------------------------------------------
# Tier 2 — Sapiens (sidecar)
# ---------------------------------------------------------------------------


async def _run_tier2_sapiens(video_path: str) -> dict | None:
    """Pose temporal-stability via the sapiens sidecar.

    Returns:
      - `{"status": "stub", "tier2_skipped": true, ...}` if the sidecar
        responds with `stub: true` (rc1 stub-mode) — composite scoring
        treats this as "no penalty, no boost".
      - A real payload with `pose_temporal_variance` / `human_detected` /
        `identity_drift_frames` once rc2 sidecar inference lands.
      - `None` on any sidecar error (unreachable, 5xx, timeout). The
        validator degrades gracefully and the composite drops the tier2
        weight.
    """
    from sapiens_client import get_sapiens_client, SapiensError

    t0 = time.perf_counter()
    try:
        client = get_sapiens_client()
        resp = await client.analyze_pose(video_path)
    except SapiensError as exc:
        logger.warning("tier2 sapiens unreachable/failed: %s", exc)
        return None
    except Exception:
        logger.warning("tier2 sapiens unexpected error", exc_info=True)
        return None

    dt = time.perf_counter() - t0
    if isinstance(resp, dict) and resp.get("stub"):
        return {
            "status": "stub",
            "tier2_skipped": True,
            "raw": resp,
            "latency_s": dt,
        }

    out = dict(resp) if isinstance(resp, dict) else {"raw": resp}
    out["latency_s"] = dt
    return out


# ---------------------------------------------------------------------------
# Tier 3 — Gemma judge (existing chat_manager)
# ---------------------------------------------------------------------------


def _sample_keyframe_indices(num_frames: int, count: int = 5) -> list[int]:
    """Pick first/middle/last + 2 random middle positions, capped to count."""
    if num_frames <= 0:
        return []
    if num_frames <= count:
        return list(range(num_frames))
    indices = {0, num_frames // 2, num_frames - 1}
    # Two random middle positions for the remaining slots, deterministic per
    # frame count (avoid spurious cache misses across repeat runs).
    if count > 3 and num_frames > 4:
        q1 = num_frames // 4
        q3 = (3 * num_frames) // 4
        indices.add(q1)
        indices.add(q3)
    return sorted(indices)[:count]


async def _run_tier3_judge(
    chat: Any,
    video_path: str,
    prompt: str,
    tier1_summary: dict,
    tier2_summary: dict | None,
) -> dict:
    """Sample keyframes, send to Gemma judge, parse + validate JSON.

    Mirrors `/v2/char/rank` (server.py:4307-4363) — system prompt is
    `config.JUDGE_PROMPT_V1`, response schema is `JudgeResponseV1`.

    Returns either:
      - `{"verdict": ..., "score": ..., "reasoning": ..., "retake_hint": ...,
         "latency_s": ..., "judge_score": ...}` on success.
      - `{"error": "...", "judge_score": 0.5, "verdict": "warn", ...}` on
        any failure (LLM unreachable, schema violation, parse error). Never
        raises — the validator must produce a composite even when tier3
        misbehaves.
    """
    import av

    t0 = time.perf_counter()
    # Count frames cheap-ish via PyAV.
    try:
        with av.open(video_path, mode="r") as container:
            stream = container.streams.video[0] if container.streams.video else None
            num_frames = int(stream.frames or 0) if stream else 0
            if num_frames <= 0 and stream:
                # Fall back to estimating via duration * rate — frames=0 is
                # common on streamed MP4s.
                rate = float(stream.average_rate or 24)
                duration = float(stream.duration or 0) * float(stream.time_base or 1)
                num_frames = max(1, int(rate * duration))
    except Exception as exc:
        logger.warning("tier3 frame-count probe failed: %s", exc)
        return {
            "error": f"frame_count_failed: {exc}",
            "judge_score": 0.5,
            "verdict": "warn",
            "score": 0.5,
            "reasoning": "frame-count probe failed; tier3 inconclusive",
            "retake_hint": None,
            "latency_s": time.perf_counter() - t0,
        }

    indices = _sample_keyframe_indices(num_frames, count=5)
    if not indices:
        return {
            "error": "no_frames",
            "judge_score": 0.5,
            "verdict": "warn",
            "score": 0.5,
            "reasoning": "no frames decoded",
            "retake_hint": None,
            "latency_s": time.perf_counter() - t0,
        }

    try:
        video_bytes = Path(video_path).read_bytes()
        pils = await asyncio.to_thread(_extract_frames_as_pils, video_bytes, indices)
    except Exception as exc:
        logger.warning("tier3 keyframe extract failed: %s", exc)
        return {
            "error": f"keyframe_extract_failed: {exc}",
            "judge_score": 0.5,
            "verdict": "warn",
            "score": 0.5,
            "reasoning": "keyframe extraction failed",
            "retake_hint": None,
            "latency_s": time.perf_counter() - t0,
        }

    import io
    image_blocks = []
    for pil in pils:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_blocks.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    summary_text = (
        f'Original prompt: "{prompt}"\n\n'
        f"Tier1 (optical flow):\n"
        f"  dynamic_degree={tier1_summary.get('dynamic_degree'):.3f}\n"
        f"  flow_windows={tier1_summary.get('flow_windows')}\n"
        f"  motion_smoothness={tier1_summary.get('motion_smoothness'):.3f}\n"
        f"Tier2 (pose stability): "
        f"{'skipped' if (tier2_summary is None or tier2_summary.get('tier2_skipped')) else json.dumps(tier2_summary)}\n\n"
        f"Below: 3-5 keyframes from the clip in temporal order. Judge whether "
        f"the clip's motion matches the prompt's intent."
    )

    messages = [
        {"role": "system", "content": config.JUDGE_PROMPT_V1},
        {
            "role": "user",
            "content": [{"type": "text", "text": summary_text}] + image_blocks,
        },
    ]

    try:
        result = await chat.generate_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            model=config.CHAR_VISION_MODEL,
        )
        text = result["choices"][0]["message"]["content"]
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            raise ValueError("no_json_found")
        validated = JudgeResponseV1.model_validate_json(json_match.group())
    except (ValidationError, ValueError, KeyError) as exc:
        logger.warning("tier3 judge schema/parse failure: %s", exc)
        return {
            "error": f"schema_violation: {exc}",
            "judge_score": 0.5,
            "verdict": "warn",
            "score": 0.5,
            "reasoning": "judge response failed schema validation",
            "retake_hint": None,
            "latency_s": time.perf_counter() - t0,
        }
    except Exception as exc:
        logger.warning("tier3 judge call failed: %s", exc)
        return {
            "error": f"judge_call_failed: {exc}",
            "judge_score": 0.5,
            "verdict": "warn",
            "score": 0.5,
            "reasoning": "judge call failed",
            "retake_hint": None,
            "latency_s": time.perf_counter() - t0,
        }

    out = validated.model_dump()
    out["judge_score"] = float(validated.score)
    out["latency_s"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def _normalize_dynamic_degree(dyn: float) -> float:
    """Map RAFT dynamic_degree to [0, 1]. Empirically, healthy clips sit in
    the 1-10 range at width=256; we saturate at 5 for the normalized bound."""
    if dyn is None:
        return 0.0
    return max(0.0, min(1.0, float(dyn) / 5.0))


def composite(
    tier1: dict | None,
    tier2: dict | None,
    tier3: dict | None,
) -> dict:
    """Compute the composite score + recommendation.

    Weights: 0.4 tier1 + 0.2 tier2 + 0.4 tier3. When tier2 is skipped/None,
    its slot contributes 0.2·1.0 (= no penalty) so the composite stays
    additive. tier3.verdict="retake" force-overrides the recommendation
    regardless of numeric composite.
    """
    parts = []
    notes: list[str] = []
    if tier1 is not None:
        t1 = _normalize_dynamic_degree(tier1.get("dynamic_degree", 0.0))
        parts.append(("tier1", 0.4, t1))
    else:
        notes.append("tier1_failed")
    # Tier2: stub / None → 1.0 contribution; real → use pose_temporal_stability
    # if present, else compute from variance (1 / (1 + var)).
    if tier2 is None:
        notes.append("tier2_unreachable")
        t2 = 1.0
    elif tier2.get("tier2_skipped"):
        notes.append("tier2_stub")
        t2 = 1.0
    else:
        if "pose_temporal_stability" in tier2:
            t2 = max(0.0, min(1.0, float(tier2["pose_temporal_stability"])))
        elif "pose_temporal_variance" in tier2:
            v = float(tier2["pose_temporal_variance"])
            t2 = 1.0 / (1.0 + max(0.0, v))
        else:
            t2 = 1.0
    parts.append(("tier2", 0.2, t2))
    if tier3 is not None:
        t3 = max(0.0, min(1.0, float(tier3.get("judge_score", tier3.get("score", 0.5)))))
        parts.append(("tier3", 0.4, t3))
    else:
        notes.append("tier3_failed")

    weight_total = sum(w for _, w, _ in parts) or 1.0
    raw = sum(w * v for _, w, v in parts)
    score = raw  # weights already sum to 1.0 in the happy path.
    if weight_total < 1.0:
        # Partial composite — rescale so a missing tier doesn't artificially
        # tank the score. Caller can read `notes` to see which tier dropped.
        score = raw / weight_total

    if tier3 is not None and tier3.get("verdict") == "retake":
        recommendation = "retake"
    elif score >= 0.65:
        recommendation = "pass"
    elif score >= 0.45:
        recommendation = "warn"
    else:
        recommendation = "retake"

    return {
        "composite_score": float(score),
        "recommendation": recommendation,
        "reasoning_summary": ", ".join(notes) if notes else "all_tiers_ok",
    }


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _video_sha256(path: str) -> str:
    """SHA-256 of the video file. Uses a streaming read to avoid loading
    arbitrary-sized clips into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validator_runs_lookup(
    history: HistoryStore, video_sha256: str, validator_version: str
) -> dict | None:
    """Return the cached payload dict if a row exists, else None."""
    row = history._conn.execute(
        """SELECT payload_json FROM validator_runs
           WHERE video_sha256 = ? AND validator_version = ?
           LIMIT 1""",
        (video_sha256, validator_version),
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["payload_json"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _validator_runs_persist(
    history: HistoryStore,
    *,
    video_uri: str,
    video_sha256: str,
    validator_version: str,
    payload: dict,
    latency_s: float,
) -> str:
    """Insert (or replace via UNIQUE conflict) a `validator_runs` row.

    Returns the run_id. Best-effort — DB errors log + bubble to the caller's
    try/except. The UNIQUE index on (video_sha256, validator_version)
    means concurrent calls for the same (video, version) are safe — last
    write wins via INSERT OR REPLACE.
    """
    run_id = uuid.uuid4().hex
    history._conn.execute(
        """INSERT OR REPLACE INTO validator_runs
           (run_id, video_uri, video_sha256, payload_json, latency_s,
            validator_version, ran_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (run_id, video_uri, video_sha256, json.dumps(payload), latency_s,
         validator_version, time.time()),
    )
    history._conn.commit()
    return run_id


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def run_all_tiers(
    *,
    video_uri: str,
    video_path: str,
    prompt: str,
    chat: Any,
    history: HistoryStore,
    validator_version: str | None = None,
    tiers: list[str] | None = None,
) -> dict:
    """Run the full validator pipeline + persist into validator_runs.

    Returns a payload dict with shape:
      {
        "video_uri": ...,
        "validator_version": ...,
        "tier1": {...} | None,
        "tier2": {...} | None,
        "tier3": {...} | None,
        "composite_score": float,
        "recommendation": "pass" | "warn" | "retake",
        "reasoning_summary": str,
        "ran_at": float,
        "cached": bool,
      }

    Cache hits (same video_sha256 + validator_version) skip all tier work
    and return the persisted payload with `cached: true` injected.
    """
    version = validator_version or config.VALIDATOR_VERSION
    tier_set = set(tiers) if tiers else {"raft", "sapiens", "gemma"}

    # Cache check
    video_sha = await asyncio.to_thread(_video_sha256, video_path)
    cached = _validator_runs_lookup(history, video_sha, version)
    if cached is not None:
        cached["cached"] = True
        return cached

    overall_t0 = time.perf_counter()

    # Tier 1 — RAFT
    tier1: dict | None = None
    if "raft" in tier_set:
        try:
            tier1 = await _run_tier1_raft(video_path)
        except Exception as exc:
            logger.warning("tier1 raft failed: %s", exc, exc_info=True)
            tier1 = None

    # Tier 2 — Sapiens
    tier2: dict | None = None
    if "sapiens" in tier_set and config.LOAD_SAPIENS:
        tier2 = await _run_tier2_sapiens(video_path)
    elif "sapiens" in tier_set:
        # LOAD_SAPIENS=0 → tier2 is treated as stub-skipped, not failed.
        tier2 = {"tier2_skipped": True, "status": "load_disabled", "latency_s": 0.0}

    # Tier 3 — Gemma judge
    tier3: dict | None = None
    if "gemma" in tier_set:
        try:
            tier3 = await _run_tier3_judge(chat, video_path, prompt, tier1 or {}, tier2)
        except Exception as exc:
            logger.warning("tier3 judge failed: %s", exc, exc_info=True)
            tier3 = None

    comp = composite(tier1, tier2, tier3)
    overall_latency = time.perf_counter() - overall_t0

    payload = {
        "video_uri": video_uri,
        "validator_version": version,
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "composite_score": comp["composite_score"],
        "recommendation": comp["recommendation"],
        "reasoning_summary": comp["reasoning_summary"],
        "ran_at": time.time(),
        "latency_s": overall_latency,
        "cached": False,
    }

    try:
        _validator_runs_persist(
            history,
            video_uri=video_uri,
            video_sha256=video_sha,
            validator_version=version,
            payload=payload,
            latency_s=overall_latency,
        )
    except Exception:
        logger.warning("validator_runs persist failed", exc_info=True)

    return payload
