"""Tests for v1.17.0-rc2 validator pipeline.

Covers:
  - tier orchestration (all-pass / stub-tier2 / tier3-retake-override / sapiens-unreachable)
  - validator_runs cache hit + miss-on-version-bump
  - POST /v2/video/analyze-motion (happy path / 404 / 401)
  - on_complete dispatch (video / non-video / opt-out / failed jobs)
  - turbo-mode coordination (sapiens-sidecar in stop/restore lists)
  - JUDGE_PROMPT_V1 schema validation
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

import config

# Disable auth + GPU for routing tests, mirror test_madmom_routing.py.
config.GPU_DEVICES = []
config.API_KEYS = set()
config.LOAD_SAPIENS = False  # default off for rc2 ship

from fastapi.testclient import TestClient  # noqa: E402

from job_queue import Job, JobStatus, JobType  # noqa: E402
from history_store import HistoryStore, _hash_key  # noqa: E402
import server as server_mod  # noqa: E402
from server import app  # noqa: E402


client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_video(tmp_path: Path, name: str = "fake.mp4", size: int = 256) -> Path:
    """Drop a synthetic file at tmp_path; tier1 will fail to decode it but
    sha256 + cache logic still work. Tests that exercise tier1 mock it."""
    p = tmp_path / name
    p.write_bytes(b"\x00\x01\x02" * size)
    return p


def _seed_opt_in(history: HistoryStore, api_key: str, opt_in: bool = True) -> None:
    """Insert / update api_key_metadata for the given key."""
    now = time.time()
    history._conn.execute(
        """INSERT OR REPLACE INTO api_key_metadata
           (api_key_hash, training_opt_in, tier, notes, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (_hash_key(api_key), 1 if opt_in else 0, "pro", None, now, now),
    )
    history._conn.commit()


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def test_composite_all_tiers_pass():
    from validator import composite
    tier1 = {"dynamic_degree": 4.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2 = {"pose_temporal_stability": 0.9}
    tier3 = {"verdict": "pass", "score": 0.85, "judge_score": 0.85, "reasoning": "ok", "retake_hint": None}
    out = composite(tier1, tier2, tier3)
    assert out["recommendation"] == "pass"
    assert out["composite_score"] >= 0.65


def test_composite_tier2_stub_treated_as_skipped():
    """sapiens stub → tier2 contributes 0.2*1.0 (no penalty) — composite still computable."""
    from validator import composite
    tier1 = {"dynamic_degree": 4.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2 = {"tier2_skipped": True, "status": "stub"}
    tier3 = {"verdict": "pass", "score": 0.85, "judge_score": 0.85}
    out = composite(tier1, tier2, tier3)
    assert "tier2_stub" in out["reasoning_summary"]
    # 0.4 * (4/5=0.8) + 0.2 * 1.0 + 0.4 * 0.85 = 0.32 + 0.2 + 0.34 = 0.86
    assert out["composite_score"] > 0.65
    assert out["recommendation"] == "pass"


def test_composite_tier3_retake_overrides():
    """tier3.verdict=retake forces final recommendation=retake regardless of score."""
    from validator import composite
    tier1 = {"dynamic_degree": 4.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2 = {"pose_temporal_stability": 0.9}
    tier3 = {"verdict": "retake", "score": 0.9, "judge_score": 0.9}
    out = composite(tier1, tier2, tier3)
    # composite numerically high, but verdict="retake" overrides.
    assert out["recommendation"] == "retake"


def test_composite_sapiens_unreachable_degrades():
    """tier2=None → composite uses 0.4*tier1 + 0.4*tier3 + 0.2*1.0 (no penalty)."""
    from validator import composite
    tier1 = {"dynamic_degree": 5.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2 = None
    tier3 = {"verdict": "pass", "score": 0.8, "judge_score": 0.8}
    out = composite(tier1, tier2, tier3)
    assert "tier2_unreachable" in out["reasoning_summary"]
    assert out["composite_score"] > 0.65


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


def test_validator_runs_cache_hit(tmp_path):
    """Second call with same (video_sha256, validator_version) → cached payload, no tier work."""
    from validator import run_all_tiers
    db = tmp_path / "history.db"
    history = HistoryStore(db_path=db)
    video = _make_fake_video(tmp_path, "v.mp4")

    call_count = {"raft": 0, "sapiens": 0, "gemma": 0}

    async def fake_raft(video_path):
        call_count["raft"] += 1
        return {"dynamic_degree": 3.0, "flow_windows": [0.5, 0.5, 0.5, 0.5], "motion_smoothness": 0.8, "latency_s": 0.01}

    async def fake_sapiens(video_path):
        call_count["sapiens"] += 1
        return {"tier2_skipped": True, "status": "stub", "latency_s": 0.0}

    async def fake_judge(chat, video_path, prompt, t1, t2):
        call_count["gemma"] += 1
        return {"verdict": "pass", "score": 0.8, "judge_score": 0.8, "reasoning": "x", "retake_hint": None, "latency_s": 0.01}

    with patch("validator._run_tier1_raft", fake_raft), \
         patch("validator._run_tier2_sapiens", fake_sapiens), \
         patch("validator._run_tier3_judge", fake_judge):
        async def go():
            return await run_all_tiers(
                video_uri=f"storage://aaa", video_path=str(video),
                prompt="x", chat=None, history=history,
                validator_version="1.17.0-rc2",
            )
        first = asyncio.run(go())
        second = asyncio.run(go())

    assert first["cached"] is False
    assert second["cached"] is True
    # LOAD_SAPIENS=False at module-load time → tier2 short-circuits with a
    # synthetic stub-skipped payload, so the sapiens fake is never invoked.
    # Tier 1 + Tier 3 fire once on the miss, then never again on the hit.
    assert call_count["raft"] == 1
    assert call_count["gemma"] == 1
    assert call_count["sapiens"] == 0


def test_validator_runs_cache_miss_on_version_bump(tmp_path):
    """Same video, different validator_version → re-runs all tiers."""
    from validator import run_all_tiers
    db = tmp_path / "history.db"
    history = HistoryStore(db_path=db)
    video = _make_fake_video(tmp_path, "v.mp4")

    call_count = {"raft": 0}

    async def fake_raft(video_path):
        call_count["raft"] += 1
        return {"dynamic_degree": 3.0, "flow_windows": [0.5] * 4, "motion_smoothness": 0.8, "latency_s": 0.01}

    async def fake_sapiens(video_path):
        return {"tier2_skipped": True, "status": "stub", "latency_s": 0.0}

    async def fake_judge(chat, video_path, prompt, t1, t2):
        return {"verdict": "pass", "score": 0.8, "judge_score": 0.8, "reasoning": "x", "retake_hint": None, "latency_s": 0.01}

    with patch("validator._run_tier1_raft", fake_raft), \
         patch("validator._run_tier2_sapiens", fake_sapiens), \
         patch("validator._run_tier3_judge", fake_judge):
        async def go(version):
            return await run_all_tiers(
                video_uri="storage://bbb", video_path=str(video),
                prompt="x", chat=None, history=history,
                validator_version=version,
            )
        asyncio.run(go("1.17.0-rc2"))
        asyncio.run(go("1.17.0-rc3"))  # bump

    assert call_count["raft"] == 2  # re-ran on version bump


# ---------------------------------------------------------------------------
# POST /v2/video/analyze-motion
# ---------------------------------------------------------------------------


def test_analyze_motion_endpoint_happy_path(tmp_path, monkeypatch):
    """Full HTTP roundtrip: 200 with composite payload."""
    from server import uploads as srv_uploads, history as srv_history

    upload_id, storage_uri = srv_uploads.create()
    srv_uploads.save(upload_id, b"\x00" * 64)

    async def fake_run_all(**kwargs):
        return {
            "video_uri": kwargs["video_uri"],
            "validator_version": "1.17.0-rc2",
            "tier1": {"dynamic_degree": 4.0, "flow_windows": [1] * 4, "motion_smoothness": 0.9},
            "tier2": {"tier2_skipped": True},
            "tier3": {"verdict": "pass", "score": 0.85, "judge_score": 0.85},
            "composite_score": 0.85,
            "recommendation": "pass",
            "reasoning_summary": "tier2_stub",
            "ran_at": time.time(),
            "latency_s": 0.1,
            "cached": False,
        }

    monkeypatch.setattr("validator.run_all_tiers", fake_run_all)
    resp = client.post("/v2/video/analyze-motion", json={
        "video_uri": storage_uri, "prompt": "a sunset",
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["recommendation"] == "pass"
    assert data["composite_score"] == pytest.approx(0.85)


def test_analyze_motion_endpoint_404_unknown_uri():
    # Use a unique UUID guaranteed not to exist on disk.
    import uuid as _uuid
    fake_uri = f"storage://{_uuid.uuid4().hex}"
    resp = client.post("/v2/video/analyze-motion", json={
        "video_uri": fake_uri, "prompt": "x",
    })
    assert resp.status_code == 404


def test_analyze_motion_endpoint_auth_required(monkeypatch):
    """Auth on → missing bearer → 401."""
    from server import uploads as srv_uploads
    upload_id, storage_uri = srv_uploads.create()
    srv_uploads.save(upload_id, b"\x00" * 16)

    monkeypatch.setattr(config, "API_KEYS", {"secret"})
    try:
        resp = client.post("/v2/video/analyze-motion", json={
            "video_uri": storage_uri, "prompt": "x",
        })
        assert resp.status_code == 401
    finally:
        monkeypatch.setattr(config, "API_KEYS", set())


# ---------------------------------------------------------------------------
# on_complete dispatch
# ---------------------------------------------------------------------------


def test_on_complete_dispatches_validator_for_video_jobs(tmp_path, monkeypatch):
    """Completed video job + opt-in → validator dispatch fired."""
    from server import history as srv_history
    api_key = "test_optin_video"
    _seed_opt_in(srv_history, api_key, opt_in=True)

    job = Job(id="j_video_done", type=JobType.TEXT_TO_VIDEO, status=JobStatus.COMPLETED,
              api_key=api_key, result_uri="storage://" + "a" * 32)

    fired = {"called": False}

    async def fake_dispatch(j):
        fired["called"] = True

    monkeypatch.setattr(server_mod, "_dispatch_validator", fake_dispatch)

    async def run_in_loop():
        server_mod._on_job_complete(job)
        # Yield once so the asyncio.create_task fires.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run_in_loop())
    assert fired["called"]


def test_on_complete_skips_non_video_jobs(monkeypatch):
    """Image jobs are not validated."""
    from server import history as srv_history
    api_key = "test_optin_img"
    _seed_opt_in(srv_history, api_key, opt_in=True)

    job = Job(id="j_img_done", type=JobType.TEXT_TO_IMAGE, status=JobStatus.COMPLETED,
              api_key=api_key, result_uri="storage://" + "b" * 32)

    fired = {"called": False}

    async def fake_dispatch(j):
        fired["called"] = True

    monkeypatch.setattr(server_mod, "_dispatch_validator", fake_dispatch)

    async def run_in_loop():
        server_mod._on_job_complete(job)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run_in_loop())
    assert not fired["called"]


def test_on_complete_skips_when_training_opt_in_zero(monkeypatch):
    """training_opt_in=0 → no dispatch."""
    from server import history as srv_history
    api_key = "test_optout_video"
    _seed_opt_in(srv_history, api_key, opt_in=False)

    job = Job(id="j_optout_done", type=JobType.AUDIO_TO_VIDEO, status=JobStatus.COMPLETED,
              api_key=api_key, result_uri="storage://" + "c" * 32)

    fired = {"called": False}

    async def fake_dispatch(j):
        fired["called"] = True

    monkeypatch.setattr(server_mod, "_dispatch_validator", fake_dispatch)

    async def run_in_loop():
        server_mod._on_job_complete(job)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run_in_loop())
    assert not fired["called"]


def test_on_complete_failed_jobs_skip_dispatch(monkeypatch):
    """Failed jobs are not validated."""
    from server import history as srv_history
    api_key = "test_optin_failed"
    _seed_opt_in(srv_history, api_key, opt_in=True)

    job = Job(id="j_failed", type=JobType.TEXT_TO_VIDEO, status=JobStatus.FAILED,
              api_key=api_key, result_uri=None, error="oops")

    fired = {"called": False}

    async def fake_dispatch(j):
        fired["called"] = True

    monkeypatch.setattr(server_mod, "_dispatch_validator", fake_dispatch)

    async def run_in_loop():
        server_mod._on_job_complete(job)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run_in_loop())
    assert not fired["called"]


def test_on_complete_unknown_key_skips_dispatch(monkeypatch):
    """Unknown api_key (no metadata row) → default opt-out → no dispatch."""
    job = Job(id="j_unknown_key", type=JobType.TEXT_TO_VIDEO, status=JobStatus.COMPLETED,
              api_key="never_seeded_key", result_uri="storage://" + "d" * 32)

    fired = {"called": False}

    async def fake_dispatch(j):
        fired["called"] = True

    monkeypatch.setattr(server_mod, "_dispatch_validator", fake_dispatch)

    async def run_in_loop():
        server_mod._on_job_complete(job)
        await asyncio.sleep(0)

    asyncio.run(run_in_loop())
    assert not fired["called"]


# ---------------------------------------------------------------------------
# Turbo coordination
# ---------------------------------------------------------------------------


def test_turbo_entry_stops_sapiens_sidecar(monkeypatch):
    """`_stop_cuda1_tenants` must include sapiens-sidecar when LOAD_SAPIENS=1."""
    monkeypatch.setattr(config, "LOAD_SAPIENS", True)
    monkeypatch.setattr(config, "LOAD_ACE", True)
    monkeypatch.setattr(config, "LOAD_JOYAI", False)
    monkeypatch.setattr(config, "LOAD_ERNIE", False)

    stopped: list[str] = []

    async def fake_systemctl(unit, action):
        stopped.append((unit, action))

    monkeypatch.setattr(server_mod, "_systemctl_unit", fake_systemctl)

    asyncio.run(server_mod._stop_cuda1_tenants())
    units = [u for u, _ in stopped]
    assert "sapiens-sidecar" in units
    assert "ltx-sidecar" in units


def test_turbo_exit_restores_sapiens_sidecar(monkeypatch):
    """`_restore_cuda1_tenants` must restart sapiens-sidecar when LOAD_SAPIENS=1."""
    monkeypatch.setattr(config, "LOAD_SAPIENS", True)
    monkeypatch.setattr(config, "LOAD_ACE", False)
    monkeypatch.setattr(config, "LOAD_JOYAI", False)
    monkeypatch.setattr(config, "LOAD_ERNIE", False)

    started: list[str] = []

    async def fake_systemctl(unit, action):
        started.append((unit, action))

    monkeypatch.setattr(server_mod, "_systemctl_unit", fake_systemctl)
    asyncio.run(server_mod._restore_cuda1_tenants())
    units = [u for u, _ in started]
    assert "sapiens-sidecar" in units


def test_turbo_skips_sapiens_when_load_sapiens_off(monkeypatch):
    """LOAD_SAPIENS=0 → don't try to start/stop the unit at all in restore."""
    monkeypatch.setattr(config, "LOAD_SAPIENS", False)
    monkeypatch.setattr(config, "LOAD_ACE", False)
    monkeypatch.setattr(config, "LOAD_JOYAI", False)
    monkeypatch.setattr(config, "LOAD_ERNIE", False)

    started: list[str] = []

    async def fake_systemctl(unit, action):
        started.append((unit, action))

    monkeypatch.setattr(server_mod, "_systemctl_unit", fake_systemctl)
    asyncio.run(server_mod._restore_cuda1_tenants())
    units = [u for u, _ in started]
    assert "sapiens-sidecar" not in units


# ---------------------------------------------------------------------------
# JUDGE_PROMPT_V1 schema
# ---------------------------------------------------------------------------


def test_judge_prompt_v1_schema_validates():
    """Sample Gemma response → Pydantic validates correctly."""
    from validator import JudgeResponseV1
    sample = json.dumps({
        "verdict": "pass",
        "score": 0.78,
        "reasoning": "motion matches prompt; smooth pan throughout",
        "retake_hint": None,
    })
    r = JudgeResponseV1.model_validate_json(sample)
    assert r.verdict == "pass"
    assert r.score == pytest.approx(0.78)
    assert r.retake_hint is None


def test_judge_prompt_v1_schema_rejects_invalid_verdict():
    from validator import JudgeResponseV1
    with pytest.raises(Exception):  # pydantic ValidationError
        JudgeResponseV1.model_validate_json('{"verdict": "skip", "score": 0.5, "reasoning": ""}')


def test_judge_prompt_v1_schema_rejects_score_out_of_range():
    from validator import JudgeResponseV1
    with pytest.raises(Exception):
        JudgeResponseV1.model_validate_json('{"verdict": "pass", "score": 1.5, "reasoning": ""}')
