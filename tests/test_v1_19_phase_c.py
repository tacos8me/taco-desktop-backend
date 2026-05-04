"""Tests for v1.18.0-rc5 Stream A plumbing fixes.

Coverage:

  - schema v6 migration: ``generations.ab_arm`` column exists, nullable.
  - ab_arm round-trip: video v2 endpoint accepts ``_ab_arm`` in body,
    ``history.save`` persists it on the row.
  - ab_decision: now reads from ``generations.ab_arm`` (column-based
    cohort lookup) instead of inferring from ``lora_applied_id``.
  - motion_intent passive-dispatch: ``_dispatch_validator`` looks up
    ``generations.motion_intent`` and forwards to ``run_all_tiers``.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import config

config.GPU_DEVICES = []
config.API_KEYS = set()

from history_store import HistoryStore, _hash_key  # noqa: E402
from scripts import ab_decision as abd  # noqa: E402


@pytest.fixture()
def fresh_db(tmp_path: Path) -> HistoryStore:
    db = tmp_path / "history.db"
    return HistoryStore(db_path=db)


# ---------------------------------------------------------------------------
# Schema v6 — ab_arm column
# ---------------------------------------------------------------------------


def test_schema_v6_ab_arm_column_exists(fresh_db: HistoryStore) -> None:
    cols = {row[1] for row in fresh_db._conn.execute("PRAGMA table_info(generations)")}
    assert "ab_arm" in cols, "v6 migration must add ab_arm to generations"


def test_schema_user_version_is_at_least_6(fresh_db: HistoryStore) -> None:
    # Test originally pinned to v6 (rc5 surface). Relaxed to >= 6 so future
    # schema bumps don't break this v6-surface intent — same fix-pattern rc1
    # applied to the v3 literal in test_v1_17_schema.py / test_v1_18_schema.py.
    v = fresh_db._conn.execute("PRAGMA user_version").fetchone()[0]
    assert v >= 6


# ---------------------------------------------------------------------------
# A1: ab_arm round-trip — POST /v2/audio-to-video → history row
# ---------------------------------------------------------------------------


def test_ab_arm_round_trip(fresh_db: HistoryStore) -> None:
    """End-to-end round-trip: MCP-shaped body → Pydantic alias → params dict
    → worker_loop's ``_params.get("ab_arm")`` save call → generations row.

    Skips the actual HTTP layer (which would also pull in the worker
    queue + a real LTX dispatch) but exercises every transform between
    them — the parts that broke pre-rc5 with silent drops.
    """
    from server import AudioToVideoRequest, _HISTORY_ONLY_PARAMS, _strip_history_params

    # 1. MCP forwards a body with `_ab_arm: "candidate"` (the wire shape
    #    that v0.8.1 `_build_clip_body` produces).
    mcp_body = {
        "prompt": "rooftop sunset",
        "audio_uri": "storage://fake-audio",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 4.0,
        "fps": 24.0,
        "_ab_arm": "candidate",
    }

    # 2. Pydantic accepts the alias; body.ab_arm carries the value.
    body = AudioToVideoRequest.model_validate(mcp_body)
    assert body.ab_arm == "candidate"

    # 3. Endpoint threads body.ab_arm into the params dict that the
    #    worker_loop will eventually pass to history.save. This mirrors
    #    the `params = dict(..., ab_arm=body.ab_arm)` line in
    #    `v2_audio_to_video`.
    params = {"prompt": body.prompt, "ab_arm": body.ab_arm, "lora_path": "/x"}

    # 4. `_HISTORY_ONLY_PARAMS` strips ab_arm before manager dispatch so
    #    `_run_a2v` never sees it. The strip is the contract that
    #    history-only fields ride job.params without polluting manager
    #    kwargs.
    assert "ab_arm" in _HISTORY_ONLY_PARAMS
    stripped = _strip_history_params(params)
    assert "ab_arm" not in stripped
    assert "lora_path" in stripped  # non-history-only kwargs survive

    # 5. worker_loop reads `_params.get("ab_arm")` and calls
    #    history.save(ab_arm=...) — replicate the relevant save call.
    fresh_db.save(
        job_id="mv_round_trip_1",
        api_key="key_x",
        job_type="audio-to-video",
        prompt=body.prompt,
        model=body.model,
        width=1920,
        height=1080,
        turbo=False,
        status="completed",
        result_uri="storage://r1",
        result_bytes=None,
        created_at=time.time(),
        completed_at=time.time(),
        ab_arm=params.get("ab_arm"),
    )

    # 6. The generations.ab_arm column carries the value from the
    #    original MCP body — completes the loop.
    row = fresh_db._conn.execute(
        "SELECT ab_arm FROM generations WHERE id = ?", ("mv_round_trip_1",)
    ).fetchone()
    assert row["ab_arm"] == "candidate"


def test_ab_arm_null_when_omitted(fresh_db: HistoryStore) -> None:
    """Non-A/B clips: ab_arm column stays NULL."""
    fresh_db.save(
        job_id="job_no_ab",
        api_key="key_x",
        job_type="audio-to-video",
        prompt="p",
        model="ltx-2-3-fast",
        width=512,
        height=512,
        turbo=False,
        status="completed",
        result_uri="storage://r2",
        result_bytes=None,
        created_at=time.time(),
        completed_at=time.time(),
    )
    row = fresh_db._conn.execute(
        "SELECT ab_arm FROM generations WHERE id = ?", ("job_no_ab",)
    ).fetchone()
    assert row["ab_arm"] is None


def test_pydantic_alias_accepts_underscore_ab_arm() -> None:
    """The wire field is ``_ab_arm`` (MCP forwards it underscore-prefixed).

    Pydantic ``alias="_ab_arm"`` + ``populate_by_name=True`` lets the
    model accept either ``_ab_arm`` (from JSON body) or ``ab_arm``
    (from Python kwargs in tests). Without this, every video v2
    endpoint silently drops the cohort marker.
    """
    from server import AudioToVideoRequest

    body = AudioToVideoRequest.model_validate({
        "prompt": "x",
        "audio_uri": "storage://a",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "_ab_arm": "candidate",
    })
    assert body.ab_arm == "candidate"


def test_pydantic_alias_default_none() -> None:
    """Body without ``_ab_arm`` → ``body.ab_arm is None``."""
    from server import TextToVideoRequest

    body = TextToVideoRequest.model_validate({
        "prompt": "x",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 4.0,
        "fps": 24.0,
    })
    assert body.ab_arm is None


# ---------------------------------------------------------------------------
# ab_decision — column-based cohort lookup
# ---------------------------------------------------------------------------


def _insert_ab_clip(
    history: HistoryStore,
    *,
    clip_id: str,
    composition_id: str,
    ab_arm: str | None,
    validator_score: float,
) -> None:
    history._conn.execute(
        """INSERT INTO generations
           (id, api_key_hash, job_type, prompt, model, width, height, turbo,
            status, result_uri, created_at,
            validator_score, composition_id, ab_arm)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            clip_id, _hash_key("k"), "audio-to-video", "p", "ltx-2-3-fast",
            512, 512, 0,
            "completed", f"storage://{clip_id}", time.time(),
            validator_score, composition_id, ab_arm,
        ),
    )
    history._conn.commit()


def test_ab_decision_reads_from_ab_arm_column(fresh_db: HistoryStore) -> None:
    """ab_decision groups by composition_id, partitioned by ab_arm column."""
    _insert_ab_clip(fresh_db, clip_id="c1", composition_id="mv_1",
                    ab_arm="candidate", validator_score=0.9)
    _insert_ab_clip(fresh_db, clip_id="c2", composition_id="mv_2",
                    ab_arm="candidate", validator_score=0.8)
    _insert_ab_clip(fresh_db, clip_id="b1", composition_id="mv_3",
                    ab_arm="baseline", validator_score=0.6)

    cand_means = abd._fetch_arm_means(
        fresh_db, arm="candidate", candidate_lora="any-lora", limit=100,
    )
    base_means = abd._fetch_arm_means(
        fresh_db, arm="baseline", candidate_lora="any-lora", limit=100,
    )
    # Per-MV mean: each MV contributes its own AVG.
    assert sorted(cand_means) == pytest.approx([0.8, 0.9])
    assert base_means == pytest.approx([0.6])


def test_ab_decision_skips_unmarked_clips(fresh_db: HistoryStore) -> None:
    """Clips with ab_arm IS NULL are excluded from both cohorts."""
    _insert_ab_clip(fresh_db, clip_id="x", composition_id="mv_x",
                    ab_arm=None, validator_score=0.95)
    cand = abd._fetch_arm_means(fresh_db, arm="candidate", candidate_lora="", limit=100)
    base = abd._fetch_arm_means(fresh_db, arm="baseline", candidate_lora="", limit=100)
    assert cand == []
    assert base == []


# ---------------------------------------------------------------------------
# A2: motion_intent passive-dispatch wire-through
# ---------------------------------------------------------------------------


def test_dispatch_validator_fetches_motion_intent(tmp_path: Path) -> None:
    """``_dispatch_validator`` reads generations.motion_intent and forwards
    it to ``run_all_tiers(motion_intent=...)``.
    """
    import server as server_mod
    from job_queue import Job, JobStatus, JobType

    # Use a fresh history to avoid clobbering global state.
    db = tmp_path / "history.db"
    h = HistoryStore(db_path=db)
    # Insert a row with motion_intent set.
    h._conn.execute(
        """INSERT INTO generations
           (id, api_key_hash, job_type, prompt, model, width, height, turbo,
            status, result_uri, created_at, motion_intent)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "j1", _hash_key("k"), "audio-to-video", "rooftop", "ltx-2-3-fast",
            512, 512, 0, "completed", "storage://r", time.time(),
            "slow drift, person walking",
        ),
    )
    h._conn.commit()

    # Stand up a fake video file for uploads.resolve.
    fake_video = tmp_path / "r.mp4"
    fake_video.write_bytes(b"fakevideo")

    job = Job(
        id="j1",
        type=JobType.AUDIO_TO_VIDEO,
        params={"prompt": "rooftop"},
        api_key="k",
    )
    job.result_uri = "storage://r"
    job.status = JobStatus.COMPLETED

    captured: dict[str, str | None] = {}

    async def _fake_run_all_tiers(**kwargs):
        captured["motion_intent"] = kwargs.get("motion_intent")
        return {
            "validator_version": "test",
            "composite_score": 0.5,
            "tier1": None, "tier2": None, "tier3": None,
            "recommendation": "warn",
            "reasoning_summary": "",
            "ran_at": time.time(),
        }

    with patch.object(server_mod, "history", h), \
         patch.object(server_mod.uploads, "resolve", return_value=fake_video), \
         patch("validator.run_all_tiers", new=_fake_run_all_tiers):
        asyncio.run(server_mod._dispatch_validator(job))

    assert captured.get("motion_intent") == "slow drift, person walking"


def test_dispatch_validator_motion_intent_null_safe(tmp_path: Path) -> None:
    """When motion_intent is NULL on the row, ``_dispatch_validator``
    forwards motion_intent=None and tier3 omits the intent line —
    byte-identical to rc4 behavior.
    """
    import server as server_mod
    from job_queue import Job, JobStatus, JobType

    db = tmp_path / "history.db"
    h = HistoryStore(db_path=db)
    h._conn.execute(
        """INSERT INTO generations
           (id, api_key_hash, job_type, prompt, model, width, height, turbo,
            status, result_uri, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "j2", _hash_key("k"), "audio-to-video", "p", "ltx-2-3-fast",
            512, 512, 0, "completed", "storage://r2", time.time(),
        ),
    )
    h._conn.commit()
    fake = tmp_path / "r2.mp4"
    fake.write_bytes(b"x")

    job = Job(id="j2", type=JobType.AUDIO_TO_VIDEO, params={"prompt": "p"}, api_key="k")
    job.result_uri = "storage://r2"
    job.status = JobStatus.COMPLETED

    captured: dict[str, str | None] = {}

    async def _fake(**kwargs):
        captured["motion_intent"] = kwargs.get("motion_intent")
        return {
            "validator_version": "t", "composite_score": 0.5,
            "tier1": None, "tier2": None, "tier3": None,
            "recommendation": "warn", "reasoning_summary": "",
            "ran_at": time.time(),
        }

    with patch.object(server_mod, "history", h), \
         patch.object(server_mod.uploads, "resolve", return_value=fake), \
         patch("validator.run_all_tiers", new=_fake):
        asyncio.run(server_mod._dispatch_validator(job))

    assert captured.get("motion_intent") is None


# ---------------------------------------------------------------------------
# v1.18.0-rc6: shot_uuid + shot_config_key round-trip
# ---------------------------------------------------------------------------


def test_shot_uuid_round_trip(fresh_db: HistoryStore) -> None:
    """End-to-end: MCP body with `shot_uuid` → Pydantic field → params dict
    → ``_HISTORY_ONLY_PARAMS`` strip → worker_loop's
    ``_params.get("shot_uuid")`` save call → generations row.

    Pre-rc6 every MCP-driven clip silently dropped this field because the
    Pydantic model didn't declare it (extra="ignore" sink).
    """
    from server import AudioToVideoRequest, _HISTORY_ONLY_PARAMS, _strip_history_params

    mcp_body = {
        "prompt": "rooftop sunset",
        "audio_uri": "storage://fake-audio",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 4.0,
        "fps": 24.0,
        "shot_uuid": "shot_abc123",
    }
    body = AudioToVideoRequest.model_validate(mcp_body)
    assert body.shot_uuid == "shot_abc123"

    params = {"prompt": body.prompt, "shot_uuid": body.shot_uuid, "lora_path": "/x"}
    assert "shot_uuid" in _HISTORY_ONLY_PARAMS
    stripped = _strip_history_params(params)
    assert "shot_uuid" not in stripped
    assert "lora_path" in stripped

    fresh_db.save(
        job_id="rc6_shot_uuid_1",
        api_key="key_x",
        job_type="audio-to-video",
        prompt=body.prompt,
        model=body.model,
        width=1920,
        height=1080,
        turbo=False,
        status="completed",
        result_uri="storage://r1",
        result_bytes=None,
        created_at=time.time(),
        completed_at=time.time(),
        shot_uuid=params.get("shot_uuid"),
    )

    row = fresh_db._conn.execute(
        "SELECT shot_uuid FROM generations WHERE id = ?", ("rc6_shot_uuid_1",)
    ).fetchone()
    assert row["shot_uuid"] == "shot_abc123"


def test_shot_config_key_round_trip(fresh_db: HistoryStore) -> None:
    """Same as ``test_shot_uuid_round_trip`` but for ``shot_config_key``."""
    from server import TextToVideoRequest, _HISTORY_ONLY_PARAMS, _strip_history_params

    mcp_body = {
        "prompt": "wide shot, person walking",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 4.0,
        "fps": 24.0,
        "shot_config_key": "wide:1920x1080:fast:slow_drift",
    }
    body = TextToVideoRequest.model_validate(mcp_body)
    assert body.shot_config_key == "wide:1920x1080:fast:slow_drift"

    params = {
        "prompt": body.prompt,
        "shot_config_key": body.shot_config_key,
        "lora_path": "/x",
    }
    assert "shot_config_key" in _HISTORY_ONLY_PARAMS
    stripped = _strip_history_params(params)
    assert "shot_config_key" not in stripped

    fresh_db.save(
        job_id="rc6_shot_cfg_1",
        api_key="key_x",
        job_type="text-to-video",
        prompt=body.prompt,
        model=body.model,
        width=1920,
        height=1080,
        turbo=False,
        status="completed",
        result_uri="storage://r2",
        result_bytes=None,
        created_at=time.time(),
        completed_at=time.time(),
        shot_config_key=params.get("shot_config_key"),
    )

    row = fresh_db._conn.execute(
        "SELECT shot_config_key FROM generations WHERE id = ?", ("rc6_shot_cfg_1",)
    ).fetchone()
    assert row["shot_config_key"] == "wide:1920x1080:fast:slow_drift"


def test_shot_lineage_fields_default_none() -> None:
    """Body without ``shot_uuid`` / ``shot_config_key`` → both None."""
    from server import ImageToVideoRequest

    body = ImageToVideoRequest.model_validate({
        "prompt": "x",
        "image_uri": "storage://i",
        "model": "ltx-2-3-fast",
        "resolution": "1920x1080",
        "duration": 4.0,
        "fps": 24.0,
    })
    assert body.shot_uuid is None
    assert body.shot_config_key is None


def test_shot_lineage_on_all_video_models() -> None:
    """Every v2 video request model declares both lineage fields, so MCP
    can forward them from any of the 6 video endpoints without silent
    drop. A regression here means an MCP-driven clip silently loses
    lineage on whichever endpoint lost the declaration.
    """
    from server import (
        TextToVideoRequest, ImageToVideoRequest, AudioToVideoRequest,
        RetakeRequest, VideoOutpaintRequest, VideoHdrRequest,
    )
    for cls in (
        TextToVideoRequest, ImageToVideoRequest, AudioToVideoRequest,
        RetakeRequest, VideoOutpaintRequest, VideoHdrRequest,
    ):
        assert "shot_uuid" in cls.model_fields, f"{cls.__name__} missing shot_uuid"
        assert "shot_config_key" in cls.model_fields, (
            f"{cls.__name__} missing shot_config_key"
        )
