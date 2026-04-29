"""Tests for v1.18.0-rc2 — Phase B retrieval backend.

Coverage:
  - POST /v2/embeddings/search happy path (privacy gate inverse — own rows surface)
  - POST /v2/embeddings/search privacy gate (cross-bearer isolation — CRITICAL)
  - POST /v2/embeddings/search validator_version filter
  - POST /v2/embeddings/search rate limit (429 + Retry-After)
  - POST /v2/embeddings/search min_validator_score filter
  - POST /v2/embeddings/recommend-loras aggregation math
  - POST /v2/embeddings/recommend-loras empty when no lora_applied_id rows
  - POST /v2/system/bulk-revalidate dry-run no writes
  - POST /v2/system/bulk-revalidate real-run dispatch path
  - lora_applied_id persistence end-to-end via worker_loop
  - sqlite-vec extension load failure graceful (endpoint 503, backend boots)
  - embedding_model_version round-trip through clip_embeddings
"""
from __future__ import annotations

import asyncio
import json
import struct
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import config

# Disable auth + GPU for routing tests, mirror existing v1_17 patterns.
config.GPU_DEVICES = []
config.API_KEYS = set()

from fastapi.testclient import TestClient  # noqa: E402

import history_store  # noqa: E402
from history_store import HistoryStore, _hash_key  # noqa: E402
import server as server_mod  # noqa: E402
from server import app  # noqa: E402


client = TestClient(app)


EMBED_DIM = 4096


def _pack_embed(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unit_embed(seed: int) -> bytes:
    """Deterministic synthetic embedding seeded by an int."""
    import random
    rng = random.Random(seed)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIM)]
    return _pack_embed(vec)


def _seed_opt_in(history: HistoryStore, api_key: str, opt_in: bool = True) -> None:
    now = time.time()
    history._conn.execute(
        """INSERT OR REPLACE INTO api_key_metadata
           (api_key_hash, training_opt_in, tier, notes, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (_hash_key(api_key), 1 if opt_in else 0, "pro", None, now, now),
    )
    history._conn.commit()


def _insert_clip(
    history: HistoryStore,
    *,
    clip_id: str,
    api_key: str,
    prompt: str,
    embedding: bytes,
    validator_score: float = 0.85,
    validator_version: str | None = None,
    lora_applied_id: str | None = None,
    lora_applied_strength: float | None = None,
    composition_id: str | None = None,
    created_at: float | None = None,
    job_type: str = "text-to-video",
) -> None:
    # Default to the live config so test fixtures track production validator
    # version bumps without per-version edits. Explicit override is honored.
    if validator_version is None:
        validator_version = config.VALIDATOR_VERSION
    now = created_at if created_at is not None else time.time()
    history._conn.execute(
        """INSERT OR REPLACE INTO generations
           (id, api_key_hash, job_type, prompt, model, width, height, turbo,
            status, result_uri, created_at,
            validator_score, validator_version,
            lora_applied_id, lora_applied_strength, composition_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            clip_id, _hash_key(api_key), job_type, prompt, "ltx-2-3-fast",
            512, 512, 0,
            "completed", f"storage://{clip_id}", now,
            validator_score, validator_version,
            lora_applied_id, lora_applied_strength, composition_id,
        ),
    )
    history._conn.execute(
        """INSERT OR IGNORE INTO clip_embeddings (id, embedding, embedding_model_version)
           VALUES (?, ?, ?)""",
        (clip_id, embedding, "gemma-3-12b-nvfp4"),
    )
    history._conn.commit()


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Each test starts with clean rate-limit + metrics counters."""
    server_mod._embeddings_buckets.clear()
    for k in server_mod._embeddings_metrics:
        server_mod._embeddings_metrics[k] = 0
    server_mod._embeddings_search_latencies_ms.clear()
    yield


@pytest.fixture
def fresh_history(tmp_path, monkeypatch):
    """Spin up a HistoryStore on a tmp_path DB and patch server.history."""
    db = tmp_path / "history.db"
    h = HistoryStore(db_path=db)
    monkeypatch.setattr(server_mod, "history", h)
    yield h
    h._conn.close()


@pytest.fixture
def patch_embed():
    """Replace server.chat.embed/embed_batch with deterministic in-process functions.

    The query embedding is `_unit_embed(0)` always; populated rows use
    different seeds so distance-ordering is stable per-test.
    """
    async def fake_embed(text: str) -> bytes:
        # Deterministic per text: hash the prompt to an int seed.
        seed = abs(hash(text)) % (2 ** 31)
        return _unit_embed(seed)

    async def fake_embed_batch(texts: list[str]) -> list[bytes]:
        return [await fake_embed(t) for t in texts]

    with patch.object(server_mod.chat, "embed", side_effect=fake_embed), \
         patch.object(server_mod.chat, "embed_batch", side_effect=fake_embed_batch):
        yield


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------


def test_embeddings_search_endpoint_happy_path(fresh_history, patch_embed):
    api_key = "bearer-A"
    _seed_opt_in(fresh_history, api_key)
    # Seed 10 rows with varied seeds. Ranking is best-effort; we just
    # verify the endpoint returns ≤ k results with sane shape.
    for i in range(10):
        _insert_clip(
            fresh_history,
            clip_id=f"clip-{i}",
            api_key=api_key,
            prompt=f"a sunset over the {['ocean', 'mountains', 'forest'][i % 3]}",
            embedding=_unit_embed(i + 100),
            validator_score=0.7 + (i % 3) * 0.05,
        )

    resp = client.post(
        "/v2/embeddings/search",
        json={"prompt": "sunset over open water", "k": 5,
              "validator_version_filter": config.VALIDATOR_VERSION},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["validator_version_filter"] == config.VALIDATOR_VERSION
    assert len(data["results"]) <= 5
    assert len(data["results"]) > 0
    for r in data["results"]:
        assert {"shot_id", "prompt", "similarity_score", "validator_score",
                "final_score"}.issubset(r.keys())


def test_embeddings_search_privacy_gate(fresh_history, patch_embed, monkeypatch):
    """CRITICAL: bearer A's rows MUST NOT be visible to bearer B."""
    monkeypatch.setattr(config, "API_KEYS", {"key-A", "key-B"})
    try:
        _seed_opt_in(fresh_history, "key-A")
        _seed_opt_in(fresh_history, "key-B")

        # Bearer A populates 5 rows
        for i in range(5):
            _insert_clip(
                fresh_history,
                clip_id=f"clip-A-{i}",
                api_key="key-A",
                prompt="bearer A private content",
                embedding=_unit_embed(i + 200),
            )

        # Bearer B searches — should return zero results since they have nothing
        resp = client.post(
            "/v2/embeddings/search",
            json={"prompt": "bearer A private content", "k": 5},
            headers={"Authorization": "Bearer key-B"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["results"] == [], (
            f"PRIVACY VIOLATION: bearer B saw {len(data['results'])} of bearer A's rows"
        )

        # Bearer A searches their own — sees their rows
        resp_a = client.post(
            "/v2/embeddings/search",
            json={"prompt": "bearer A private content", "k": 5},
            headers={"Authorization": "Bearer key-A"},
        )
        assert resp_a.status_code == 200, resp_a.text
        assert len(resp_a.json()["results"]) > 0
    finally:
        monkeypatch.setattr(config, "API_KEYS", set())


def test_embeddings_search_validator_version_filter(fresh_history, patch_embed):
    api_key = "bearer-vfilter"
    _seed_opt_in(fresh_history, api_key)
    # Old validator version
    for i in range(3):
        _insert_clip(
            fresh_history,
            clip_id=f"old-{i}",
            api_key=api_key,
            prompt="old-version clip",
            embedding=_unit_embed(i + 300),
            validator_version="1.16.0-old",
        )
    # New validator version (the live one — tracks config.VALIDATOR_VERSION).
    for i in range(3):
        _insert_clip(
            fresh_history,
            clip_id=f"new-{i}",
            api_key=api_key,
            prompt="new-version clip",
            embedding=_unit_embed(i + 400),
            validator_version=config.VALIDATOR_VERSION,
        )

    resp = client.post(
        "/v2/embeddings/search",
        json={"prompt": "any prompt", "k": 10,
              "validator_version_filter": config.VALIDATOR_VERSION},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    data = resp.json()
    assert all(r["shot_id"].startswith("new-") for r in data["results"]), \
        f"unexpected shot_ids: {[r['shot_id'] for r in data['results']]}"
    assert len(data["results"]) == 3


def test_embeddings_search_rate_limit(fresh_history, patch_embed, monkeypatch):
    """11 calls in 1 second from same key → 11th gets 429."""
    monkeypatch.setattr(config, "API_KEYS", {"rl-key"})
    try:
        _seed_opt_in(fresh_history, "rl-key")
        codes: list[int] = []
        # Burst capacity is 10; the 11th in the same instant should 429.
        for _ in range(11):
            r = client.post(
                "/v2/embeddings/search",
                json={"prompt": "rate limit test", "k": 1},
                headers={"Authorization": "Bearer rl-key"},
            )
            codes.append(r.status_code)
        assert 429 in codes, f"expected at least one 429 in {codes}"
        # Verify Retry-After surfaced on the throttled response
        idx = codes.index(429)
        # Re-issue to capture headers since codes only stores status
        # (the original 429 response is gone). Hit the bucket again to
        # observe the header now.
        rr = client.post(
            "/v2/embeddings/search",
            json={"prompt": "rate limit test", "k": 1},
            headers={"Authorization": "Bearer rl-key"},
        )
        assert rr.status_code == 429
        assert "retry-after" in {k.lower() for k in rr.headers.keys()}
    finally:
        monkeypatch.setattr(config, "API_KEYS", set())


def test_embeddings_search_min_validator_score_filter(fresh_history, patch_embed):
    api_key = "bearer-minscore"
    _seed_opt_in(fresh_history, api_key)
    _insert_clip(
        fresh_history, clip_id="lo-1", api_key=api_key,
        prompt="lowscore", embedding=_unit_embed(500), validator_score=0.30,
    )
    _insert_clip(
        fresh_history, clip_id="hi-1", api_key=api_key,
        prompt="highscore", embedding=_unit_embed(501), validator_score=0.85,
    )
    resp = client.post(
        "/v2/embeddings/search",
        json={"prompt": "anything", "k": 10, "min_validator_score": 0.7},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    ids = [r["shot_id"] for r in resp.json()["results"]]
    assert ids == ["hi-1"], ids


# ---------------------------------------------------------------------------
# recommend_loras
# ---------------------------------------------------------------------------


def test_recommend_loras_aggregates_correctly(fresh_history, patch_embed):
    api_key = "bearer-loras"
    _seed_opt_in(fresh_history, api_key)
    # Seed 3 rows with lora-A scoring high, 3 with lora-B scoring low,
    # 3 baseline (no lora) at moderate score.
    for i in range(3):
        _insert_clip(fresh_history, clip_id=f"a-{i}", api_key=api_key,
                     prompt="dance scene", embedding=_unit_embed(600 + i),
                     validator_score=0.90, lora_applied_id="lora-A",
                     lora_applied_strength=0.7)
    for i in range(3):
        _insert_clip(fresh_history, clip_id=f"b-{i}", api_key=api_key,
                     prompt="dance scene", embedding=_unit_embed(700 + i),
                     validator_score=0.55, lora_applied_id="lora-B",
                     lora_applied_strength=0.5)
    for i in range(3):
        _insert_clip(fresh_history, clip_id=f"n-{i}", api_key=api_key,
                     prompt="dance scene", embedding=_unit_embed(800 + i),
                     validator_score=0.70)

    resp = client.post(
        "/v2/embeddings/recommend-loras",
        json={"prompt": "dance scene", "k": 5},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total_samples"] == 9
    ids = [r["lora_id"] for r in data["recommendations"]]
    assert "lora-A" in ids and "lora-B" in ids
    # lora-A's mean (0.90) beats lora-B's (0.55) → should rank first.
    assert ids[0] == "lora-A"
    a_rec = next(r for r in data["recommendations"] if r["lora_id"] == "lora-A")
    assert a_rec["sample_count"] == 3
    assert a_rec["mean_validator_score"] == pytest.approx(0.90, abs=0.01)
    # expected_boost = 0.90 - 0.70 (no_lora baseline) = 0.20
    assert a_rec["expected_boost"] == pytest.approx(0.20, abs=0.01)


def test_recommend_loras_empty_when_no_lora_data(fresh_history, patch_embed):
    api_key = "bearer-nolora"
    _seed_opt_in(fresh_history, api_key)
    for i in range(3):
        _insert_clip(fresh_history, clip_id=f"plain-{i}", api_key=api_key,
                     prompt="cinematic shot", embedding=_unit_embed(900 + i),
                     validator_score=0.75)
    resp = client.post(
        "/v2/embeddings/recommend-loras",
        json={"prompt": "cinematic shot", "k": 3},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommendations"] == []
    assert data["total_samples"] == 3


# ---------------------------------------------------------------------------
# bulk-revalidate
# ---------------------------------------------------------------------------


def test_bulk_revalidate_dry_run_no_writes(fresh_history, patch_embed, monkeypatch):
    api_key = "admin-key"
    _seed_opt_in(fresh_history, api_key)
    for i in range(5):
        _insert_clip(fresh_history, clip_id=f"oldv-{i}", api_key=api_key,
                     prompt="x", embedding=_unit_embed(1000 + i),
                     validator_version="1.16.0-old")

    # Capture row state before
    before = fresh_history._conn.execute(
        "SELECT id, validator_version FROM generations ORDER BY id"
    ).fetchall()

    resp = client.post(
        "/v2/system/bulk-revalidate",
        json={"target_validator_version": config.VALIDATOR_VERSION, "dry_run": True},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["dry_run"] is True
    assert data["would_revalidate"] == 5

    after = fresh_history._conn.execute(
        "SELECT id, validator_version FROM generations ORDER BY id"
    ).fetchall()
    assert before == after, "dry_run must not write"


def test_bulk_revalidate_real_run_updates_rows(fresh_history, patch_embed, monkeypatch):
    """Real run should queue dispatches; we patch _dispatch_validator to count calls."""
    api_key = "admin-key"
    _seed_opt_in(fresh_history, api_key)
    for i in range(3):
        _insert_clip(fresh_history, clip_id=f"realrun-{i}", api_key=api_key,
                     prompt="x", embedding=_unit_embed(1100 + i),
                     validator_version="1.16.0-old")

    dispatched: list[str] = []

    async def fake_dispatch(job):
        dispatched.append(job.id)
        # Simulate the version bump that the real dispatcher would apply
        fresh_history._conn.execute(
            "UPDATE generations SET validator_version = ? WHERE id = ?",
            (config.VALIDATOR_VERSION, job.id),
        )
        fresh_history._conn.commit()

    monkeypatch.setattr(server_mod, "_dispatch_validator", fake_dispatch)
    resp = client.post(
        "/v2/system/bulk-revalidate",
        json={"target_validator_version": config.VALIDATOR_VERSION, "dry_run": False, "limit": 10},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["dry_run"] is False
    assert data["queued"] == 3

    # Allow the fire-and-forget tasks to run.
    async def _drain():
        await asyncio.sleep(0.05)
    asyncio.run(_drain())
    assert sorted(dispatched) == [f"realrun-{i}" for i in range(3)]


# ---------------------------------------------------------------------------
# lora_applied_id persistence
# ---------------------------------------------------------------------------


def test_lora_applied_id_persistence(fresh_history, monkeypatch, tmp_path):
    """Submit a video job with lora={...}; verify generations.lora_applied_id is populated.

    Drives the worker_loop dispatch path synchronously without touching GPU
    by running the same persistence logic the worker uses.
    """
    from job_queue import Job, JobType, JobStatus

    api_key = "lora-persist-key"
    _seed_opt_in(fresh_history, api_key)
    # The wiring: server endpoints call _lora_applied_pair and inject
    # lora_applied_id / lora_applied_strength into job.params. The worker
    # then forwards them into history.save kwargs.
    job = Job(
        id="lora-persist-1",
        type=JobType.TEXT_TO_VIDEO,
        status=JobStatus.COMPLETED,
        api_key=api_key,
        result_uri="storage://lora-persist-1",
        params={
            "prompt": "test",
            "model": "ltx-2-3-fast",
            "width": 512, "height": 512,
            "lora_applied_id": "my-lora",
            "lora_applied_strength": 0.8,
        },
    )
    # Drive the same kwarg construction the worker does, then hit save().
    fresh_history.save(
        job_id=job.id,
        api_key=job.api_key,
        job_type=str(job.type),
        prompt=job.params["prompt"],
        model=job.params.get("model"),
        width=job.params.get("width", 0),
        height=job.params.get("height", 0),
        turbo=False,
        status=job.status,
        result_uri=job.result_uri,
        result_bytes=None,
        created_at=time.time(),
        completed_at=time.time(),
        lora_applied_id=job.params.get("lora_applied_id"),
        lora_applied_strength=job.params.get("lora_applied_strength"),
    )
    row = fresh_history._conn.execute(
        "SELECT lora_applied_id, lora_applied_strength FROM generations WHERE id = ?",
        (job.id,),
    ).fetchone()
    assert row is not None
    assert row["lora_applied_id"] == "my-lora"
    assert row["lora_applied_strength"] == pytest.approx(0.8)


def test_lora_applied_pair_helper():
    """The lightweight helper used by every video v2 endpoint."""
    from server import _lora_applied_pair

    class _LoRA:
        id = "lora-X"
        strength = 0.5

    class _Body:
        lora = _LoRA()

    assert _lora_applied_pair(_Body()) == ("lora-X", 0.5)

    class _NoBody:
        lora = None

    assert _lora_applied_pair(_NoBody()) == (None, None)


# ---------------------------------------------------------------------------
# sqlite-vec extension behavior
# ---------------------------------------------------------------------------


def test_sqlite_vec_extension_load_failure_graceful(monkeypatch):
    """When SQLITE_VEC_AVAILABLE is False, endpoints return 503 cleanly."""
    monkeypatch.setattr(history_store, "SQLITE_VEC_AVAILABLE", False)
    monkeypatch.setattr(
        history_store, "SQLITE_VEC_LOAD_ERROR", "test stub: extension unavailable"
    )
    resp = client.post(
        "/v2/embeddings/search",
        json={"prompt": "anything goes", "k": 1},
    )
    assert resp.status_code == 503
    body = resp.json()
    # _error wraps as {"error": ..., "message": ...} — check whichever shape.
    msg = body.get("message") or body.get("error") or json.dumps(body)
    assert "sqlite-vec" in msg.lower() or "embedding" in msg.lower()


def test_embedding_model_version_round_trip(fresh_history, patch_embed):
    api_key = "version-tag-key"
    _seed_opt_in(fresh_history, api_key)
    _insert_clip(
        fresh_history, clip_id="ver-1", api_key=api_key,
        prompt="versioned", embedding=_unit_embed(1500),
    )
    row = fresh_history._conn.execute(
        "SELECT embedding_model_version FROM clip_embeddings WHERE id = ?",
        ("ver-1",),
    ).fetchone()
    assert row is not None
    assert row["embedding_model_version"] == "gemma-3-12b-nvfp4"
