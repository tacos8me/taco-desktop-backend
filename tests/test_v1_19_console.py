"""Tests for v1.19 — operator console stats endpoints (Stream B-back).

Coverage:
  - GET /v1/system/validator-stats: shape, histogram bucketing, by_version
  - GET /v1/system/ab-status: admin gate, t-test math
  - GET /v1/system/training-runs: admin gate, eval_metrics_json parsing
  - GET /v1/system/preference-pairs-count: by_source aggregate
  - GET /v1/system/validator-failures: ring buffer roundtrip
  - GET /v1/api-keys/me/training-opt-in: read-back
  - POST /v1/api-keys/me/training-opt-in: toggle persists
  - Rate-limit middleware coverage extension
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import config

# Disable auth + GPU before importing server (mirror existing v1_17/v1_18 patterns)
config.GPU_DEVICES = []
config.API_KEYS = set()

from fastapi.testclient import TestClient  # noqa: E402

from history_store import HistoryStore, _hash_key  # noqa: E402
import server as server_mod  # noqa: E402
from server import app  # noqa: E402
import _validator_failures  # noqa: E402


client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_history(tmp_path: Path, monkeypatch):
    """Brand-new HistoryStore patched into server.history."""
    db = tmp_path / "history.db"
    h = HistoryStore(db_path=db)
    monkeypatch.setattr(server_mod, "history", h)
    yield h
    h._conn.close()


@pytest.fixture(autouse=True)
def _reset_failures():
    _validator_failures.clear()
    yield
    _validator_failures.clear()


def _seed_opt_in(history: HistoryStore, key: str, opt_in: int = 1) -> str:
    h = _hash_key(key)
    now = time.time()
    history._conn.execute(
        """INSERT OR REPLACE INTO api_key_metadata
           (api_key_hash, training_opt_in, tier, notes, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (h, opt_in, "pro", None, now, now),
    )
    history._conn.commit()
    return h


def _insert_clip(
    history: HistoryStore,
    *,
    clip_id: str,
    api_key: str = "bearer-X",
    validator_score: float | None = 0.85,
    validator_version: str = "1.17.0-rc5",
    composition_id: str | None = None,
    ab_arm: str | None = None,
    lora_applied_id: str | None = None,
    created_at: float | None = None,
) -> None:
    now = created_at if created_at is not None else time.time()
    history._conn.execute(
        """INSERT OR REPLACE INTO generations
           (id, api_key_hash, job_type, prompt, model, width, height, turbo,
            status, result_uri, created_at,
            validator_score, validator_version,
            composition_id, ab_arm, lora_applied_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            clip_id, _hash_key(api_key), "text-to-video", "test", "ltx-2-3-fast",
            512, 512, 0,
            "completed", f"storage://{clip_id}", now,
            validator_score, validator_version,
            composition_id, ab_arm, lora_applied_id,
        ),
    )
    history._conn.commit()


def _insert_pair(history: HistoryStore, *, signal_source: str, age_s: float = 0.0) -> None:
    history._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            used_in_training_run_id, created_at, validator_version)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (f"c-{time.time_ns()}", f"r-{time.time_ns()}", signal_source,
         0.5, None, time.time() - age_s, "1.17.0-rc5"),
    )
    history._conn.commit()


def _insert_training_run(history: HistoryStore, *, run_id: str, **kw) -> None:
    history._conn.execute(
        """INSERT OR REPLACE INTO training_runs
           (run_id, base_model, base_model_sha, lora_output_path,
            lora_registry_id, num_pairs, val_loss, eval_metrics_json,
            trained_at, deployed_at, deprecated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            kw.get("base_model", "ltx-2.3"),
            kw.get("base_model_sha", "abc123"),
            kw.get("lora_output_path", f"/tmp/{run_id}.safetensors"),
            kw.get("lora_registry_id", run_id),
            kw.get("num_pairs", 100),
            kw.get("val_loss", 0.42),
            kw.get("eval_metrics_json", '{"acc": 0.91}'),
            kw.get("trained_at", time.time()),
            kw.get("deployed_at"),
            kw.get("deprecated_at"),
        ),
    )
    history._conn.commit()


# ---------------------------------------------------------------------------
# /v1/system/validator-stats
# ---------------------------------------------------------------------------


def test_validator_stats_shape_and_buckets(fresh_history):
    # Seed scores spanning multiple buckets
    for i, score in enumerate([0.05, 0.15, 0.65, 0.72, 0.85, 0.95]):
        _insert_clip(fresh_history, clip_id=f"c{i}", validator_score=score)
    resp = client.get("/v1/system/validator-stats?window=24h")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert {
        "histogram", "mean", "p50", "p95", "count", "by_version",
        "timeline", "timeline_bucket_seconds", "window_seconds",
    } <= data.keys()
    assert len(data["histogram"]) == 10
    assert len(data["timeline"]) == 48
    assert data["count"] == 6
    # Bucket 0 (0.0-0.1) should have 1 (0.05); bucket 1: 1; bucket 6: 1; etc.
    assert data["histogram"][0] == 1
    assert data["histogram"][1] == 1
    # Cross-check against direct SQL aggregate
    direct = fresh_history._conn.execute(
        "SELECT AVG(validator_score) AS m FROM generations WHERE validator_score IS NOT NULL"
    ).fetchone()
    assert abs(data["mean"] - float(direct["m"])) < 1e-3


def test_validator_stats_by_version(fresh_history):
    _insert_clip(fresh_history, clip_id="a", validator_score=0.9, validator_version="1.17.0-rc5")
    _insert_clip(fresh_history, clip_id="b", validator_score=0.8, validator_version="1.17.0-rc5")
    _insert_clip(fresh_history, clip_id="c", validator_score=0.5, validator_version="1.17.0-rc4")
    resp = client.get("/v1/system/validator-stats")
    assert resp.status_code == 200
    bv = resp.json()["by_version"]
    assert bv["1.17.0-rc5"]["count"] == 2
    assert bv["1.17.0-rc4"]["count"] == 1
    assert abs(bv["1.17.0-rc5"]["mean"] - 0.85) < 1e-3


def test_validator_stats_timeline_groups_scores(fresh_history):
    now = time.time()
    _insert_clip(fresh_history, clip_id="old-a", validator_score=0.2, created_at=now - 7200)
    _insert_clip(fresh_history, clip_id="new-a", validator_score=0.8, created_at=now - 700)
    _insert_clip(fresh_history, clip_id="new-b", validator_score=0.6, created_at=now - 680)
    resp = client.get("/v1/system/validator-stats?window=3h&timeline_buckets=12")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["timeline"]) == 12
    non_empty = [slot for slot in data["timeline"] if slot["count"]]
    assert [slot["count"] for slot in non_empty] == [1, 2]
    assert non_empty[0]["mean"] == pytest.approx(0.2, abs=1e-3)
    assert non_empty[1]["mean"] == pytest.approx(0.7, abs=1e-3)


def test_validator_stats_recent_runs_and_trend(fresh_history):
    now = time.time()
    for i, score in enumerate([0.40, 0.50, 0.60, 0.70, 0.80, 0.90]):
        _insert_clip(
            fresh_history,
            clip_id=f"trend-{i}",
            validator_score=score,
            validator_version="1.19.0-rc2",
            lora_applied_id="lora-A" if i >= 3 else None,
            ab_arm="candidate" if i >= 3 else "baseline",
            created_at=now - (6 - i) * 60,
        )
    fresh_history._conn.execute(
        """UPDATE generations
           SET validator_payload_json = ?, gen_config_json = ?, params_json = ?
           WHERE id = ?""",
        (
            '{"recommendation":"pass","tier3":{"verdict":"pass","score":0.9}}',
            '{"sampler":"cfg_pp","fast_stage1_steps":12,"cfg_scale":3.0,'
            '"scheduler_max_shift":1.85,"scheduler_base_shift":0.95}',
            '{"image_strength":0.8,"duration":1.96,"resolution":"2560x1440"}',
            "trend-5",
        ),
    )
    fresh_history._conn.commit()

    resp = client.get("/v1/system/validator-stats?window=24h")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["trend"]["latest_score"] == pytest.approx(0.90, abs=1e-3)
    assert data["trend"]["latest_delta"] == pytest.approx(0.10, abs=1e-3)
    assert data["trend"]["last5_count"] == 5
    assert data["trend"]["previous5_count"] == 1
    assert data["trend"]["last5_mean"] == pytest.approx(0.70, abs=1e-3)
    assert data["trend"]["previous5_mean"] == pytest.approx(0.40, abs=1e-3)
    assert data["trend"]["last5_delta"] == pytest.approx(0.30, abs=1e-3)

    recent = data["recent"]
    assert len(recent) == 6
    assert recent[0]["id"] == "trend-5"
    assert recent[0]["validator_score"] == pytest.approx(0.90, abs=1e-3)
    assert recent[0]["delta_vs_previous"] == pytest.approx(0.10, abs=1e-3)
    assert recent[0]["recommendation"] == "pass"
    assert recent[0]["tier3_verdict"] == "pass"
    assert recent[0]["lora_applied_id"] == "lora-A"
    assert recent[0]["ab_arm"] == "candidate"
    assert recent[0]["adjustment_summary"] == "cfg_pp · s1 12 · cfg 3.0 · shift 1.85"
    assert recent[0]["generation_config"]["sampler"] == "cfg_pp"
    assert recent[0]["generation_config"]["stage1_steps"] == 12
    assert recent[0]["generation_config"]["image_strength"] == pytest.approx(0.8)

    limited = client.get("/v1/system/validator-stats?window=24h&recent_limit=3")
    assert limited.status_code == 200, limited.text
    assert len(limited.json()["recent"]) == 3


# ---------------------------------------------------------------------------
# /v1/system/ab-status
# ---------------------------------------------------------------------------


def test_ab_status_admin_gated(fresh_history, monkeypatch):
    # check_api_key middleware requires the bearer to be in API_KEYS;
    # _require_admin then narrows to ADMIN_KEYS. Both keys must be in
    # API_KEYS to clear the outer middleware.
    monkeypatch.setattr(config, "API_KEYS", {"u-key", "a-key"})
    monkeypatch.setattr(config, "ADMIN_KEYS", {"a-key"})
    try:
        # Non-admin → 403
        resp = client.get("/v1/system/ab-status",
                          headers={"Authorization": "Bearer u-key"})
        assert resp.status_code == 403, resp.text
        # Admin → 200
        resp = client.get("/v1/system/ab-status",
                          headers={"Authorization": "Bearer a-key"})
        assert resp.status_code == 200, resp.text
        assert "experiments" in resp.json()
    finally:
        monkeypatch.setattr(config, "API_KEYS", set())
        monkeypatch.setattr(config, "ADMIN_KEYS", set())


def test_ab_status_experiment_shape(fresh_history):
    _insert_training_run(fresh_history, run_id="sft-cand-v1", lora_registry_id="sft-cand-v1")
    # Two cohorts on 30 distinct compositions each
    for i in range(30):
        _insert_clip(fresh_history, clip_id=f"b{i}", composition_id=f"comp-b-{i}",
                     ab_arm="baseline", validator_score=0.70)
        _insert_clip(fresh_history, clip_id=f"c{i}", composition_id=f"comp-c-{i}",
                     ab_arm="candidate", lora_applied_id="sft-cand-v1",
                     validator_score=0.80)
    resp = client.get("/v1/system/ab-status")
    assert resp.status_code == 200
    exps = resp.json()["experiments"]
    assert len(exps) == 1
    e = exps[0]
    assert e["candidate_lora"] == "sft-cand-v1"
    assert e["n_b"] == 30 and e["n_c"] == 30
    assert e["mean_b"] == pytest.approx(0.70, abs=1e-3)
    assert e["mean_c"] == pytest.approx(0.80, abs=1e-3)
    assert e["status"] == "in_progress"
    # With clear separation 0.70 vs 0.80 across 30 vs 30, p must be < 0.05
    assert e["p_value"] < 0.05


# ---------------------------------------------------------------------------
# /v1/system/training-runs
# ---------------------------------------------------------------------------


def test_training_runs_lists_recent_and_parses_metrics(fresh_history):
    _insert_training_run(fresh_history, run_id="r1",
                         eval_metrics_json='{"acc": 0.91, "loss": 0.42}')
    _insert_training_run(fresh_history, run_id="r2",
                         eval_metrics_json='{"acc": 0.88}', trained_at=time.time() - 100)
    resp = client.get("/v1/system/training-runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 2
    # r1 is newer → first
    assert runs[0]["run_id"] == "r1"
    assert runs[0]["eval_metrics"] == {"acc": 0.91, "loss": 0.42}
    # Cross-check with direct SQL row count
    n = fresh_history._conn.execute("SELECT COUNT(*) AS c FROM training_runs").fetchone()["c"]
    assert n == len(runs)


# ---------------------------------------------------------------------------
# /v1/system/preference-pairs-count
# ---------------------------------------------------------------------------


def test_preference_pairs_count_aggregates_by_source(fresh_history):
    for _ in range(3):
        _insert_pair(fresh_history, signal_source="user_retake")
    for _ in range(2):
        _insert_pair(fresh_history, signal_source="validator_pass")
    # one ancient pair (>7d old)
    _insert_pair(fresh_history, signal_source="composition_kept", age_s=10 * 86400.0)
    resp = client.get("/v1/system/preference-pairs-count")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 6
    assert data["by_source"]["user_retake"] == 3
    assert data["by_source"]["validator_pass"] == 2
    assert data["last_24h"] == 5  # the 10-day-old pair excluded
    assert data["last_7d"] == 5
    # Cross-check direct SQL
    direct_total = fresh_history._conn.execute(
        "SELECT COUNT(*) AS c FROM preference_pairs"
    ).fetchone()["c"]
    assert direct_total == data["total"]


# ---------------------------------------------------------------------------
# /v1/system/validator-failures
# ---------------------------------------------------------------------------


def test_validator_failures_ring_buffer_roundtrip(fresh_history):
    _validator_failures.record("job-1", "tier1", "RAFT cuda OOM")
    _validator_failures.record("job-2", "dispatch", "Traceback long..." + "x" * 800)
    resp = client.get("/v1/system/validator-failures?window=1h")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    # Newest first
    assert data["recent"][0]["job_id"] == "job-2"
    # 500-char truncation enforced
    assert len(data["recent"][0]["error"]) <= 500


# ---------------------------------------------------------------------------
# /v1/api-keys/me/training-opt-in (GET + POST)
# ---------------------------------------------------------------------------


def test_training_opt_in_get_default_unknown(fresh_history):
    # No API_KEYS → empty bearer; no metadata row → opted_in=False
    resp = client.get("/v1/api-keys/me/training-opt-in")
    assert resp.status_code == 200
    assert resp.json() == {"opted_in": False}


def test_training_opt_in_post_then_get_roundtrip(fresh_history, monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {"my-key"})
    try:
        # Initial GET → False (no metadata row yet)
        r0 = client.get("/v1/api-keys/me/training-opt-in",
                        headers={"Authorization": "Bearer my-key"})
        assert r0.json() == {"opted_in": False}
        # Toggle ON
        r1 = client.post("/v1/api-keys/me/training-opt-in",
                         headers={"Authorization": "Bearer my-key"},
                         json={"opted_in": True})
        assert r1.status_code == 200, r1.text
        assert r1.json() == {"opted_in": True}
        # GET reflects the change
        r2 = client.get("/v1/api-keys/me/training-opt-in",
                        headers={"Authorization": "Bearer my-key"})
        assert r2.json() == {"opted_in": True}
        # Toggle OFF
        r3 = client.post("/v1/api-keys/me/training-opt-in",
                         headers={"Authorization": "Bearer my-key"},
                         json={"opted_in": False})
        assert r3.json() == {"opted_in": False}
        # Cross-check via direct SQL
        row = fresh_history._conn.execute(
            "SELECT training_opt_in FROM api_key_metadata WHERE api_key_hash = ?",
            (_hash_key("my-key"),),
        ).fetchone()
        assert row["training_opt_in"] == 0
    finally:
        monkeypatch.setattr(config, "API_KEYS", set())


def test_training_opt_in_post_rejects_bad_body(fresh_history, monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {"my-key"})
    try:
        r = client.post("/v1/api-keys/me/training-opt-in",
                        headers={"Authorization": "Bearer my-key"},
                        json={"foo": "bar"})
        assert r.status_code == 422
    finally:
        monkeypatch.setattr(config, "API_KEYS", set())


# ---------------------------------------------------------------------------
# /v1/compositions/{comp_id}/clips (Stream B-portal)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_compositions(fresh_history, tmp_path: Path, monkeypatch):
    """CompositionStore pointed at the same fresh history.db."""
    from composition_store import CompositionStore
    cs = CompositionStore(db_path=fresh_history._db_path)
    monkeypatch.setattr(server_mod, "compositions", cs)
    yield cs
    cs._conn.close()


def _seed_composition(
    compositions, history, *, comp_id: str, owner_key: str,
    clips_payload: list[dict] | None = None,
):
    """Insert a compositions row directly + best-effort composition_clips."""
    from history_store import _hash_key
    import json as _json
    now = time.time()
    data = {"clips": clips_payload or []}
    compositions._conn.execute(
        """INSERT OR REPLACE INTO compositions
           (id, api_key_hash, name, data, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (comp_id, _hash_key(owner_key), "Test Comp",
         _json.dumps(data), now, now),
    )
    compositions._conn.commit()


def test_compositions_clips_privacy_gate(fresh_history, fresh_compositions, monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {"bearer-A", "bearer-B"})
    try:
        _seed_composition(fresh_compositions, fresh_history,
                          comp_id="comp-A", owner_key="bearer-A")
        # Bearer B queries A's comp → 403 (not 404 — must NOT leak existence
        # via 200 vs 404; 403 is the explicit privacy-gate signal).
        r = client.get(
            "/v1/compositions/comp-A/clips",
            headers={"Authorization": "Bearer bearer-B"},
        )
        assert r.status_code == 403, r.text
        # Owner sees their own comp.
        r2 = client.get(
            "/v1/compositions/comp-A/clips",
            headers={"Authorization": "Bearer bearer-A"},
        )
        assert r2.status_code == 200, r2.text
        # Unknown comp_id → 404 regardless of bearer.
        r3 = client.get(
            "/v1/compositions/comp-does-not-exist/clips",
            headers={"Authorization": "Bearer bearer-A"},
        )
        assert r3.status_code == 404
    finally:
        monkeypatch.setattr(config, "API_KEYS", set())


def test_compositions_clips_response_shape(fresh_history, fresh_compositions):
    # Auth disabled (API_KEYS empty); seed two clips with mixed
    # historyId / storage_uri lineage.
    _insert_clip(
        fresh_history, clip_id="hist-1", validator_score=0.82,
        composition_id="comp-X",
    )
    _insert_clip(
        fresh_history, clip_id="hist-2", validator_score=0.41,
        composition_id="comp-X",
    )
    # Decorate hist-1 with a recommendation payload to verify it surfaces.
    fresh_history._conn.execute(
        "UPDATE generations SET validator_payload_json = ? WHERE id = ?",
        ('{"recommendation": "pass"}', "hist-1"),
    )
    fresh_history._conn.commit()
    _seed_composition(fresh_compositions, fresh_history,
                      comp_id="comp-X", owner_key="bearer-X")
    # Record lineage rows so the JOIN path returns clips.
    fresh_history.record_composition_clips(
        "comp-X",
        [{"historyId": "hist-1"}, {"historyId": "hist-2"}],
    )

    r = client.get("/v1/compositions/comp-X/clips")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["composition_id"] == "comp-X"
    assert data["total_clips"] == 2
    assert isinstance(data["exported_at"], (int, float))
    clips = data["clips"]
    by_pos = {c["position_in_comp"]: c for c in clips}
    assert by_pos[0]["historyId"] == "hist-1"
    assert by_pos[0]["validator_score"] == pytest.approx(0.82, abs=1e-3)
    assert by_pos[0]["validator_recommendation"] == "pass"
    assert by_pos[0]["thumbnail_url"] == "/v2/history/hist-1/thumbnail"
    assert by_pos[0]["kept"] is True
    assert by_pos[0]["deprecated_reason"] is None
    assert by_pos[1]["validator_score"] == pytest.approx(0.41, abs=1e-3)
