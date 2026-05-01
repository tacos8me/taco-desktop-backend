"""Tests for v1.19.0-rc2 operator-depth endpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import config

config.GPU_DEVICES = []
config.API_KEYS = set()

from fastapi.testclient import TestClient  # noqa: E402

from history_store import CURRENT_SCHEMA_VERSION, HistoryStore, _hash_key  # noqa: E402
import server as server_mod  # noqa: E402
from server import app  # noqa: E402


client = TestClient(app)


@pytest.fixture()
def fresh_history(tmp_path: Path, monkeypatch):
    db = tmp_path / "history.db"
    h = HistoryStore(db_path=db)
    monkeypatch.setattr(server_mod, "history", h)
    yield h
    h._conn.close()


@pytest.fixture(autouse=True)
def _reset_rc2_globals():
    old_api = set(config.API_KEYS)
    old_admin = set(config.ADMIN_KEYS)
    old_metrics_history = dict(server_mod._metrics_history)
    old_counters = dict(server_mod._validator_dispatch_counter)
    try:
        yield
    finally:
        config.API_KEYS = old_api
        config.ADMIN_KEYS = old_admin
        server_mod._metrics_history.clear()
        server_mod._metrics_history.update(old_metrics_history)
        server_mod._validator_dispatch_counter.clear()
        server_mod._validator_dispatch_counter.update(old_counters)


def _save_job(
    history: HistoryStore,
    job_id: str,
    *,
    api_key: str = "bearer-A",
    parent_clip_id: str | None = None,
    shot_uuid: str | None = None,
    validator_score: float | None = 0.71,
) -> None:
    now = time.time()
    history.save(
        job_id=job_id,
        api_key=api_key,
        job_type="text-to-video",
        prompt=f"prompt {job_id}",
        model="ltx-2-3-fast",
        width=512,
        height=512,
        turbo=False,
        status="completed",
        result_uri=f"storage://{job_id}",
        result_bytes=None,
        created_at=now - 2,
        completed_at=now,
        raw_request={"prompt": f"prompt {job_id}", "duration": 5},
        gen_config_snapshot={"sampler": "euler"},
        parent_clip_id=parent_clip_id,
        shot_uuid=shot_uuid,
        shot_config_key="shot-key",
        composition_id="comp-1",
        lora_applied_id="lora-1",
        lora_applied_strength=0.3,
        ab_arm="candidate",
    )
    history._conn.execute(
        """UPDATE generations
           SET validator_score = ?,
               validator_payload_json = ?,
               validator_version = ?,
               motion_intent = ?
           WHERE id = ?""",
        (
            validator_score,
            json.dumps({"composite_score": validator_score, "recommendation": "pass"}),
            "1.19.0-rc1",
            "slow push",
            job_id,
        ),
    )
    history._conn.commit()


def test_schema_state_counts_match_sql(fresh_history):
    _save_job(fresh_history, "job-schema")
    resp = client.get("/v1/system/schema-state")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["schema_version"] == CURRENT_SCHEMA_VERSION
    direct = fresh_history._conn.execute(
        "SELECT COUNT(*) AS n FROM generations"
    ).fetchone()["n"]
    assert data["tables"]["generations"]["row_count"] == direct
    assert "idx_api_key_hash" in data["indexes"]
    assert isinstance(data["db_size_bytes"], int)


def test_metrics_history_returns_per_minute_samples(fresh_history):
    server_mod._validator_dispatch_counter["success"] = 3
    server_mod._validator_dispatch_counter["failure"] = 1
    resp = client.get(
        "/v1/system/metrics/history?window=24h&metric=validator_dispatch.success"
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["metric"] == "validator_dispatch.success"
    assert data["samples"]
    assert data["samples"][-1]["value"] == 3.0


def test_system_job_detail_privacy_gate_and_admin_override(fresh_history, monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {"bearer-A", "bearer-B", "admin-key"})
    monkeypatch.setattr(config, "ADMIN_KEYS", {"admin-key"})
    _save_job(fresh_history, "job-owned", api_key="bearer-A")

    denied = client.get(
        "/v1/system/jobs/job-owned",
        headers={"Authorization": "Bearer bearer-B"},
    )
    assert denied.status_code == 403, denied.text

    owner = client.get(
        "/v1/system/jobs/job-owned",
        headers={"Authorization": "Bearer bearer-A"},
    )
    assert owner.status_code == 200, owner.text
    assert owner.json()["params"]["prompt"] == "prompt job-owned"

    admin = client.get(
        "/v1/system/jobs/job-owned",
        headers={"Authorization": "Bearer admin-key"},
    )
    assert admin.status_code == 200, admin.text
    assert admin.json()["lineage"]["lora_applied_id"] == "lora-1"


def test_system_job_detail_unknown_404(fresh_history):
    resp = client.get("/v1/system/jobs/nope")
    assert resp.status_code == 404


def test_system_job_detail_lineage_cycle_terminates(fresh_history):
    _save_job(fresh_history, "job-a", parent_clip_id="job-b", shot_uuid="shot-1")
    _save_job(fresh_history, "job-b", parent_clip_id="job-a", shot_uuid="shot-1")
    resp = client.get("/v1/system/jobs/job-a")
    assert resp.status_code == 200, resp.text
    lineage = resp.json()["lineage"]
    assert len(lineage["ancestors"]) == 1
    assert lineage["ancestors"][0]["id"] == "job-b"
    assert len(lineage["descendants"]) == 1
    assert lineage["descendants"][0]["id"] == "job-b"
    assert len(lineage["siblings"]) == 1


def test_log_stream_accepts_token_and_terminates_subprocess(monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {"admin-key"})
    monkeypatch.setattr(config, "ADMIN_KEYS", {"admin-key"})
    created = []

    class FakeStdout:
        def __init__(self):
            self.lines = [
                json.dumps({
                    "__REALTIME_TIMESTAMP": "1710000000000000",
                    "_SYSTEMD_UNIT": "taco-backend.service",
                    "PRIORITY": "3",
                    "MESSAGE": "boom",
                }) + "\n"
            ]

        def __iter__(self):
            return self

        def __next__(self):
            if self.lines:
                return self.lines.pop(0)
            raise StopIteration

    class FakeProc:
        def __init__(self, *args, **kwargs):
            self.stdout = FakeStdout()
            self.terminated = False
            self.killed = False
            created.append(self)

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.terminated = True

        def wait(self, timeout=None):
            self.terminated = True
            return 0

    monkeypatch.setattr(server_mod.subprocess, "Popen", FakeProc)
    resp = client.get(
        "/v1/system/logs/stream?source=taco-backend&min_severity=INFO&token=admin-key"
    )
    assert resp.status_code == 200, resp.text
    assert "data:" in resp.text
    assert '"severity": "ERROR"' in resp.text
    assert created and created[0].terminated


def test_log_stream_rejects_unknown_source(fresh_history):
    resp = client.get("/v1/system/logs/stream?source=bad")
    assert resp.status_code == 422


def test_bearers_admin_shape_and_aggregates(fresh_history, monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {"bearer-A", "admin-key"})
    monkeypatch.setattr(config, "ADMIN_KEYS", {"admin-key"})
    key_hash = _hash_key("bearer-A")
    now = time.time()
    fresh_history._conn.execute(
        """INSERT OR REPLACE INTO api_key_metadata
           (api_key_hash, training_opt_in, tier, notes, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (key_hash, 1, "pro", "primary", now - 100, now - 50),
    )
    fresh_history._conn.commit()
    _save_job(fresh_history, "job-bearer-1", api_key="bearer-A", validator_score=0.8)
    _save_job(fresh_history, "job-bearer-2", api_key="bearer-A", validator_score=0.6)

    denied = client.get(
        "/v1/system/bearers",
        headers={"Authorization": "Bearer bearer-A"},
    )
    assert denied.status_code == 403

    resp = client.get(
        "/v1/system/bearers",
        headers={"Authorization": "Bearer admin-key"},
    )
    assert resp.status_code == 200, resp.text
    bearer = resp.json()["bearers"][0]
    assert bearer["api_key_hash"] == key_hash
    assert bearer["label"] == "primary"
    assert bearer["training_opt_in"] is True
    assert bearer["total_clips"] == 2
    assert bearer["mean_validator_score"] == pytest.approx(0.7, abs=1e-3)
    assert bearer["ab_arm_distribution"]["candidate"] == 2


def test_bearer_training_opt_in_admin_persists(fresh_history, monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {"bearer-A", "admin-key"})
    monkeypatch.setattr(config, "ADMIN_KEYS", {"admin-key"})
    key_hash = _hash_key("bearer-A")

    denied = client.post(
        f"/v1/system/bearers/{key_hash}/training-opt-in",
        headers={"Authorization": "Bearer bearer-A"},
        json={"opted_in": False},
    )
    assert denied.status_code == 403

    resp = client.post(
        f"/v1/system/bearers/{key_hash}/training-opt-in",
        headers={"Authorization": "Bearer admin-key"},
        json={"opted_in": False},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["opted_in"] is False
    row = fresh_history._conn.execute(
        "SELECT training_opt_in FROM api_key_metadata WHERE api_key_hash = ?",
        (key_hash,),
    ).fetchone()
    assert row["training_opt_in"] == 0


def test_sidecars_shape_uses_health_pid_and_vram(monkeypatch):
    monkeypatch.setattr(config, "DUAL_GPU_LTX", True)
    monkeypatch.setattr(server_mod, "_systemctl_main_pid", lambda unit: 1234)
    monkeypatch.setattr(server_mod, "_gpu_memory_by_pid", lambda: {1234: 2048})

    async def fake_health(url):
        if not url:
            return False, None, "no_url", None
        return True, "ready", None, "test-model"

    monkeypatch.setattr(server_mod, "_sidecar_health", fake_health)
    resp = client.get("/v1/system/sidecars")
    assert resp.status_code == 200, resp.text
    sidecars = {s["name"]: s for s in resp.json()["sidecars"]}
    assert sidecars["ltx-sidecar"]["configured"] is True
    assert sidecars["ltx-sidecar"]["active"] is True
    assert sidecars["ltx-sidecar"]["pid"] == 1234
    assert sidecars["ltx-sidecar"]["model_loaded"] == "test-model"
    assert sidecars["ltx-sidecar"]["vram_resident_gb"] == 2.0


def test_storage_counts_configured_directories(fresh_history, tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    thumb_dir = tmp_path / "thumbs"
    artifacts_dir = tmp_path / "validator_artifacts"
    upload_dir.mkdir()
    thumb_dir.mkdir()
    artifacts_dir.mkdir()
    (upload_dir / "a.bin").write_bytes(b"a" * 10)
    (upload_dir / "b.bin").write_bytes(b"b" * 5)
    (thumb_dir / "t.jpg").write_bytes(b"t" * 3)
    (artifacts_dir / "pose.npz").write_bytes(b"p" * 7)
    monkeypatch.setattr(config, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(config, "THUMBNAIL_DIR", thumb_dir)
    monkeypatch.setattr(config, "VALIDATOR_ARTIFACTS_DIR", artifacts_dir)

    resp = client.get("/v1/system/storage")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["uploads_bytes"] == 15
    assert data["uploads_count"] == 2
    assert data["thumbnails_bytes"] == 3
    assert data["thumbnails_count"] == 1
    assert data["validator_artifacts_bytes"] == 7
    assert data["validator_artifacts_count"] == 1


def test_storage_auth_gated_when_api_keys_configured(fresh_history, monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {"bearer-A"})
    resp = client.get("/v1/system/storage")
    assert resp.status_code == 401
