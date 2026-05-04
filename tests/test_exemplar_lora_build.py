"""Tests for v1.19.0 / Phase 3 prereq — async LoRA build path.

The real ``--execute`` LoRA build is 10-15 GPU-hours; running it
synchronously inside the request handler would block the ASGI worker
and starve the entire job queue. These tests verify the spawn-and-go
contract: the endpoint returns 202 within milliseconds, registers a
PID for cancellation, and a small monitor task flips
``training_runs.status`` when the subprocess exits.

NO real GPU work happens here — the ``subprocess.Popen`` spawn is
patched with a fast mock that exits immediately with a chosen
returncode (0 / non-zero / SIGTERM-equivalent).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import config

config.GPU_DEVICES = []
config.API_KEYS = set()
config.ADMIN_KEYS = set()

from fastapi.testclient import TestClient  # noqa: E402

from history_store import HistoryStore, _hash_key  # noqa: E402

import server as server_mod  # noqa: E402
from server import app  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_lora_from_exemplars as builder  # noqa: E402


client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch, tmp_path: Path):
    db = tmp_path / "history.db"
    fresh = HistoryStore(db_path=db)
    monkeypatch.setattr(server_mod, "history", fresh)
    server_mod._lora_build_processes.clear()
    yield fresh
    try:
        fresh._conn.close()
    except Exception:
        pass


class _FakeProc:
    """Mock subprocess.Popen with a configurable returncode + lifecycle."""

    def __init__(self, returncode: int = 0, *, hold_until_terminate: bool = False) -> None:
        self._returncode = returncode
        self._hold = hold_until_terminate
        self._terminated = False
        self._poll_count = 0
        self.pid = 99999

    def poll(self) -> int | None:
        self._poll_count += 1
        if self._hold and not self._terminated:
            return None
        if self._hold and self._terminated:
            return 143  # SIGTERM convention
        return self._returncode

    def terminate(self) -> None:
        self._terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return self._returncode


def _seed_25_clips(history: HistoryStore, monkeypatch, tmp_path: Path) -> None:
    """Stage 25 valid exemplar clips to clear the build threshold."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(config, "UPLOAD_DIR", upload_dir)
    for i in range(25):
        clip_id = f"clip_{i}"
        upload_id = f"{i:032x}"
        (upload_dir / upload_id).write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100)
        history._conn.execute(
            """INSERT OR REPLACE INTO generations
               (id, api_key_hash, job_type, prompt, model, width, height,
                turbo, status, result_uri, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                clip_id, _hash_key("k"), "audio-to-video",
                "a cinematic shot", "ltx-2-3-fast", 512, 512, 0,
                "completed", f"storage://{upload_id}", time.time(),
            ),
        )
    history._conn.commit()
    create = client.post("/v1/exemplar-sets", json={"set_id": "asyncset"})
    assert create.status_code == 201, create.text
    for i in range(25):
        r = client.post(
            "/v1/exemplar-sets/asyncset/members",
            json={"clip_id": f"clip_{i}"},
        )
        assert r.status_code == 201, r.text


def test_execute_returns_202_within_500ms(
    _isolate_db: HistoryStore, tmp_path: Path, monkeypatch,
) -> None:
    """The ``--execute`` path must NOT block the ASGI worker for a
    long-running build. Instead, the endpoint spawns the orchestrator
    out-of-band and returns 202 immediately. The mock subprocess exits
    cleanly, so the monitor will eventually flip status to completed —
    but the response itself must come back well under 500ms.
    """
    monkeypatch.setattr(builder, "TRAINING_RUNS_DIR", tmp_path / "training_runs")
    _seed_25_clips(_isolate_db, monkeypatch, tmp_path)

    fake_procs: list[_FakeProc] = []

    def _fake_spawn(**kwargs: Any) -> Any:
        proc = _FakeProc(returncode=0, hold_until_terminate=True)
        fake_procs.append(proc)
        return proc

    monkeypatch.setattr(server_mod, "_spawn_lora_build_subprocess", _fake_spawn)

    t0 = time.perf_counter()
    resp = client.post(
        "/v1/exemplar-sets/asyncset/build-lora",
        json={"dry_run": False, "rank": 16, "steps": 100},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 202, resp.text
    assert elapsed_ms < 500.0, (
        f"build endpoint took {elapsed_ms:.1f}ms — must be <500ms"
    )
    body = resp.json()
    assert "training_run_id" in body
    assert body["status"] == "running"
    assert body["scheduled_for"] is not None
    assert body["expected_eta_hours"] > 0
    # The monitor registered the PID so cancel can SIGTERM it.
    assert body["training_run_id"] in server_mod._lora_build_processes


def test_status_flips_to_completed_on_clean_exit(
    _isolate_db: HistoryStore, tmp_path: Path, monkeypatch,
) -> None:
    """When the subprocess exits with rc=0 the monitor flips the
    ``training_runs.status`` row from 'running' to 'completed'.
    """
    monkeypatch.setattr(builder, "TRAINING_RUNS_DIR", tmp_path / "training_runs")
    _seed_25_clips(_isolate_db, monkeypatch, tmp_path)

    def _fake_spawn(**kwargs: Any) -> Any:
        return _FakeProc(returncode=0)

    monkeypatch.setattr(server_mod, "_spawn_lora_build_subprocess", _fake_spawn)

    resp = client.post(
        "/v1/exemplar-sets/asyncset/build-lora",
        json={"dry_run": False, "rank": 16, "steps": 100},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["training_run_id"]

    # TestClient ends its per-request event loop before the monitor
    # task sees the proc exit, so drive the monitor directly here.
    proc = _FakeProc(returncode=0)
    asyncio.run(server_mod._monitor_lora_build_subprocess(
        run_id=run_id, proc=proc, history_ref=_isolate_db,
    ))

    row = _isolate_db._conn.execute(
        "SELECT status FROM training_runs WHERE run_id = ?", (run_id,),
    ).fetchone()
    assert row["status"] == "completed", f"expected completed, got {row['status']}"
    assert run_id not in server_mod._lora_build_processes


def test_status_flips_to_cancelled_on_cancel_post(
    _isolate_db: HistoryStore, tmp_path: Path, monkeypatch,
) -> None:
    """POST /cancel SIGTERMs the registered subprocess and flips
    ``training_runs.status`` to 'cancelled'.
    """
    monkeypatch.setattr(builder, "TRAINING_RUNS_DIR", tmp_path / "training_runs")
    _seed_25_clips(_isolate_db, monkeypatch, tmp_path)

    captured: dict[str, _FakeProc] = {}

    def _fake_spawn(**kwargs: Any) -> Any:
        proc = _FakeProc(hold_until_terminate=True)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(server_mod, "_spawn_lora_build_subprocess", _fake_spawn)

    resp = client.post(
        "/v1/exemplar-sets/asyncset/build-lora",
        json={"dry_run": False, "rank": 16, "steps": 100},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["training_run_id"]
    assert run_id in server_mod._lora_build_processes

    cancel = client.post(f"/v1/exemplar-sets/asyncset/build/{run_id}/cancel")
    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["status"] == "cancelled"

    # The proc.terminate() was invoked.
    assert captured["proc"]._terminated is True

    # DB row now reflects 'cancelled'.
    row = _isolate_db._conn.execute(
        "SELECT status FROM training_runs WHERE run_id = ?", (run_id,),
    ).fetchone()
    assert row["status"] == "cancelled"


def test_status_flips_to_failed_on_nonzero_exit(
    _isolate_db: HistoryStore, tmp_path: Path, monkeypatch,
) -> None:
    """When the subprocess exits with a non-zero, non-SIGTERM rc the
    monitor flips status to 'failed'.
    """
    monkeypatch.setattr(builder, "TRAINING_RUNS_DIR", tmp_path / "training_runs")
    _seed_25_clips(_isolate_db, monkeypatch, tmp_path)

    def _fake_spawn(**kwargs: Any) -> Any:
        return _FakeProc(returncode=2)

    monkeypatch.setattr(server_mod, "_spawn_lora_build_subprocess", _fake_spawn)

    resp = client.post(
        "/v1/exemplar-sets/asyncset/build-lora",
        json={"dry_run": False, "rank": 16, "steps": 100},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["training_run_id"]

    # Drive the monitor directly — see test_status_flips_to_completed.
    proc = _FakeProc(returncode=2)
    asyncio.run(server_mod._monitor_lora_build_subprocess(
        run_id=run_id, proc=proc, history_ref=_isolate_db,
    ))

    row = _isolate_db._conn.execute(
        "SELECT status FROM training_runs WHERE run_id = ?", (run_id,),
    ).fetchone()
    assert row["status"] == "failed", f"expected failed, got {row['status']}"
