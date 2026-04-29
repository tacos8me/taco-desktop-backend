"""Tests for v1.17.0-rc1 schema v3 migration + composition lineage + retake provenance."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import history_store
from history_store import (
    CURRENT_SCHEMA_VERSION,
    HistoryStore,
    _hash_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _make_v2_db(path: Path) -> None:
    """Build a database that looks like the pre-v3 (v2) schema."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE generations (
            id TEXT PRIMARY KEY,
            api_key_hash TEXT NOT NULL,
            job_type TEXT NOT NULL,
            prompt TEXT NOT NULL,
            model TEXT,
            width INTEGER,
            height INTEGER,
            turbo INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            result_uri TEXT,
            thumbnail_uri TEXT,
            created_at REAL NOT NULL,
            completed_at REAL,
            error TEXT,
            params_json TEXT,
            gen_config_json TEXT,
            seed INTEGER,
            enhanced_prompt TEXT
        );
        CREATE INDEX idx_api_key_hash ON generations(api_key_hash, created_at DESC);
        """
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


def test_schema_v3_migration_runs_clean(tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    cols = _table_columns(store._conn, "generations")
    expected_new_cols = {
        "validator_score", "validator_payload_json", "validator_version",
        "validator_artifact_uri", "parent_clip_id", "shot_uuid",
        "shot_config_key", "composition_id", "lora_applied_id",
        "lora_applied_strength", "prompt_embedding",
    }
    assert expected_new_cols.issubset(cols)

    tables = _tables(store._conn)
    for required in (
        "composition_clips", "validator_runs", "preference_pairs",
        "training_runs", "api_key_metadata",
    ):
        assert required in tables, f"missing table: {required}"

    user_version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    # v1.18.0-rc1 bumped CURRENT_SCHEMA_VERSION 3→4; the v3 surface this
    # test asserts is still present, but the DB ladders all the way up.
    assert user_version == CURRENT_SCHEMA_VERSION
    assert CURRENT_SCHEMA_VERSION >= 3


def test_schema_v3_migration_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    # Re-running _migrate must be a no-op.
    store._migrate()
    store._migrate()
    user_version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == CURRENT_SCHEMA_VERSION
    # Duplicate-create-table guards work.
    cols = _table_columns(store._conn, "generations")
    assert "validator_score" in cols


def test_schema_v3_existing_v2_db_upgrades(tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    _make_v2_db(db)

    # Insert a row using the v2 column set.
    conn = sqlite3.connect(str(db))
    conn.execute(
        """INSERT INTO generations (id, api_key_hash, job_type, prompt, status,
            created_at, params_json) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("legacy_1", _hash_key("legacy"), "text-to-video", "a sunset",
         "completed", time.time(), json.dumps({"k": "v"})),
    )
    conn.commit()
    conn.close()

    store = HistoryStore(db_path=db)
    user_version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    # Ladder migrates v2 → CURRENT_SCHEMA_VERSION (4 as of v1.18.0-rc1),
    # touching every intermediate version's ALTERs in order.
    assert user_version == CURRENT_SCHEMA_VERSION

    row = store._conn.execute(
        "SELECT id, prompt, validator_score, parent_clip_id, params_json "
        "FROM generations WHERE id = 'legacy_1'"
    ).fetchone()
    assert row["prompt"] == "a sunset"
    assert row["validator_score"] is None
    assert row["parent_clip_id"] is None
    assert json.loads(row["params_json"]) == {"k": "v"}


# ---------------------------------------------------------------------------
# api_key_metadata seeding
# ---------------------------------------------------------------------------


def test_api_key_metadata_seeded_from_api_keys_file(tmp_path: Path, monkeypatch) -> None:
    # Place a fake .api_keys next to the history_store module.
    module_dir = Path(history_store.__file__).parent
    fake_keys_file = module_dir / ".api_keys.test"
    fake_keys_file.write_text("# header comment\nkeyA\nkeyB\n\n")

    # Redirect _maybe_seed_api_key_metadata to read our fake file by patching
    # Path(__file__).parent / ".api_keys" — easiest is monkeypatching the
    # module-level Path reference. Simpler: temporarily move the real .api_keys
    # out of the way and write our test content as ".api_keys".
    real_keys_file = module_dir / ".api_keys"
    backup = None
    if real_keys_file.exists():
        backup = real_keys_file.read_text()
    try:
        real_keys_file.write_text("# header\nkeyA\nkeyB\n")
        db = tmp_path / "history.db"
        store = HistoryStore(db_path=db)
        rows = store._conn.execute(
            "SELECT api_key_hash, training_opt_in, tier FROM api_key_metadata"
        ).fetchall()
        hashes = {r["api_key_hash"] for r in rows}
        assert _hash_key("keyA") in hashes
        assert _hash_key("keyB") in hashes
        assert all(r["training_opt_in"] == 1 for r in rows)
        assert all(r["tier"] == "pro" for r in rows)
    finally:
        if backup is not None:
            real_keys_file.write_text(backup)
        else:
            real_keys_file.unlink(missing_ok=True)
        fake_keys_file.unlink(missing_ok=True)


def test_api_key_metadata_seed_idempotent(tmp_path: Path) -> None:
    module_dir = Path(history_store.__file__).parent
    real_keys_file = module_dir / ".api_keys"
    backup = real_keys_file.read_text() if real_keys_file.exists() else None
    try:
        real_keys_file.write_text("keyA\n")
        db = tmp_path / "history.db"
        store = HistoryStore(db_path=db)
        first_count = store._conn.execute(
            "SELECT COUNT(*) FROM api_key_metadata"
        ).fetchone()[0]
        # Re-running _maybe_seed_api_key_metadata must not double-insert.
        store._maybe_seed_api_key_metadata()
        second_count = store._conn.execute(
            "SELECT COUNT(*) FROM api_key_metadata"
        ).fetchone()[0]
        assert first_count == second_count == 1
    finally:
        if backup is not None:
            real_keys_file.write_text(backup)
        else:
            real_keys_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Composition lineage
# ---------------------------------------------------------------------------


def test_composition_clips_lineage_write_on_export(tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    clips = [
        {"historyId": "h_a", "duration": 4.0},
        {"historyId": "h_b", "duration": 4.0},
        {"historyId": "h_c", "duration": 4.0},
    ]
    n = store.record_composition_clips("comp_1", clips)
    assert n == 3
    rows = store._conn.execute(
        "SELECT comp_id, clip_history_id, position, was_final "
        "FROM composition_clips WHERE comp_id = ? ORDER BY position",
        ("comp_1",),
    ).fetchall()
    assert [r["clip_history_id"] for r in rows] == ["h_a", "h_b", "h_c"]
    assert [r["position"] for r in rows] == [0, 1, 2]
    assert all(r["was_final"] == 1 for r in rows)


def test_composition_clips_lineage_handles_flash_inserts(tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    clips = [
        {"historyId": "h_real", "duration": 4.0},
        {"storage_uri": "storage://flash_uuid", "duration": 1.5},  # flash insert
    ]
    n = store.record_composition_clips("comp_2", clips)
    assert n == 2
    flash_row = store._conn.execute(
        "SELECT clip_history_id, was_final FROM composition_clips "
        "WHERE comp_id = ? AND position = 1",
        ("comp_2",),
    ).fetchone()
    assert flash_row["clip_history_id"] is None
    assert flash_row["was_final"] == 1


# ---------------------------------------------------------------------------
# Retake parent_clip_id
# ---------------------------------------------------------------------------


def test_retake_parent_clip_id_populated(tmp_path: Path) -> None:
    """Source clip lives in history → find_id_by_result_uri returns its id."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    # Insert a parent generation row.
    store._conn.execute(
        """INSERT INTO generations
           (id, api_key_hash, job_type, prompt, status, result_uri,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("parent_clip_a", _hash_key("k"), "audio-to-video", "p",
         "completed", "storage://upload_a", time.time()),
    )
    store._conn.commit()
    found = store.find_id_by_result_uri("storage://upload_a")
    assert found == "parent_clip_a"

    # Save a retake row that links to it.
    store.save(
        job_id="retake_1", api_key="k", job_type="retake", prompt="p2",
        model=None, width=0, height=0, turbo=False, status="completed",
        result_uri="storage://upload_b", result_bytes=None,
        created_at=time.time(), completed_at=time.time(),
        parent_clip_id=found,
    )
    row = store._conn.execute(
        "SELECT parent_clip_id FROM generations WHERE id = 'retake_1'"
    ).fetchone()
    assert row["parent_clip_id"] == "parent_clip_a"


def test_retake_unknown_video_uri_no_parent_link(tmp_path: Path) -> None:
    """video_uri with no matching history row → parent_clip_id stays NULL, no error."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    found = store.find_id_by_result_uri("storage://does_not_exist")
    assert found is None

    store.save(
        job_id="retake_orphan", api_key="k", job_type="retake", prompt="p",
        model=None, width=0, height=0, turbo=False, status="completed",
        result_uri="storage://upload_c", result_bytes=None,
        created_at=time.time(), completed_at=time.time(),
        parent_clip_id=found,  # i.e. None
    )
    row = store._conn.execute(
        "SELECT parent_clip_id FROM generations WHERE id = 'retake_orphan'"
    ).fetchone()
    assert row["parent_clip_id"] is None


# ---------------------------------------------------------------------------
# validator_runs / preference_pairs
# ---------------------------------------------------------------------------


def test_validator_runs_unique_index_video_sha_version(tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    now = time.time()
    store._conn.execute(
        """INSERT INTO validator_runs
           (run_id, video_uri, video_sha256, payload_json, latency_s,
            validator_version, ran_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("run_1", "storage://a", "sha_abc", "{}", 0.5, "1.17.0", now),
    )
    store._conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """INSERT INTO validator_runs
               (run_id, video_uri, video_sha256, payload_json, latency_s,
                validator_version, ran_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("run_2", "storage://a", "sha_abc", "{}", 0.6, "1.17.0", now),
        )
        store._conn.commit()


def test_preference_pairs_table_works(tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    now = time.time()
    store._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source,
            signal_strength, used_in_training_run_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("chosen_a", "rejected_b", "user_retake", 0.9, None, now),
    )
    store._conn.commit()
    row = store._conn.execute(
        "SELECT chosen_clip_id, rejected_clip_id, signal_source, signal_strength "
        "FROM preference_pairs"
    ).fetchone()
    assert row["chosen_clip_id"] == "chosen_a"
    assert row["rejected_clip_id"] == "rejected_b"
    assert row["signal_source"] == "user_retake"
    assert row["signal_strength"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# on_complete callback wiring
# ---------------------------------------------------------------------------


def test_on_complete_callback_invoked() -> None:
    """worker_loop must call on_complete(job) exactly once after a job
    reaches a terminal state. Mirrors how the server registers
    _decr_queue_on_complete to release per-key counters."""
    import asyncio
    from job_queue import (
        Job, JobStatus, JobType, JobStore, worker_loop,
    )
    from upload_store import UploadStore

    job_store = JobStore()
    queue: asyncio.Queue[str] = asyncio.Queue()
    lock = asyncio.Lock()

    job = Job(id="j_done", type=JobType.TEXT_TO_VIDEO, params={})
    job_store.add(job)

    invocations: list[Job] = []

    def on_complete(j: Job) -> None:
        invocations.append(j)

    async def fake_dispatch(_job: Job) -> bytes:
        return b"\x00" * 8

    async def run() -> None:
        # tmp UploadStore — points at a freshly-mkdir'd dir
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            uploads = UploadStore(base_dir=Path(td))
            await queue.put(job.id)
            worker = asyncio.create_task(
                worker_loop(
                    job_store, queue, lock, fake_dispatch, uploads,
                    history=None, on_complete=on_complete,
                )
            )
            # Wait until the queue drains.
            await queue.join()
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    asyncio.run(run())
    assert len(invocations) == 1
    assert invocations[0].id == "j_done"
    assert invocations[0].status == JobStatus.COMPLETED
