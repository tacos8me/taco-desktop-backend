"""Tests for v1.18.0-rc1 schema v4 migration — keystone for Phase B + Phase C.

Schema v4 is purely additive:
  - ``generations`` gains ``motion_intent`` + ``embedding_model_version``
  - ``preference_pairs`` gains ``validator_version`` + 2 new indexes
    (``idx_pp_validator_version``, ``idx_pp_unique_pair_source`` UNIQUE)
  - ``training_runs`` gains 5 reproducibility columns
    (``training_seed``, ``hyperparams_json``, ``dataset_snapshot_path``,
    ``code_sha``, ``validator_version_at_train``)
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

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


def _indexes(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    return {row[1] for row in rows}


def _make_v3_db(path: Path) -> None:
    """Build a database that looks like the post-rc1 (v3) schema."""
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
            enhanced_prompt TEXT,
            validator_score REAL,
            validator_payload_json TEXT,
            validator_version TEXT,
            validator_artifact_uri TEXT,
            parent_clip_id TEXT,
            shot_uuid TEXT,
            shot_config_key TEXT,
            composition_id TEXT,
            lora_applied_id TEXT,
            lora_applied_strength REAL,
            prompt_embedding BLOB
        );
        CREATE INDEX idx_api_key_hash ON generations(api_key_hash, created_at DESC);
        CREATE INDEX idx_gen_shot_config_key ON generations(shot_config_key);
        CREATE INDEX idx_gen_parent_clip_id ON generations(parent_clip_id);
        CREATE INDEX idx_gen_composition_id ON generations(composition_id);

        CREATE TABLE composition_clips (
            comp_id TEXT NOT NULL,
            clip_history_id TEXT,
            position INTEGER NOT NULL,
            was_final INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            PRIMARY KEY (comp_id, clip_history_id, position)
        );
        CREATE INDEX idx_comp_clips_clip ON composition_clips(clip_history_id);

        CREATE TABLE validator_runs (
            run_id TEXT PRIMARY KEY,
            video_uri TEXT,
            video_sha256 TEXT,
            payload_json TEXT,
            latency_s REAL,
            validator_version TEXT,
            ran_at REAL
        );
        CREATE UNIQUE INDEX idx_validator_runs_video_version
            ON validator_runs(video_sha256, validator_version);

        CREATE TABLE preference_pairs (
            pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chosen_clip_id TEXT,
            rejected_clip_id TEXT,
            signal_source TEXT,
            signal_strength REAL,
            used_in_training_run_id TEXT,
            created_at REAL
        );

        CREATE TABLE training_runs (
            run_id TEXT PRIMARY KEY,
            base_model TEXT,
            base_model_sha TEXT,
            lora_output_path TEXT,
            lora_registry_id TEXT,
            num_pairs INTEGER,
            val_loss REAL,
            eval_metrics_json TEXT,
            trained_at REAL,
            deployed_at REAL,
            deprecated_at REAL
        );

        CREATE TABLE api_key_metadata (
            api_key_hash TEXT PRIMARY KEY,
            training_opt_in INTEGER NOT NULL DEFAULT 1,
            tier TEXT DEFAULT 'pro',
            notes TEXT,
            created_at REAL,
            updated_at REAL
        );
        """
    )
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


def test_schema_v4_migration_runs_clean(tmp_path: Path) -> None:
    """Fresh DB → migrate → all 7 new columns + 2 new indexes exist."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    gen_cols = _table_columns(store._conn, "generations")
    assert "motion_intent" in gen_cols
    assert "embedding_model_version" in gen_cols

    pp_cols = _table_columns(store._conn, "preference_pairs")
    assert "validator_version" in pp_cols

    pp_indexes = _indexes(store._conn, "preference_pairs")
    assert "idx_pp_validator_version" in pp_indexes
    assert "idx_pp_unique_pair_source" in pp_indexes

    tr_cols = _table_columns(store._conn, "training_runs")
    for col in (
        "training_seed",
        "hyperparams_json",
        "dataset_snapshot_path",
        "code_sha",
        "validator_version_at_train",
    ):
        assert col in tr_cols, f"missing training_runs column: {col}"

    user_version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == CURRENT_SCHEMA_VERSION
    assert user_version >= 4


def test_schema_v4_migration_idempotent(tmp_path: Path) -> None:
    """Running ``_migrate()`` twice on the same DB → no error, schema identical."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    gen_cols_first = _table_columns(store._conn, "generations")
    pp_indexes_first = _indexes(store._conn, "preference_pairs")
    tr_cols_first = _table_columns(store._conn, "training_runs")

    # Re-running _migrate must be a no-op.
    store._migrate()
    store._migrate()

    user_version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == CURRENT_SCHEMA_VERSION
    assert user_version >= 4

    assert _table_columns(store._conn, "generations") == gen_cols_first
    assert _indexes(store._conn, "preference_pairs") == pp_indexes_first
    assert _table_columns(store._conn, "training_runs") == tr_cols_first


def test_schema_v3_to_v4_upgrade_preserves_rows(tmp_path: Path) -> None:
    """Populate v3 DB with sample rows, run ``_migrate()``, verify all rows
    preserved + new columns NULL."""
    db = tmp_path / "history.db"
    _make_v3_db(db)

    # Insert sample rows using the v3 column set.
    conn = sqlite3.connect(str(db))
    conn.execute(
        """INSERT INTO generations (id, api_key_hash, job_type, prompt, status,
            created_at, validator_score, validator_version, parent_clip_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "g_v3_a", _hash_key("k"), "audio-to-video", "a sunset",
            "completed", time.time(), 0.82, "1.17.0-rc5", None,
        ),
    )
    conn.execute(
        """INSERT INTO preference_pairs (chosen_clip_id, rejected_clip_id,
            signal_source, signal_strength, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        ("g_v3_a", "g_v3_b", "user_retake", 0.9, time.time()),
    )
    conn.execute(
        """INSERT INTO training_runs (run_id, base_model, num_pairs, trained_at)
           VALUES (?, ?, ?, ?)""",
        ("tr_legacy", "ltx-2-3-fast", 100, time.time()),
    )
    conn.commit()
    conn.close()

    store = HistoryStore(db_path=db)
    user_version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == CURRENT_SCHEMA_VERSION
    assert user_version >= 4

    # Verify generations row preserved + new cols NULL.
    row = store._conn.execute(
        "SELECT id, prompt, validator_score, motion_intent, "
        "embedding_model_version FROM generations WHERE id = 'g_v3_a'"
    ).fetchone()
    assert row["prompt"] == "a sunset"
    assert row["validator_score"] == pytest.approx(0.82)
    assert row["motion_intent"] is None
    assert row["embedding_model_version"] is None

    # Verify preference_pairs row preserved + new col NULL.
    pp_row = store._conn.execute(
        "SELECT chosen_clip_id, rejected_clip_id, signal_source, "
        "signal_strength, validator_version FROM preference_pairs"
    ).fetchone()
    assert pp_row["chosen_clip_id"] == "g_v3_a"
    assert pp_row["signal_source"] == "user_retake"
    assert pp_row["validator_version"] is None

    # Verify training_runs row preserved + 5 new cols NULL.
    tr_row = store._conn.execute(
        "SELECT run_id, base_model, training_seed, hyperparams_json, "
        "dataset_snapshot_path, code_sha, validator_version_at_train "
        "FROM training_runs WHERE run_id = 'tr_legacy'"
    ).fetchone()
    assert tr_row["base_model"] == "ltx-2-3-fast"
    assert tr_row["training_seed"] is None
    assert tr_row["hyperparams_json"] is None
    assert tr_row["dataset_snapshot_path"] is None
    assert tr_row["code_sha"] is None
    assert tr_row["validator_version_at_train"] is None


def test_motion_intent_column_nullable(tmp_path: Path) -> None:
    """INSERT row without motion_intent succeeds; SELECT returns NULL."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    store._conn.execute(
        """INSERT INTO generations (id, api_key_hash, job_type, prompt,
            status, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("g_no_intent", _hash_key("k"), "text-to-video", "p",
         "completed", time.time()),
    )
    store._conn.commit()
    row = store._conn.execute(
        "SELECT motion_intent, embedding_model_version FROM generations "
        "WHERE id = 'g_no_intent'"
    ).fetchone()
    assert row["motion_intent"] is None
    assert row["embedding_model_version"] is None

    # And explicit population also works.
    store._conn.execute(
        """INSERT INTO generations (id, api_key_hash, job_type, prompt,
            status, created_at, motion_intent, embedding_model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("g_with_intent", _hash_key("k"), "text-to-video", "p",
         "completed", time.time(), "static portrait", "gemma-3-12b-it"),
    )
    store._conn.commit()
    row = store._conn.execute(
        "SELECT motion_intent, embedding_model_version FROM generations "
        "WHERE id = 'g_with_intent'"
    ).fetchone()
    assert row["motion_intent"] == "static portrait"
    assert row["embedding_model_version"] == "gemma-3-12b-it"


def test_preference_pairs_validator_version_index_works(tmp_path: Path) -> None:
    """Query plan for a validator_version-filtered SELECT uses
    ``idx_pp_validator_version``."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    # Index must exist by name.
    pp_indexes = _indexes(store._conn, "preference_pairs")
    assert "idx_pp_validator_version" in pp_indexes

    # And the query planner picks it up. EXPLAIN QUERY PLAN returns rows
    # whose 'detail' column references the index name when used.
    plan_rows = store._conn.execute(
        "EXPLAIN QUERY PLAN SELECT pair_id FROM preference_pairs "
        "WHERE validator_version = ?",
        ("1.17.0-rc5",),
    ).fetchall()
    plan_text = " ".join(str(r["detail"]) for r in plan_rows)
    assert "idx_pp_validator_version" in plan_text, plan_text


def test_preference_pairs_unique_pair_source_constraint(tmp_path: Path) -> None:
    """INSERT same ``(chosen, rejected, source)`` twice fails with
    ``IntegrityError``; INSERT OR IGNORE succeeds (idempotence for Phase C
    pair construction)."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    now = time.time()
    store._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("c_a", "r_b", "user_retake", 0.9, "1.17.0-rc5", now),
    )
    store._conn.commit()

    # Second insert with same (chosen, rejected, source) → IntegrityError.
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """INSERT INTO preference_pairs
               (chosen_clip_id, rejected_clip_id, signal_source,
                signal_strength, validator_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("c_a", "r_b", "user_retake", 0.9, "1.17.0-rc5", now),
        )
        store._conn.commit()
    store._conn.rollback()

    # INSERT OR IGNORE succeeds (no-op on duplicate).
    store._conn.execute(
        """INSERT OR IGNORE INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("c_a", "r_b", "user_retake", 0.9, "1.17.0-rc5", now),
    )
    store._conn.commit()

    # But a different signal_source on the same pair is allowed (multi-source
    # aggregation pattern from Phase C plan).
    store._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("c_a", "r_b", "validator_pass", 0.7, "1.17.0-rc5", now),
    )
    store._conn.commit()

    count = store._conn.execute(
        "SELECT COUNT(*) FROM preference_pairs"
    ).fetchone()[0]
    assert count == 2


def test_training_runs_reproducibility_columns(tmp_path: Path) -> None:
    """INSERT row populating all 5 new columns + SELECT round-trip."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    now = time.time()
    hyperparams = {"lr": 5e-4, "batch_size": 1, "epochs": 3, "rank": 64}
    store._conn.execute(
        """INSERT INTO training_runs
           (run_id, base_model, base_model_sha, lora_output_path,
            num_pairs, val_loss, trained_at, training_seed,
            hyperparams_json, dataset_snapshot_path, code_sha,
            validator_version_at_train)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "sft-quality-v0.0.1", "ltx-2-3-fast", "abc123",
            "/loras/sft-quality-v0.0.1.safetensors",
            850, 0.124, now, 42, json.dumps(hyperparams),
            "/snapshots/2026-04-29.jsonl", "deadbeef",
            "1.17.0-rc5",
        ),
    )
    store._conn.commit()

    row = store._conn.execute(
        """SELECT run_id, base_model, training_seed, hyperparams_json,
                  dataset_snapshot_path, code_sha, validator_version_at_train
           FROM training_runs WHERE run_id = 'sft-quality-v0.0.1'"""
    ).fetchone()
    assert row["run_id"] == "sft-quality-v0.0.1"
    assert row["base_model"] == "ltx-2-3-fast"
    assert row["training_seed"] == 42
    assert json.loads(row["hyperparams_json"]) == hyperparams
    assert row["dataset_snapshot_path"] == "/snapshots/2026-04-29.jsonl"
    assert row["code_sha"] == "deadbeef"
    assert row["validator_version_at_train"] == "1.17.0-rc5"


def test_v3_db_loaded_by_v4_code_works(tmp_path: Path) -> None:
    """Open existing v3 DB; verify migration runs to v4; verify queries
    still work on the upgraded schema."""
    db = tmp_path / "history.db"
    _make_v3_db(db)

    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    conn.close()

    store = HistoryStore(db_path=db)
    upgraded = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert upgraded == CURRENT_SCHEMA_VERSION
    assert upgraded >= 4

    # v3-shape inserts still work (no required new columns).
    store._conn.execute(
        """INSERT INTO generations (id, api_key_hash, job_type, prompt,
            status, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("g_post_upgrade", _hash_key("k"), "text-to-video", "p",
         "completed", time.time()),
    )
    store._conn.commit()

    # And a SELECT touching new columns succeeds.
    row = store._conn.execute(
        "SELECT id, motion_intent FROM generations WHERE id = 'g_post_upgrade'"
    ).fetchone()
    assert row["id"] == "g_post_upgrade"
    assert row["motion_intent"] is None


def test_v4_db_loaded_by_v3_code_silently_ignores_extra_columns(
    tmp_path: Path,
) -> None:
    """v4+ DB opened by code that only knows the v3 column set: SQLite is
    column-tolerant, so v3-shape SELECTs and INSERTs continue to work
    even though extra columns are present (additive safety)."""
    db = tmp_path / "history.db"
    # Migrate to current schema (v4 introduced the additive columns this
    # test exercises; future migrations stay additive over the same set).
    HistoryStore(db_path=db)._conn.close()

    # Now open with a raw connection and run only v3-shape queries.
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == CURRENT_SCHEMA_VERSION
    assert user_version >= 4

    # v3-shape INSERT into generations (no new columns referenced) → succeeds.
    conn.execute(
        """INSERT INTO generations (id, api_key_hash, job_type, prompt,
            status, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("g_v3_shape", _hash_key("k"), "text-to-video", "p",
         "completed", time.time()),
    )
    conn.commit()

    # v3-shape SELECT → succeeds; row exists.
    row = conn.execute(
        "SELECT id, prompt FROM generations WHERE id = 'g_v3_shape'"
    ).fetchone()
    assert row["id"] == "g_v3_shape"
    assert row["prompt"] == "p"

    # v3-shape INSERT into preference_pairs (no validator_version) → also OK.
    conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source,
            signal_strength, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        ("c_x", "r_y", "composition_kept", 0.5, time.time()),
    )
    conn.commit()
    conn.close()


def test_existing_v3_test_suite_still_passes(tmp_path: Path) -> None:
    """Meta-test: the v3-era assertions from ``test_v1_17_schema.py`` still
    hold under schema v4 (no regressions on the v3 surface).

    Mirrors the spirit of the four foundational v3 tests
    (migration_runs_clean, idempotent, existing_v2_db_upgrades,
    validator_runs_unique_index). If the v4 migration touched any v3
    structure, this re-check would catch it."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    # All v3 expected-new columns must still exist on a fresh v4 DB.
    cols = _table_columns(store._conn, "generations")
    expected_v3_cols = {
        "validator_score", "validator_payload_json", "validator_version",
        "validator_artifact_uri", "parent_clip_id", "shot_uuid",
        "shot_config_key", "composition_id", "lora_applied_id",
        "lora_applied_strength", "prompt_embedding",
    }
    assert expected_v3_cols.issubset(cols)

    # All v3 tables must still exist.
    tables = {
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for required in (
        "composition_clips", "validator_runs", "preference_pairs",
        "training_runs", "api_key_metadata",
    ):
        assert required in tables, f"v4 migration broke v3 table: {required}"

    # validator_runs UNIQUE index from v3 still enforces.
    now = time.time()
    store._conn.execute(
        """INSERT INTO validator_runs
           (run_id, video_uri, video_sha256, payload_json, latency_s,
            validator_version, ran_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("run_a", "storage://x", "sha_xyz", "{}", 0.5, "1.17.0", now),
    )
    store._conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """INSERT INTO validator_runs
               (run_id, video_uri, video_sha256, payload_json, latency_s,
                validator_version, ran_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("run_b", "storage://x", "sha_xyz", "{}", 0.6, "1.17.0", now),
        )
        store._conn.commit()
    store._conn.rollback()


def test_v5_clip_embeddings_dim(tmp_path):
    """v5 (rc4) rebuilt ``clip_embeddings`` at FLOAT[4096] for qwen3-embed-8b.

    rc1+rc2+rc3 created the virtual table at FLOAT[3584] based on a wrong
    assumption that Gemma 3 12B was an embedding model with that hidden dim.
    The v5 migration drops + recreates the table; safe because no backfill
    had run before rc4.
    """
    import history_store as hs_mod

    if not hs_mod.SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec extension not available; v5 rebuild requires it")

    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    row = store._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='clip_embeddings'"
    ).fetchone()
    assert row is not None, "clip_embeddings virtual table missing"
    sql = row[0]
    assert "FLOAT[4096]" in sql, f"expected FLOAT[4096] in clip_embeddings sql, got: {sql!r}"
    assert "FLOAT[3584]" not in sql, f"v5 migration did not drop the old 3584 dim: {sql!r}"

    user_version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == CURRENT_SCHEMA_VERSION
    assert user_version >= 5
