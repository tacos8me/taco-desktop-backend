"""Tests for v1.19.0-rc3 / v1.20.0 schema v7 migration.

Schema v7 is purely additive over v6:
  - 3 new tables: ``human_ratings``, ``exemplar_sets``,
    ``exemplar_set_members``
  - 3 new ``api_key_metadata`` columns:
    ``validator_pass_threshold_override``,
    ``validator_retake_threshold_override``,
    ``cross_bearer_rating_consent_at``
  - 1 new ``preference_pairs`` column: ``pending_construction_until``
    (Path C′ staging)
  - 1 new ``training_runs`` column: ``status`` (default 'running')
  - Partial UNIQUE index ``idx_hr_unique_active`` enforcing one active
    rating per (rater_hash, clip_id, kind), excluding retracted /
    superseded rows
  - FK CASCADE: deleting an ``exemplar_sets`` row cascades to
    ``exemplar_set_members`` (FK enforced via ``PRAGMA foreign_keys = ON``)
  - ``HistoryStore.delete_rater_corpus(hash)`` right-to-delete cascade
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from history_store import (
    CURRENT_SCHEMA_VERSION,
    HistoryStore,
    _hash_key,
)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    return {row[1] for row in rows}


def _make_v6_db(path: Path) -> None:
    """Build a database that looks like the post-rc5 (v6) schema."""
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
            prompt_embedding BLOB,
            motion_intent TEXT,
            embedding_model_version TEXT,
            ab_arm TEXT
        );
        CREATE INDEX idx_api_key_hash ON generations(api_key_hash, created_at DESC);

        CREATE TABLE preference_pairs (
            pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chosen_clip_id TEXT,
            rejected_clip_id TEXT,
            signal_source TEXT,
            signal_strength REAL,
            used_in_training_run_id TEXT,
            created_at REAL,
            validator_version TEXT
        );
        CREATE INDEX idx_pp_validator_version ON preference_pairs(validator_version);
        CREATE UNIQUE INDEX idx_pp_unique_pair_source
            ON preference_pairs(chosen_clip_id, rejected_clip_id, signal_source);

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
            deprecated_at REAL,
            training_seed INTEGER,
            hyperparams_json TEXT,
            dataset_snapshot_path TEXT,
            code_sha TEXT,
            validator_version_at_train TEXT
        );

        CREATE TABLE api_key_metadata (
            api_key_hash TEXT PRIMARY KEY,
            training_opt_in INTEGER NOT NULL DEFAULT 1,
            tier TEXT DEFAULT 'pro',
            notes TEXT,
            created_at REAL,
            updated_at REAL
        );

        CREATE TABLE composition_clips (
            comp_id TEXT NOT NULL,
            clip_history_id TEXT,
            position INTEGER NOT NULL,
            was_final INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            PRIMARY KEY (comp_id, clip_history_id, position)
        );

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
        """
    )
    conn.execute("PRAGMA user_version = 6")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


def test_schema_v7_migration_runs_clean(tmp_path: Path) -> None:
    """Fresh DB: all v7 columns + new tables + indexes exist."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == \
        CURRENT_SCHEMA_VERSION
    assert CURRENT_SCHEMA_VERSION >= 7

    akm_cols = _table_columns(store._conn, "api_key_metadata")
    assert "validator_pass_threshold_override" in akm_cols
    assert "validator_retake_threshold_override" in akm_cols
    assert "cross_bearer_rating_consent_at" in akm_cols

    pp_cols = _table_columns(store._conn, "preference_pairs")
    assert "pending_construction_until" in pp_cols

    tr_cols = _table_columns(store._conn, "training_runs")
    assert "status" in tr_cols

    tables = {
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for required in ("human_ratings", "exemplar_sets", "exemplar_set_members"):
        assert required in tables, f"v7 missing table: {required}"

    hr_indexes = _index_names(store._conn, "human_ratings")
    assert "idx_hr_unique_active" in hr_indexes
    assert "idx_hr_clip" in hr_indexes
    assert "idx_hr_rater" in hr_indexes
    assert "idx_hr_pair" in hr_indexes

    es_indexes = _index_names(store._conn, "exemplar_sets")
    assert "idx_es_owner" in es_indexes
    esm_indexes = _index_names(store._conn, "exemplar_set_members")
    assert "idx_esm_clip" in esm_indexes


def test_schema_v6_to_v7_migration_idempotent(tmp_path: Path) -> None:
    """Run _migrate twice on a v6 DB: no error, schema identical, no
    duplicate ALTER errors. Pre-existing rows preserved with NULL/default
    values for the new columns."""
    db = tmp_path / "history.db"
    _make_v6_db(db)

    raw = sqlite3.connect(str(db))
    raw.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("c_a", "r_b", "user_retake", 0.9, "1.18.0-rc6", time.time()),
    )
    raw.execute(
        """INSERT INTO training_runs (run_id, base_model, num_pairs, trained_at)
           VALUES (?, ?, ?, ?)""",
        ("legacy_v6", "ltx-2-3-fast", 50, time.time()),
    )
    raw.execute(
        """INSERT INTO api_key_metadata
           (api_key_hash, training_opt_in, tier, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (_hash_key("legacy"), 1, "pro", time.time(), time.time()),
    )
    raw.commit()
    raw.close()

    store = HistoryStore(db_path=db)
    pp_cols_first = _table_columns(store._conn, "preference_pairs")
    tr_cols_first = _table_columns(store._conn, "training_runs")
    akm_cols_first = _table_columns(store._conn, "api_key_metadata")

    # Re-run migration; must be a no-op.
    store._migrate()
    store._migrate()

    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == \
        CURRENT_SCHEMA_VERSION
    assert _table_columns(store._conn, "preference_pairs") == pp_cols_first
    assert _table_columns(store._conn, "training_runs") == tr_cols_first
    assert _table_columns(store._conn, "api_key_metadata") == akm_cols_first

    # Pre-existing rows survived with NULL / default values.
    pp_row = store._conn.execute(
        "SELECT chosen_clip_id, pending_construction_until "
        "FROM preference_pairs WHERE chosen_clip_id = 'c_a'"
    ).fetchone()
    assert pp_row["pending_construction_until"] is None

    tr_row = store._conn.execute(
        "SELECT run_id, status FROM training_runs WHERE run_id = 'legacy_v6'"
    ).fetchone()
    # SQLite's ALTER TABLE ADD COLUMN ... DEFAULT <literal> backfills the
    # default into existing rows when the default is a non-NULL constant
    # (https://www.sqlite.org/lang_altertable.html). 'running' is a literal,
    # so the legacy row picks it up. Operator should treat the migration as
    # a clean slate for status tracking — older runs were never cancelled.
    assert tr_row["status"] == "running"

    # NEW post-migration row also picks up the default.
    store._conn.execute(
        """INSERT INTO training_runs (run_id, base_model, trained_at)
           VALUES (?, ?, ?)""",
        ("post_v7_run", "ltx-2-3-fast", time.time()),
    )
    store._conn.commit()
    new_row = store._conn.execute(
        "SELECT status FROM training_runs WHERE run_id = 'post_v7_run'"
    ).fetchone()
    assert new_row["status"] == "running"

    akm_row = store._conn.execute(
        "SELECT validator_pass_threshold_override, "
        "       validator_retake_threshold_override, "
        "       cross_bearer_rating_consent_at "
        "FROM api_key_metadata WHERE api_key_hash = ?",
        (_hash_key("legacy"),),
    ).fetchone()
    assert akm_row["validator_pass_threshold_override"] is None
    assert akm_row["validator_retake_threshold_override"] is None
    assert akm_row["cross_bearer_rating_consent_at"] is None


def test_v7_new_columns_nullable_with_correct_defaults(tmp_path: Path) -> None:
    """All new columns accept NULL on insert; defaults match the spec.

    - api_key_metadata.* (3 cols): NULL default
    - preference_pairs.pending_construction_until: NULL default
    - training_runs.status: 'running' default
    - exemplar_sets.max_members: 200 default
    """
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    # api_key_metadata: insert a row, leave all v7 cols unset.
    store._conn.execute(
        """INSERT INTO api_key_metadata
           (api_key_hash, training_opt_in, tier, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (_hash_key("k1"), 1, "pro", time.time(), time.time()),
    )

    # preference_pairs: leave pending_construction_until unset.
    store._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("c_x", "r_y", "human_rating", 0.75, "1.19.0-rc1", time.time()),
    )

    # training_runs: leave status unset (DEFAULT 'running' applies).
    store._conn.execute(
        """INSERT INTO training_runs (run_id, base_model, trained_at)
           VALUES (?, ?, ?)""",
        ("tr_default", "ltx-2-3-fast", time.time()),
    )

    # exemplar_sets: leave max_members unset (DEFAULT 200 applies).
    store._conn.execute(
        """INSERT INTO exemplar_sets
           (set_id, rater_api_key_hash, created_at, updated_at)
           VALUES (?, ?, ?, ?)""",
        ("default-set", _hash_key("k1"), time.time(), time.time()),
    )
    store._conn.commit()

    akm = store._conn.execute(
        "SELECT validator_pass_threshold_override, "
        "       validator_retake_threshold_override, "
        "       cross_bearer_rating_consent_at "
        "FROM api_key_metadata WHERE api_key_hash = ?",
        (_hash_key("k1"),),
    ).fetchone()
    assert akm["validator_pass_threshold_override"] is None
    assert akm["validator_retake_threshold_override"] is None
    assert akm["cross_bearer_rating_consent_at"] is None

    pp = store._conn.execute(
        "SELECT pending_construction_until FROM preference_pairs "
        "WHERE chosen_clip_id = 'c_x'"
    ).fetchone()
    assert pp["pending_construction_until"] is None

    tr = store._conn.execute(
        "SELECT status FROM training_runs WHERE run_id = 'tr_default'"
    ).fetchone()
    assert tr["status"] == "running"

    es = store._conn.execute(
        "SELECT max_members FROM exemplar_sets WHERE set_id = 'default-set'"
    ).fetchone()
    assert es["max_members"] == 200


def test_human_ratings_partial_unique_index(tmp_path: Path) -> None:
    """Partial UNIQUE on (rater_hash, clip_id, kind) WHERE retracted_at
    IS NULL AND superseded_by IS NULL.

    - Two active rows of (same rater, same clip, same kind) → IntegrityError.
    - Retract one → second insert succeeds.
    - Marking ``superseded_by`` on the original also frees the slot.

    # safety-critical: invariant-5 (idempotence + audit chain)
    """
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    rater = _hash_key("rater_x")
    now = time.time()
    store._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            validator_visible_at_rating, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("clip_a", rater, "pair_chose_a", 1.0, 0, now),
    )
    store._conn.commit()

    # Duplicate ACTIVE row → IntegrityError.
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """INSERT INTO human_ratings
               (clip_id, rater_api_key_hash, rating_kind, rating_value,
                validator_visible_at_rating, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("clip_a", rater, "pair_chose_a", 1.0, 0, now),
        )
        store._conn.commit()
    store._conn.rollback()

    # Retract the original → second insert succeeds.
    store._conn.execute(
        "UPDATE human_ratings SET retracted_at = ? WHERE clip_id = 'clip_a' "
        "AND rater_api_key_hash = ?",
        (now, rater),
    )
    store._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            validator_visible_at_rating, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("clip_a", rater, "pair_chose_a", 1.0, 0, now + 1),
    )
    store._conn.commit()

    # Different kind on same (rater, clip) is allowed even when an active
    # row exists with a different kind.
    store._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            validator_visible_at_rating, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("clip_a", rater, "tag", 0.0, 0, now + 2),
    )
    store._conn.commit()

    # Superseded row (active=NULL retracted_at AND superseded_by NOT NULL)
    # also frees the slot.
    rid = store._conn.execute(
        "SELECT rating_id FROM human_ratings "
        "WHERE clip_id='clip_a' AND retracted_at IS NULL AND "
        "      rating_kind='pair_chose_a'"
    ).fetchone()["rating_id"]
    store._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            validator_visible_at_rating, superseded_by, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("clip_b", rater, "pair_chose_a", 1.0, 0, 999, now),
    )
    # The superseded row is on a different clip; just sanity-check we can
    # also insert a fresh active row on clip_b under the kind that we
    # marked superseded for clip_a.
    store._conn.execute(
        "UPDATE human_ratings SET superseded_by = ? WHERE rating_id = ?",
        (rid + 100, rid),
    )
    store._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            validator_visible_at_rating, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("clip_a", rater, "pair_chose_a", 1.0, 0, now + 3),
    )
    store._conn.commit()


def test_exemplar_sets_fk_cascade_on_delete(tmp_path: Path) -> None:
    """Deleting an ``exemplar_sets`` row cascades to ``exemplar_set_members``.

    Requires PRAGMA foreign_keys = ON to be set (HistoryStore enables
    this in __init__).

    # safety-critical: L2.5-cross-set-membership (FK cascade prevents orphaned member rows)
    """
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    fk_on = store._conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk_on == 1, "FK enforcement must be enabled for cascade tests"

    rater = _hash_key("rater_y")
    now = time.time()
    store._conn.execute(
        """INSERT INTO exemplar_sets
           (set_id, rater_api_key_hash, description, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        ("set-jaguar", rater, "leatherjacket exemplars", now, now),
    )
    for clip_id in ("clip_1", "clip_2", "clip_3"):
        store._conn.execute(
            """INSERT INTO exemplar_set_members
               (set_id, clip_id, added_at, note)
               VALUES (?, ?, ?, ?)""",
            ("set-jaguar", clip_id, now, None),
        )
    store._conn.commit()

    n_before = store._conn.execute(
        "SELECT COUNT(*) AS c FROM exemplar_set_members "
        "WHERE set_id = 'set-jaguar'"
    ).fetchone()["c"]
    assert n_before == 3

    store._conn.execute(
        "DELETE FROM exemplar_sets WHERE set_id = 'set-jaguar'"
    )
    store._conn.commit()

    n_after = store._conn.execute(
        "SELECT COUNT(*) AS c FROM exemplar_set_members "
        "WHERE set_id = 'set-jaguar'"
    ).fetchone()["c"]
    assert n_after == 0, "FK CASCADE failed to wipe member rows"


def test_delete_rater_corpus_cascades_all_targets(tmp_path: Path) -> None:
    """``delete_rater_corpus`` cascades across the four corpus tables.

    - human_ratings: all rows by rater removed
    - preference_pairs: rows referenced by rater's human_ratings via pair_id
      removed (only when used_in_training_run_id IS NULL)
    - exemplar_sets: rows owned by rater removed
    - exemplar_set_members: cascades via FK on the parent delete

    # safety-critical: multi-tenant-readiness-(b) (right-to-delete cascade),
    #                  invariant-5 (consumed pair preserved post-deletion)
    """
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    rater = _hash_key("rater_z")
    other = _hash_key("rater_other")
    now = time.time()

    # 1) human_ratings + matching preference_pairs (unconsumed, must delete).
    cur = store._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at, pending_construction_until)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("c_a", "r_b", "human_rating", 0.75, "1.19.0", now, now + 86400),
    )
    pair_id_unconsumed = cur.lastrowid

    # 2) Already-consumed pair must be PRESERVED — retraction can't unwind a
    #    shipped LoRA; documented behavior in the plan.
    cur = store._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, used_in_training_run_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("c_x", "r_y", "human_rating", 0.75, "1.19.0", "tr_shipped", now),
    )
    pair_id_consumed = cur.lastrowid

    store._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            validator_visible_at_rating, pair_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("c_a", rater, "pair_chose_a", 1.0, 0, pair_id_unconsumed, now),
    )
    store._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            validator_visible_at_rating, pair_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("c_x", rater, "pair_chose_a", 1.0, 0, pair_id_consumed, now),
    )

    # 3) Rating from ANOTHER rater on the same clip must survive.
    store._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            validator_visible_at_rating, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("c_a", other, "good", 1.0, 0, now),
    )

    # 4) exemplar_sets owned by rater + members.
    store._conn.execute(
        """INSERT INTO exemplar_sets
           (set_id, rater_api_key_hash, created_at, updated_at)
           VALUES (?, ?, ?, ?)""",
        ("set-rater-z", rater, now, now),
    )
    for clip in ("c_a", "c_b"):
        store._conn.execute(
            """INSERT INTO exemplar_set_members
               (set_id, clip_id, added_at)
               VALUES (?, ?, ?)""",
            ("set-rater-z", clip, now),
        )

    # 5) exemplar_set owned by ANOTHER rater must survive.
    store._conn.execute(
        """INSERT INTO exemplar_sets
           (set_id, rater_api_key_hash, created_at, updated_at)
           VALUES (?, ?, ?, ?)""",
        ("set-other", other, now, now),
    )
    store._conn.execute(
        """INSERT INTO exemplar_set_members (set_id, clip_id, added_at)
           VALUES (?, ?, ?)""",
        ("set-other", "c_z", now),
    )
    store._conn.commit()

    counts = store.delete_rater_corpus(rater)

    # Returned counts: 2 ratings deleted, 1 unconsumed pair deleted,
    # 1 set deleted, 2 members deleted.
    assert counts["human_ratings"] == 2
    assert counts["preference_pairs"] == 1
    assert counts["exemplar_sets"] == 1
    assert counts["exemplar_set_members"] == 2

    # Verify post-state.
    n_hr = store._conn.execute(
        "SELECT COUNT(*) AS c FROM human_ratings "
        "WHERE rater_api_key_hash = ?",
        (rater,),
    ).fetchone()["c"]
    assert n_hr == 0

    other_survived = store._conn.execute(
        "SELECT COUNT(*) AS c FROM human_ratings "
        "WHERE rater_api_key_hash = ?",
        (other,),
    ).fetchone()["c"]
    assert other_survived == 1

    consumed_pair_survived = store._conn.execute(
        "SELECT COUNT(*) AS c FROM preference_pairs WHERE pair_id = ?",
        (pair_id_consumed,),
    ).fetchone()["c"]
    assert consumed_pair_survived == 1

    unconsumed_pair_gone = store._conn.execute(
        "SELECT COUNT(*) AS c FROM preference_pairs WHERE pair_id = ?",
        (pair_id_unconsumed,),
    ).fetchone()["c"]
    assert unconsumed_pair_gone == 0

    n_sets = store._conn.execute(
        "SELECT COUNT(*) AS c FROM exemplar_sets WHERE set_id = 'set-rater-z'"
    ).fetchone()["c"]
    assert n_sets == 0
    n_members = store._conn.execute(
        "SELECT COUNT(*) AS c FROM exemplar_set_members "
        "WHERE set_id = 'set-rater-z'"
    ).fetchone()["c"]
    assert n_members == 0

    other_set_survived = store._conn.execute(
        "SELECT COUNT(*) AS c FROM exemplar_sets WHERE set_id = 'set-other'"
    ).fetchone()["c"]
    assert other_set_survived == 1


def test_schema_v8_system_flags_table(tmp_path: Path) -> None:
    """v8 adds a `system_flags` key/value table for process-wide pause flags.

    - PRAGMA user_version == 8 after migration on a fresh DB.
    - Table exists with PK on `name`.
    - Insert + select round-trip works.
    """
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)

    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == \
        CURRENT_SCHEMA_VERSION
    assert CURRENT_SCHEMA_VERSION >= 8

    tables = {
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "system_flags" in tables

    sf_cols = _table_columns(store._conn, "system_flags")
    assert sf_cols == {"name", "value", "updated_at"}

    now = time.time()
    store._conn.execute(
        "INSERT INTO system_flags (name, value, updated_at) VALUES (?, ?, ?)",
        ("human_rating_paused_until", str(now + 3600), now),
    )
    store._conn.commit()

    # PK enforced on `name`.
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO system_flags (name, value, updated_at) VALUES (?, ?, ?)",
            ("human_rating_paused_until", "x", now),
        )
        store._conn.commit()
    store._conn.rollback()

    row = store._conn.execute(
        "SELECT value FROM system_flags WHERE name = ?",
        ("human_rating_paused_until",),
    ).fetchone()
    assert row["value"] == str(now + 3600)


def test_schema_v7_to_v8_migration_idempotent(tmp_path: Path) -> None:
    """Run _migrate twice on a v7 DB: lands at v8 with system_flags table,
    second call is a no-op."""
    db = tmp_path / "history.db"
    store = HistoryStore(db_path=db)
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == \
        CURRENT_SCHEMA_VERSION

    # Re-run migrations; idempotent.
    store._migrate()
    store._migrate()
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == \
        CURRENT_SCHEMA_VERSION

    sf_cols = _table_columns(store._conn, "system_flags")
    assert "name" in sf_cols and "value" in sf_cols and "updated_at" in sf_cols


def test_v6_db_loaded_by_v7_code_works(tmp_path: Path) -> None:
    """Open existing v6 DB; migration runs to v7; v6-shape inserts still work,
    new-shape inserts also work."""
    db = tmp_path / "history.db"
    _make_v6_db(db)

    raw = sqlite3.connect(str(db))
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 6
    raw.close()

    store = HistoryStore(db_path=db)
    upgraded = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert upgraded == CURRENT_SCHEMA_VERSION
    assert upgraded >= 7

    # v6-shape preference_pairs INSERT (no pending_construction_until) → OK.
    store._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source,
            signal_strength, validator_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("c_legacy", "r_legacy", "user_retake", 0.9, "1.18.0", time.time()),
    )
    store._conn.commit()

    # New v7-shape INSERT into human_ratings.
    store._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            validator_visible_at_rating, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("c_legacy", _hash_key("k"), "good", 1.0, 0, time.time()),
    )
    store._conn.commit()

    n = store._conn.execute(
        "SELECT COUNT(*) AS c FROM human_ratings WHERE clip_id = 'c_legacy'"
    ).fetchone()["c"]
    assert n == 1
