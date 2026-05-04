"""Phase 3 prereq — Path C′ ETL + train_dpo_sft staged-pair filter.

Covers:
  - construct_human_rating_pairs runs cleanly on an empty table.
  - Valid pair → pending_construction_until cleared; train_dpo_sft picks
    it up.
  - Bearer-opt-out / validator-NULL / cross-bearer-consent-revoked /
    validator_visible → row DELETED.
  - system_flags.human_rating_paused_until set in the future → ETL
    skips entirely.
  - train_dpo_sft.select_chosen_ids skips rows with
    pending_construction_until set.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import config

config.GPU_DEVICES = []
config.API_KEYS = set()

from history_store import HistoryStore, _hash_key  # noqa: E402
from scripts import construct_human_rating_pairs as chp  # noqa: E402
from scripts import train_dpo_sft as tds  # noqa: E402


@pytest.fixture()
def fresh_db(tmp_path: Path) -> HistoryStore:
    db = tmp_path / "history.db"
    return HistoryStore(db_path=db)


def _seed_opt_in(
    history: HistoryStore,
    key: str,
    *,
    opt_in: int = 1,
    consent: bool = False,
) -> str:
    h = _hash_key(key)
    now = time.time()
    history._conn.execute(
        """INSERT OR REPLACE INTO api_key_metadata
           (api_key_hash, training_opt_in, tier, notes, created_at,
            updated_at, cross_bearer_rating_consent_at)
           VALUES (?, ?, 'pro', NULL, ?, ?, ?)""",
        (h, opt_in, now, now, now if consent else None),
    )
    history._conn.commit()
    return h


def _insert_clip(
    history: HistoryStore,
    *,
    clip_id: str,
    bearer_hash: str,
    validator_score: float | None = 0.85,
    validator_version: str = "1.19.0",
) -> None:
    now = time.time()
    history._conn.execute(
        """INSERT INTO generations
           (id, api_key_hash, job_type, prompt, model, width, height,
            turbo, status, result_uri, created_at, completed_at,
            validator_score, validator_version)
           VALUES (?, ?, 'text-to-video', 'p', 'ltx-2-3-fast',
                   1024, 576, 0, 'completed', 'storage://x.mp4', ?, ?, ?, ?)""",
        (clip_id, bearer_hash, now, now, validator_score, validator_version),
    )
    history._conn.commit()


def _stage_pair(
    history: HistoryStore,
    *,
    chosen: str,
    rejected: str,
    rater_hash: str,
    validator_visible: bool = False,
    pending_until: float | None = None,
    payload: dict | None = None,
) -> int:
    """Stage a (preference_pair, human_rating) tuple.

    Returns the pair_id.
    """
    now = time.time()
    if pending_until is None:
        pending_until = now - 60  # already eligible
    cur = history._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source,
            signal_strength, validator_version, created_at,
            pending_construction_until)
           VALUES (?, ?, 'human_rating', 0.75, '1.19.0', ?, ?)""",
        (chosen, rejected, now, pending_until),
    )
    pair_id = int(cur.lastrowid)

    payload_json = json.dumps(payload) if payload else None
    history._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            rating_payload_json, validator_visible_at_rating, pair_id,
            pair_partner_clip_id, created_at)
           VALUES (?, ?, 'pair_chose_a', 1.0, ?, ?, ?, ?, ?)""",
        (
            chosen,
            rater_hash,
            payload_json,
            1 if validator_visible else 0,
            pair_id,
            rejected,
            now,
        ),
    )
    history._conn.commit()
    return pair_id


def test_etl_empty_table_is_noop(fresh_db: HistoryStore) -> None:
    result = chp.construct(fresh_db, dry_run=False)
    assert result == {"paused": 0, "cleared": 0, "deleted": 0, "scanned": 0}


def test_etl_valid_pair_clears_and_train_picks_up(fresh_db: HistoryStore) -> None:
    bearer = _seed_opt_in(fresh_db, "bearer-1")
    rater = bearer  # rater is also the bearer → no cross-bearer consent needed
    _insert_clip(fresh_db, clip_id="c_chosen", bearer_hash=bearer)
    _insert_clip(fresh_db, clip_id="c_rejected", bearer_hash=bearer)

    pair_id = _stage_pair(
        fresh_db, chosen="c_chosen", rejected="c_rejected", rater_hash=rater
    )

    result = chp.construct(fresh_db, dry_run=False)
    assert result["cleared"] == 1
    assert result["deleted"] == 0

    row = fresh_db._conn.execute(
        "SELECT pending_construction_until FROM preference_pairs WHERE pair_id = ?",
        (pair_id,),
    ).fetchone()
    assert row["pending_construction_until"] is None

    chosen_ids = tds.select_chosen_ids(
        fresh_db, validator_version="1.19.0", min_signal_strength=0.5
    )
    assert "c_chosen" in chosen_ids


def test_etl_bearer_opt_out_deletes_pair(fresh_db: HistoryStore) -> None:
    bearer = _seed_opt_in(fresh_db, "bearer-2")
    _insert_clip(fresh_db, clip_id="c_chosen", bearer_hash=bearer)
    _insert_clip(fresh_db, clip_id="c_rejected", bearer_hash=bearer)
    pair_id = _stage_pair(
        fresh_db, chosen="c_chosen", rejected="c_rejected", rater_hash=bearer
    )

    # Flip bearer opt-out AFTER staging.
    fresh_db._conn.execute(
        "UPDATE api_key_metadata SET training_opt_in = 0 WHERE api_key_hash = ?",
        (bearer,),
    )
    fresh_db._conn.commit()

    result = chp.construct(fresh_db, dry_run=False)
    assert result["deleted"] == 1
    assert result["cleared"] == 0

    n = fresh_db._conn.execute(
        "SELECT COUNT(*) AS c FROM preference_pairs WHERE pair_id = ?",
        (pair_id,),
    ).fetchone()["c"]
    assert n == 0

    chosen_ids = tds.select_chosen_ids(
        fresh_db, validator_version="1.19.0", min_signal_strength=0.5
    )
    assert "c_chosen" not in chosen_ids


def test_etl_validator_visible_at_rating_drops_pair(fresh_db: HistoryStore) -> None:
    bearer = _seed_opt_in(fresh_db, "bearer-3")
    _insert_clip(fresh_db, clip_id="c_chosen", bearer_hash=bearer)
    _insert_clip(fresh_db, clip_id="c_rejected", bearer_hash=bearer)
    _stage_pair(
        fresh_db,
        chosen="c_chosen",
        rejected="c_rejected",
        rater_hash=bearer,
        validator_visible=True,
    )

    result = chp.construct(fresh_db, dry_run=False)
    assert result["deleted"] == 1


def test_etl_cross_bearer_consent_revoked_drops(fresh_db: HistoryStore) -> None:
    bearer = _seed_opt_in(fresh_db, "bearer-4")
    rater = _seed_opt_in(fresh_db, "rater-x", consent=False)
    _insert_clip(fresh_db, clip_id="c_chosen", bearer_hash=bearer)
    _insert_clip(fresh_db, clip_id="c_rejected", bearer_hash=bearer)
    _stage_pair(
        fresh_db, chosen="c_chosen", rejected="c_rejected", rater_hash=rater
    )

    result = chp.construct(fresh_db, dry_run=False)
    assert result["deleted"] == 1


def test_etl_cross_bearer_consent_present_clears(fresh_db: HistoryStore) -> None:
    bearer = _seed_opt_in(fresh_db, "bearer-5")
    rater = _seed_opt_in(fresh_db, "rater-y", consent=True)
    _insert_clip(fresh_db, clip_id="c_chosen", bearer_hash=bearer)
    _insert_clip(fresh_db, clip_id="c_rejected", bearer_hash=bearer)
    _stage_pair(
        fresh_db, chosen="c_chosen", rejected="c_rejected", rater_hash=rater
    )

    result = chp.construct(fresh_db, dry_run=False)
    assert result["cleared"] == 1
    assert result["deleted"] == 0


def test_etl_pause_flag_skips_run(fresh_db: HistoryStore) -> None:
    bearer = _seed_opt_in(fresh_db, "bearer-6")
    _insert_clip(fresh_db, clip_id="c_chosen", bearer_hash=bearer)
    _insert_clip(fresh_db, clip_id="c_rejected", bearer_hash=bearer)
    pair_id = _stage_pair(
        fresh_db, chosen="c_chosen", rejected="c_rejected", rater_hash=bearer
    )

    future = time.time() + 3600
    fresh_db._conn.execute(
        "INSERT INTO system_flags (name, value, updated_at) VALUES (?, ?, ?)",
        ("human_rating_paused_until", str(future), time.time()),
    )
    fresh_db._conn.commit()

    result = chp.construct(fresh_db, dry_run=False)
    assert result["paused"] == 1
    assert result["scanned"] == 0

    # The pair stays staged.
    row = fresh_db._conn.execute(
        "SELECT pending_construction_until FROM preference_pairs WHERE pair_id = ?",
        (pair_id,),
    ).fetchone()
    assert row["pending_construction_until"] is not None


def test_etl_dry_run_does_not_mutate(fresh_db: HistoryStore) -> None:
    bearer = _seed_opt_in(fresh_db, "bearer-7")
    _insert_clip(fresh_db, clip_id="c_chosen", bearer_hash=bearer)
    _insert_clip(fresh_db, clip_id="c_rejected", bearer_hash=bearer)
    pair_id = _stage_pair(
        fresh_db, chosen="c_chosen", rejected="c_rejected", rater_hash=bearer
    )

    result = chp.construct(fresh_db, dry_run=True)
    assert result["cleared"] == 1

    row = fresh_db._conn.execute(
        "SELECT pending_construction_until FROM preference_pairs WHERE pair_id = ?",
        (pair_id,),
    ).fetchone()
    assert row["pending_construction_until"] is not None


def test_train_dpo_sft_skips_staged_rows(fresh_db: HistoryStore) -> None:
    """A pair with pending_construction_until set must NOT enter training."""
    now = time.time()
    fresh_db._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source,
            signal_strength, validator_version, created_at,
            pending_construction_until)
           VALUES ('c_staged', 'r_staged', 'human_rating', 0.9,
                   '1.19.0', ?, ?)""",
        (now, now + 86400),
    )
    fresh_db._conn.execute(
        """INSERT INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source,
            signal_strength, validator_version, created_at,
            pending_construction_until)
           VALUES ('c_ready', 'r_ready', 'human_rating', 0.9,
                   '1.19.0', ?, NULL)""",
        (now,),
    )
    fresh_db._conn.commit()

    chosen_ids = tds.select_chosen_ids(
        fresh_db, validator_version="1.19.0", min_signal_strength=0.5
    )
    assert "c_ready" in chosen_ids
    assert "c_staged" not in chosen_ids


def test_etl_delete_rater_corpus_safe_with_system_flags(
    fresh_db: HistoryStore,
) -> None:
    """delete_rater_corpus continues to work after the v8 schema bump.

    No rater-scoped flag keys are landed today, but the cascade helper
    must still operate safely against the new system_flags table.
    """
    rater = _seed_opt_in(fresh_db, "rater-d")
    fresh_db._conn.execute(
        "INSERT INTO system_flags (name, value, updated_at) VALUES (?, ?, ?)",
        ("human_rating_paused_until", "0", time.time()),
    )
    fresh_db._conn.commit()
    counts = fresh_db.delete_rater_corpus(rater)
    assert counts["human_ratings"] == 0
    n = fresh_db._conn.execute(
        "SELECT COUNT(*) FROM system_flags"
    ).fetchone()[0]
    assert n == 1
