#!/usr/bin/env python3
"""Phase 3 prereq — human_rating preference_pairs construction (Path C′).

Hourly cron-suitable. Walks ``preference_pairs`` rows that were staged
by the rating endpoint with ``signal_source='human_rating'`` AND
``pending_construction_until`` set in the past, re-validates the 5
stop-the-line invariants, and either clears the column (the row becomes
training-eligible) or DELETEs the row (an invariant has flipped after
the rating was captured).

Defaults to ``--execute`` (this is a maintenance ETL, not a training
run); ``--dry-run`` flag is provided for safety and CI.

Invariants re-checked per pair:

  (a) chosen_clip's bearer still has ``training_opt_in=1``
  (b) rejected_clip's bearer still has ``training_opt_in=1`` (when
      rejected_clip_id is non-NULL)
  (c) ``validator_score`` still NOT NULL on both clips
  (d) cross-bearer rating consent still in place when the rater is
      neither bearer (rater hash is looked up via the linking
      ``human_ratings.pair_id``)
  (e) the linking ``human_ratings.rating_payload_json`` reports
      ``validator_visible_at_rating`` falsy (visible = down-weight or
      drop; spec: drop)

Process-wide pause flag: ``system_flags.human_rating_paused_until``. If
set and ``> now``, the script logs a single line and exits 0 without
touching any rows.

Usage::

    uv run python scripts/construct_human_rating_pairs.py
    uv run python scripts/construct_human_rating_pairs.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from history_store import HistoryStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("construct_human_rating_pairs")


def _is_paused(history: HistoryStore, now: float) -> float | None:
    """Return the pause expiry epoch when paused, else ``None``."""
    row = history._conn.execute(
        "SELECT value FROM system_flags WHERE name = ?",
        ("human_rating_paused_until",),
    ).fetchone()
    if row is None or row["value"] is None:
        return None
    try:
        until = float(row["value"])
    except (TypeError, ValueError):
        return None
    if until > now:
        return until
    return None


def _bearer_opted_in(history: HistoryStore, clip_id: str) -> bool:
    """True iff the bearer of ``clip_id`` still has training_opt_in=1.

    Lookup chain: generations.api_key_hash → api_key_metadata. A missing
    metadata row is treated as opt-out (defense-in-depth).
    """
    row = history._conn.execute(
        """SELECT m.training_opt_in
           FROM generations g
           JOIN api_key_metadata m ON m.api_key_hash = g.api_key_hash
           WHERE g.id = ?""",
        (clip_id,),
    ).fetchone()
    return bool(row and row["training_opt_in"])


def _bearer_hash(history: HistoryStore, clip_id: str) -> str | None:
    row = history._conn.execute(
        "SELECT api_key_hash FROM generations WHERE id = ?",
        (clip_id,),
    ).fetchone()
    return row["api_key_hash"] if row else None


def _validator_present(history: HistoryStore, clip_id: str) -> bool:
    row = history._conn.execute(
        "SELECT validator_score FROM generations WHERE id = ?",
        (clip_id,),
    ).fetchone()
    return bool(row and row["validator_score"] is not None)


def _cross_bearer_consent(history: HistoryStore, rater_hash: str) -> bool:
    row = history._conn.execute(
        "SELECT cross_bearer_rating_consent_at FROM api_key_metadata "
        "WHERE api_key_hash = ?",
        (rater_hash,),
    ).fetchone()
    return bool(row and row["cross_bearer_rating_consent_at"] is not None)


def _evaluate_pair(
    history: HistoryStore,
    pair_row,
) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``ok=True`` → clear the staging column."""
    chosen = pair_row["chosen_clip_id"]
    rejected = pair_row["rejected_clip_id"]
    pair_id = pair_row["pair_id"]

    if not _bearer_opted_in(history, chosen):
        return False, "chosen_bearer_opt_out"
    if rejected is not None and not _bearer_opted_in(history, rejected):
        return False, "rejected_bearer_opt_out"

    if not _validator_present(history, chosen):
        return False, "chosen_validator_null"
    if rejected is not None and not _validator_present(history, rejected):
        return False, "rejected_validator_null"

    rating = history._conn.execute(
        """SELECT rater_api_key_hash, rating_payload_json,
                  validator_visible_at_rating
           FROM human_ratings
           WHERE pair_id = ? AND retracted_at IS NULL
             AND superseded_by IS NULL
           ORDER BY created_at DESC
           LIMIT 1""",
        (pair_id,),
    ).fetchone()
    if rating is None:
        return False, "no_active_rating"

    if rating["validator_visible_at_rating"]:
        return False, "validator_visible_at_rating"

    payload_visible = False
    raw = rating["rating_payload_json"]
    if raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("validator_visible_at_rating"):
            payload_visible = True
    if payload_visible:
        return False, "validator_visible_at_rating_payload"

    rater = rating["rater_api_key_hash"]
    chosen_bearer = _bearer_hash(history, chosen)
    rejected_bearer = _bearer_hash(history, rejected) if rejected else None
    if rater not in {chosen_bearer, rejected_bearer}:
        if not _cross_bearer_consent(history, rater):
            return False, "cross_bearer_consent_revoked"

    return True, "ok"


def construct(history: HistoryStore, *, dry_run: bool) -> dict[str, int]:
    now = time.time()
    pause_until = _is_paused(history, now)
    if pause_until is not None:
        logger.info(
            "human_rating_paused_until=%s (now=%s); skipping run",
            pause_until,
            now,
        )
        return {"paused": 1, "cleared": 0, "deleted": 0, "scanned": 0}

    rows = history._conn.execute(
        """SELECT pair_id, chosen_clip_id, rejected_clip_id,
                  validator_version, signal_strength
           FROM preference_pairs
           WHERE signal_source = 'human_rating'
             AND pending_construction_until IS NOT NULL
             AND pending_construction_until < ?""",
        (now,),
    ).fetchall()

    cleared = 0
    deleted = 0
    for row in rows:
        ok, reason = _evaluate_pair(history, row)
        if ok:
            cleared += 1
            if dry_run:
                logger.info(
                    "[dry-run] would clear pending_construction_until pair_id=%d",
                    row["pair_id"],
                )
            else:
                history._conn.execute(
                    "UPDATE preference_pairs "
                    "SET pending_construction_until = NULL "
                    "WHERE pair_id = ?",
                    (row["pair_id"],),
                )
        else:
            deleted += 1
            if dry_run:
                logger.info(
                    "[dry-run] would DELETE pair_id=%d (reason=%s)",
                    row["pair_id"],
                    reason,
                )
            else:
                history._conn.execute(
                    "DELETE FROM preference_pairs WHERE pair_id = ?",
                    (row["pair_id"],),
                )
                logger.info(
                    "deleted pair_id=%d (reason=%s)", row["pair_id"], reason
                )

    if not dry_run:
        history._conn.commit()
    logger.info(
        "scanned=%d cleared=%d deleted=%d (dry_run=%s)",
        len(rows),
        cleared,
        deleted,
        dry_run,
    )
    return {
        "paused": 0,
        "cleared": cleared,
        "deleted": deleted,
        "scanned": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the actions that would be taken without writing to the DB.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Override the history DB path (defaults to config.HISTORY_DB).",
    )
    args = parser.parse_args()

    db_path = args.db if args.db is not None else config.HISTORY_DB
    history = HistoryStore(db_path=db_path)
    construct(history, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
