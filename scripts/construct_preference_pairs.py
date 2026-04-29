#!/usr/bin/env python3
"""v1.18.0-rc3 — Phase C preference-pair ETL.

Weekly cron. Walks ``generations`` + ``composition_clips`` + retake
provenance (``parent_clip_id``) to construct rows in ``preference_pairs``
across four signal sources, version-scoped to a single
``validator_version`` so cross-version pairs never enter training.

Sources (signal_strength constant per source):

  1. ``user_retake`` (0.9)   — retake explicitly chose new clip over
                               its parent. r.id chosen, p.id rejected.
  2. ``composition_kept``    — clip kept in a final composition wins
     (0.5)                     against same-shot_config_key clips that
                               weren't kept.
  3. ``validator_pass`` (0.7)— same-shot_config_key cohort, top
                               validator_score wins against bottom.
  4. ``validator_fail`` (0.3)— synthetic negatives: any clip with
                               validator_score < 0.45 ("retake"
                               recommendation) is paired as rejected
                               against same-shot_config_key clips with
                               score >= 0.65.

All sources MUST:

  - Filter by ``validator_version`` (schema v4 column on
    preference_pairs). Pairs across mixed versions are excluded.
  - Filter by ``api_key_metadata.training_opt_in = 1``. Opt-out
    bearers' clips never enter training.
  - Use ``INSERT OR IGNORE`` against ``idx_pp_unique_pair_source``
    (UNIQUE on (chosen_clip_id, rejected_clip_id, signal_source))
    so re-runs are idempotent.
  - Use ``--since-watermark`` for incremental runs (default).

Idempotence model: a watermark (epoch float) lives at
``/mnt/nvme-1/servers/taco-backend/.preference_pairs_watermark``.
Each run reads the watermark, includes only ``generations`` rows with
``created_at > watermark``, then writes ``time.time()`` back on success.
``--full-rebuild`` ignores the watermark.

Usage::

    uv run python scripts/construct_preference_pairs.py
    uv run python scripts/construct_preference_pairs.py --dry-run
    uv run python scripts/construct_preference_pairs.py --full-rebuild
    uv run python scripts/construct_preference_pairs.py --validator-version 1.17.0-rc5
    uv run python scripts/construct_preference_pairs.py --source user_retake
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
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
logger = logging.getLogger("construct_preference_pairs")


WATERMARK_PATH = _ROOT / ".preference_pairs_watermark"

VALIDATOR_PASS_THRESHOLD = 0.65   # tier-3 verdict pass cutoff (mirror validator.py)
VALIDATOR_RETAKE_THRESHOLD = 0.45  # tier-3 verdict retake cutoff

SIGNAL_STRENGTH = {
    "user_retake": 0.9,
    "composition_kept": 0.5,
    "validator_pass": 0.7,
    "validator_fail": 0.3,
}

ALL_SOURCES = ("user_retake", "composition_kept", "validator_pass", "validator_fail")


# ---------------------------------------------------------------------------
# Watermark helpers
# ---------------------------------------------------------------------------


def get_last_watermark() -> float:
    if not WATERMARK_PATH.exists():
        return 0.0
    try:
        return float(WATERMARK_PATH.read_text().strip())
    except (ValueError, OSError):
        logger.warning("watermark file unreadable; treating as 0.0")
        return 0.0


def set_last_watermark(value: float) -> None:
    tmp = WATERMARK_PATH.with_suffix(".tmp")
    tmp.write_text(f"{value:.6f}\n")
    tmp.replace(WATERMARK_PATH)


# ---------------------------------------------------------------------------
# Source 1: user_retake
# ---------------------------------------------------------------------------


def construct_user_retake_pairs(
    conn: sqlite3.Connection,
    *,
    validator_version: str,
    since_watermark: float,
    dry_run: bool,
) -> int:
    """Retake winners (chosen) vs their parents (rejected).

    A retake explicitly carries ``parent_clip_id`` (set in
    ``server.py /v2/retake``). The signal is strong: the operator looked
    at the parent, hit retake, and the new clip is what they chose to
    keep. Both clips must share validator_version and both bearers must
    be opt-in.
    """
    sql_select = """
        SELECT r.id AS chosen, r.parent_clip_id AS rejected, r.created_at
        FROM generations r
        JOIN generations p ON p.id = r.parent_clip_id
        JOIN api_key_metadata mr ON mr.api_key_hash = r.api_key_hash
        JOIN api_key_metadata mp ON mp.api_key_hash = p.api_key_hash
        WHERE r.parent_clip_id IS NOT NULL
          AND r.validator_version = ?
          AND p.validator_version = ?
          AND mr.training_opt_in = 1
          AND mp.training_opt_in = 1
          AND r.created_at > ?
    """
    rows = conn.execute(
        sql_select, (validator_version, validator_version, since_watermark)
    ).fetchall()
    if dry_run:
        return len(rows)
    if not rows:
        return 0
    now = time.time()
    cur = conn.executemany(
        """INSERT OR IGNORE INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at)
           VALUES (?, ?, 'user_retake', ?, ?, ?)""",
        [
            (
                r["chosen"], r["rejected"],
                SIGNAL_STRENGTH["user_retake"], validator_version, now,
            )
            for r in rows
        ],
    )
    conn.commit()
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(rows)


# ---------------------------------------------------------------------------
# Source 2: composition_kept
# ---------------------------------------------------------------------------


def construct_composition_kept_pairs(
    conn: sqlite3.Connection,
    *,
    validator_version: str,
    since_watermark: float,
    dry_run: bool,
) -> int:
    """Clips kept in a final composition WIN vs same-shot_config_key
    clips that weren't kept.

    Signal source: ``composition_clips`` is the canonical lineage table
    populated by ``POST /v2/compositions/{id}/export`` (v1.17.0-rc1).
    Same-shot_config_key cohort lookup ensures we're comparing
    alternates of the same shot, not unrelated clips.
    """
    sql_select = """
        SELECT cc.clip_history_id AS chosen, g.id AS rejected, g.created_at
        FROM composition_clips cc
        JOIN generations gc ON gc.id = cc.clip_history_id
        JOIN generations g  ON g.shot_config_key = gc.shot_config_key
                            AND g.id != gc.id
        JOIN api_key_metadata mc ON mc.api_key_hash = gc.api_key_hash
        JOIN api_key_metadata mg ON mg.api_key_hash = g.api_key_hash
        LEFT JOIN composition_clips cc2
            ON cc2.clip_history_id = g.id
        WHERE cc.was_final = 1
          AND gc.shot_config_key IS NOT NULL
          AND gc.validator_version = ?
          AND g.validator_version = ?
          AND mc.training_opt_in = 1
          AND mg.training_opt_in = 1
          AND cc2.clip_history_id IS NULL  -- rejected wasn't kept anywhere
          AND cc.created_at > ?
    """
    rows = conn.execute(
        sql_select, (validator_version, validator_version, since_watermark)
    ).fetchall()
    if dry_run:
        return len(rows)
    if not rows:
        return 0
    now = time.time()
    cur = conn.executemany(
        """INSERT OR IGNORE INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at)
           VALUES (?, ?, 'composition_kept', ?, ?, ?)""",
        [
            (
                r["chosen"], r["rejected"],
                SIGNAL_STRENGTH["composition_kept"], validator_version, now,
            )
            for r in rows
        ],
    )
    conn.commit()
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(rows)


# ---------------------------------------------------------------------------
# Source 3: validator_pass
# ---------------------------------------------------------------------------


def construct_validator_pass_pairs(
    conn: sqlite3.Connection,
    *,
    validator_version: str,
    since_watermark: float,
    dry_run: bool,
) -> int:
    """Within same-shot_config_key cohort: passing clips beat warning clips.

    Constructed pairs: each (chosen, rejected) where
    ``chosen.validator_score >= 0.65`` and
    ``rejected.validator_score < 0.65 AND >= 0.45``. Excludes
    sub-0.45 (those go to validator_fail with the stronger negative signal).
    Both must share shot_config_key and validator_version.
    """
    sql_select = """
        SELECT a.id AS chosen, b.id AS rejected, a.created_at
        FROM generations a
        JOIN generations b ON b.shot_config_key = a.shot_config_key
                           AND b.id != a.id
        JOIN api_key_metadata ma ON ma.api_key_hash = a.api_key_hash
        JOIN api_key_metadata mb ON mb.api_key_hash = b.api_key_hash
        WHERE a.shot_config_key IS NOT NULL
          AND a.validator_score >= ?
          AND b.validator_score >= ?
          AND b.validator_score < ?
          AND a.validator_version = ?
          AND b.validator_version = ?
          AND ma.training_opt_in = 1
          AND mb.training_opt_in = 1
          AND a.created_at > ?
    """
    rows = conn.execute(
        sql_select,
        (
            VALIDATOR_PASS_THRESHOLD,
            VALIDATOR_RETAKE_THRESHOLD,
            VALIDATOR_PASS_THRESHOLD,
            validator_version,
            validator_version,
            since_watermark,
        ),
    ).fetchall()
    if dry_run:
        return len(rows)
    if not rows:
        return 0
    now = time.time()
    cur = conn.executemany(
        """INSERT OR IGNORE INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at)
           VALUES (?, ?, 'validator_pass', ?, ?, ?)""",
        [
            (
                r["chosen"], r["rejected"],
                SIGNAL_STRENGTH["validator_pass"], validator_version, now,
            )
            for r in rows
        ],
    )
    conn.commit()
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(rows)


# ---------------------------------------------------------------------------
# Source 4: validator_fail (synthetic negatives)
# ---------------------------------------------------------------------------


def construct_validator_fail_pairs(
    conn: sqlite3.Connection,
    *,
    validator_version: str,
    since_watermark: float,
    dry_run: bool,
) -> int:
    """Synthetic-negative pairing: low-validator clips paired as rejected
    against high-validator same-cohort clips.

    Weakest signal (0.3) — validator_score is a noisy proxy for
    operator preference, but at the extremes it's reliable enough
    to seed early-corpus training.
    """
    sql_select = """
        SELECT a.id AS chosen, b.id AS rejected, b.created_at
        FROM generations a
        JOIN generations b ON b.shot_config_key = a.shot_config_key
                           AND b.id != a.id
        JOIN api_key_metadata ma ON ma.api_key_hash = a.api_key_hash
        JOIN api_key_metadata mb ON mb.api_key_hash = b.api_key_hash
        WHERE a.shot_config_key IS NOT NULL
          AND a.validator_score >= ?
          AND b.validator_score < ?
          AND a.validator_version = ?
          AND b.validator_version = ?
          AND ma.training_opt_in = 1
          AND mb.training_opt_in = 1
          AND b.created_at > ?
    """
    rows = conn.execute(
        sql_select,
        (
            VALIDATOR_PASS_THRESHOLD,
            VALIDATOR_RETAKE_THRESHOLD,
            validator_version,
            validator_version,
            since_watermark,
        ),
    ).fetchall()
    if dry_run:
        return len(rows)
    if not rows:
        return 0
    now = time.time()
    cur = conn.executemany(
        """INSERT OR IGNORE INTO preference_pairs
           (chosen_clip_id, rejected_clip_id, signal_source, signal_strength,
            validator_version, created_at)
           VALUES (?, ?, 'validator_fail', ?, ?, ?)""",
        [
            (
                r["chosen"], r["rejected"],
                SIGNAL_STRENGTH["validator_fail"], validator_version, now,
            )
            for r in rows
        ],
    )
    conn.commit()
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


_SOURCE_FNS = {
    "user_retake": construct_user_retake_pairs,
    "composition_kept": construct_composition_kept_pairs,
    "validator_pass": construct_validator_pass_pairs,
    "validator_fail": construct_validator_fail_pairs,
}


def run(
    *,
    validator_version: str,
    full_rebuild: bool,
    dry_run: bool,
    sources: tuple[str, ...] = ALL_SOURCES,
    db_path: Path | None = None,
) -> dict[str, int]:
    """Run the ETL. Returns per-source row counts (constructed or
    would-be-constructed in dry-run)."""
    history = HistoryStore(db_path=db_path) if db_path else HistoryStore()
    since = 0.0 if full_rebuild else get_last_watermark()
    logger.info(
        "etl start: validator_version=%s since_watermark=%.3f full_rebuild=%s dry_run=%s sources=%s",
        validator_version, since, full_rebuild, dry_run, sources,
    )
    counts: dict[str, int] = {}
    for source in sources:
        fn = _SOURCE_FNS[source]
        n = fn(
            history._conn,
            validator_version=validator_version,
            since_watermark=since,
            dry_run=dry_run,
        )
        counts[source] = n
        logger.info("source=%s rows=%d (%s)", source, n, "would-insert" if dry_run else "inserted")
    if not dry_run:
        set_last_watermark(time.time())
        logger.info("watermark advanced to now")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since-watermark",
        action="store_true",
        default=True,
        help="(default) read .preference_pairs_watermark and process only newer rows",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="ignore watermark, reprocess everything (idempotent via INSERT OR IGNORE)",
    )
    parser.add_argument(
        "--validator-version",
        default=None,
        help=f"override (default: config.VALIDATOR_VERSION = {config.VALIDATOR_VERSION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report counts without writing or advancing watermark",
    )
    parser.add_argument(
        "--source",
        choices=ALL_SOURCES,
        default=None,
        help="filter to one signal source (default: all four)",
    )
    args = parser.parse_args()

    validator_version = args.validator_version or config.VALIDATOR_VERSION
    sources = (args.source,) if args.source else ALL_SOURCES

    counts = run(
        validator_version=validator_version,
        full_rebuild=args.full_rebuild,
        dry_run=args.dry_run,
        sources=sources,
    )

    total = sum(counts.values())
    print(
        "pair construction summary: " + ", ".join(
            f"{src}={counts.get(src, 0)}" for src in ALL_SOURCES
        ) + f", total={total}"
    )
    if args.dry_run:
        print("dry-run: no rows written, watermark unchanged")


if __name__ == "__main__":
    main()
