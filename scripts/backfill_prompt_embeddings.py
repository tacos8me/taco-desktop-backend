#!/usr/bin/env python3
"""v1.18.0-rc2 — backfill prompt embeddings into the sqlite-vec virtual
table for retrieval (Phase B).

Idempotent: queries `generations` rows whose `id` is not already present in
`clip_embeddings`, restricted to opt-in bearers via
`api_key_metadata.training_opt_in = 1`. Inserts in batches of 64 (matches
llama-swap's optimal batch size). Resumable on crash (next run re-queries
the missing-ids set and continues). Rate-limits via `--sleep-ms` to avoid
saturating llama-swap when other endpoints are using it.

Usage::

    uv run python scripts/backfill_prompt_embeddings.py
    uv run python scripts/backfill_prompt_embeddings.py --dry-run
    uv run python scripts/backfill_prompt_embeddings.py --limit 1000
    uv run python scripts/backfill_prompt_embeddings.py --rebuild
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Make the repo root importable when invoked from anywhere.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from chat_manager import ChatManager, EMBEDDING_MODEL_VERSION  # noqa: E402
import history_store  # noqa: E402
from history_store import HistoryStore  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_prompt_embeddings")


_BATCH_SIZE = 64


async def _backfill(
    *,
    dry_run: bool,
    limit: int | None,
    rebuild: bool,
    sleep_ms: int,
) -> None:
    history = HistoryStore()
    # Read the module attribute after the constructor — `SQLITE_VEC_AVAILABLE`
    # is mutated inside `HistoryStore._load_sqlite_vec()`. Importing the name
    # directly would snapshot the pre-construction `False`.
    if not history_store.SQLITE_VEC_AVAILABLE:
        logger.error(
            "sqlite-vec extension not loaded (%s) — cannot backfill",
            history_store.SQLITE_VEC_LOAD_ERROR,
        )
        return

    if rebuild and not dry_run:
        logger.warning("--rebuild: deleting all existing rows from clip_embeddings")
        history._conn.execute("DELETE FROM clip_embeddings")
        history._conn.commit()

    chat = ChatManager()
    chat.load()

    total_done = 0
    total_skipped = 0
    started = time.perf_counter()

    while True:
        select_sql = """
            SELECT g.id, g.prompt
            FROM generations g
            JOIN api_key_metadata m ON g.api_key_hash = m.api_key_hash
            WHERE g.id NOT IN (SELECT id FROM clip_embeddings)
              AND m.training_opt_in = 1
              AND g.prompt IS NOT NULL
              AND g.prompt != ''
            LIMIT ?
        """
        rows = history._conn.execute(select_sql, (_BATCH_SIZE,)).fetchall()
        if not rows:
            break
        if dry_run:
            logger.info(
                "dry_run: would embed %d rows (sample: %s)",
                len(rows),
                [r["id"] for r in rows[:3]],
            )
            total_done += len(rows)
            if limit is not None and total_done >= limit:
                break
            # In dry-run we have to bail out of the loop because nothing
            # actually changes — otherwise we'd spin forever.
            break

        prompts = [r["prompt"] for r in rows]
        try:
            embeddings = await chat.embed_batch(prompts)
        except Exception as exc:
            logger.error("embed_batch failed (will retry next run): %s", exc)
            break

        try:
            history._conn.executemany(
                """INSERT OR IGNORE INTO clip_embeddings
                   (id, embedding, embedding_model_version)
                   VALUES (?, ?, ?)""",
                [
                    (r["id"], emb, EMBEDDING_MODEL_VERSION)
                    for r, emb in zip(rows, embeddings)
                ],
            )
            history._conn.commit()
        except Exception as exc:
            logger.error("INSERT failed: %s", exc)
            break

        total_done += len(rows)
        logger.info(
            "backfilled %d (cum %d) in %.1fs",
            len(rows),
            total_done,
            time.perf_counter() - started,
        )
        if limit is not None and total_done >= limit:
            logger.info("--limit %d reached; stopping", limit)
            break
        if sleep_ms:
            await asyncio.sleep(sleep_ms / 1000.0)

    logger.info(
        "done: %d rows embedded, %d skipped, %.1fs total",
        total_done,
        total_skipped,
        time.perf_counter() - started,
    )

    await chat.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be embedded without writing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many rows (default: no limit)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="delete + re-embed all rows (destructive)",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=500,
        help="ms to sleep between batches (default 500)",
    )
    args = parser.parse_args()

    asyncio.run(_backfill(
        dry_run=args.dry_run,
        limit=args.limit,
        rebuild=args.rebuild,
        sleep_ms=args.sleep_ms,
    ))


if __name__ == "__main__":
    main()
