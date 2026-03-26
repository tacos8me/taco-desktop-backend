"""SQLite-backed timeline composition storage, keyed by hashed API key."""

import hashlib
import json
import logging
import secrets
import sqlite3
import time
from pathlib import Path

import config

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS compositions (
    id TEXT PRIMARY KEY,
    api_key_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comp_key ON compositions(api_key_hash, updated_at DESC);
"""


class CompositionStore:
    def __init__(self, db_path=None):
        self._db_path = str(db_path or config.HISTORY_DB)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        logger.info("CompositionStore opened at %s", self._db_path)

    def _hash_key(self, api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create(self, api_key: str, name: str, data: dict) -> dict:
        comp_id = secrets.token_urlsafe(16)
        now = time.time()
        self._conn.execute(
            """INSERT INTO compositions (id, api_key_hash, name, data, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (comp_id, self._hash_key(api_key), name, json.dumps(data), now, now),
        )
        self._conn.commit()
        return {"id": comp_id, "created_at": now}

    def list(self, api_key: str, limit: int = 50, offset: int = 0) -> list[dict]:
        rows = self._conn.execute(
            """SELECT id, name, data, created_at, updated_at FROM compositions
               WHERE api_key_hash = ?
               ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
            (self._hash_key(api_key), limit, offset),
        ).fetchall()
        results = []
        for row in rows:
            data = json.loads(row["data"])
            clips = data.get("clips", [])
            total_duration = sum(c.get("duration", 0) for c in clips)
            results.append({
                "id": row["id"],
                "name": row["name"],
                "clip_count": len(clips),
                "total_duration": total_duration,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return results

    def get(self, comp_id: str, api_key: str) -> dict | None:
        row = self._conn.execute(
            """SELECT id, name, data, created_at, updated_at FROM compositions
               WHERE id = ? AND api_key_hash = ?""",
            (comp_id, self._hash_key(api_key)),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "data": json.loads(row["data"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update(self, comp_id: str, api_key: str, name: str, data: dict) -> bool:
        now = time.time()
        cursor = self._conn.execute(
            """UPDATE compositions SET name = ?, data = ?, updated_at = ?
               WHERE id = ? AND api_key_hash = ?""",
            (name, json.dumps(data), now, comp_id, self._hash_key(api_key)),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete(self, comp_id: str, api_key: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM compositions WHERE id = ? AND api_key_hash = ?",
            (comp_id, self._hash_key(api_key)),
        )
        self._conn.commit()
        return cursor.rowcount > 0
