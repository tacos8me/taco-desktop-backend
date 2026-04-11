"""SQLite-backed generation history, keyed by hashed API key."""

from __future__ import annotations

import hashlib
import io
import logging
import sqlite3
import time
from pathlib import Path

from PIL import Image

import config

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
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
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_key_hash ON generations(api_key_hash, created_at DESC);
"""


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _is_mp4_bytes(data: bytes) -> bool:
    """Heuristic MP4 detection via the ISO base media `ftyp` box.

    Any ISO-BMFF container (MP4, MOV, M4A, etc.) has a 4-byte big-endian box
    size followed by the 4-byte type `ftyp` at offset 4. We don't care about
    the brand — just whether this is "a video-like container we should try
    to decode with PyAV rather than PIL".
    """
    return len(data) >= 12 and data[4:8] == b"ftyp"


def _first_video_frame_as_pil(video_bytes: bytes) -> Image.Image | None:
    """Decode the first video frame of an MP4 as a PIL.Image in RGB.

    Returns None on any failure — caller falls through to the warning path.
    Uses PyAV (already a dependency via `media_io.encode_video`) with an
    in-memory `io.BytesIO` container so we don't touch the filesystem.
    """
    try:
        import av  # local import keeps module load fast when only images are thumbnailed
    except Exception:
        return None
    try:
        with av.open(io.BytesIO(video_bytes), mode="r") as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            # thread_type="AUTO" speeds up decode of short clips on multi-core CPUs
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                # to_image() returns a PIL.Image directly; PyAV does the RGB conversion
                return frame.to_image()
    except Exception:
        return None
    return None


def _make_thumbnail(media_bytes: bytes, upload_id: str) -> str | None:
    """Create a 256px-wide JPEG thumbnail for image OR video bytes.

    - Images (WEBP/PNG/JPEG): loaded via `PIL.Image.open`
    - Videos (MP4/MOV/etc., detected via `ftyp` box): first frame extracted
      via `PyAV`, then thumbnailed the same way

    Returns the thumbnail's storage id (under `config.THUMBNAIL_DIR`) or
    `None` on any failure — callers must treat `None` as "no thumbnail,
    show a placeholder".
    """
    try:
        if _is_mp4_bytes(media_bytes):
            img = _first_video_frame_as_pil(media_bytes)
            if img is None:
                logger.warning(
                    "Failed to create thumbnail for %s: MP4-like container but first "
                    "frame decode returned None", upload_id,
                )
                return None
        else:
            img = Image.open(io.BytesIO(media_bytes))
            img.load()  # force decode so we catch errors here, not later

        if img.width == 0 or img.height == 0:
            return None
        # Convert to RGB before JPEG save — handles RGBA from WEBP, palette
        # modes from PNG, and the PyAV frame (which is already RGB but safe to force)
        img = img.convert("RGB")
        ratio = 256 / img.width
        img = img.resize((256, max(1, int(img.height * ratio))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        thumb_id = f"thumb_{upload_id}"
        thumb_path = config.THUMBNAIL_DIR / thumb_id
        thumb_path.write_bytes(buf.getvalue())
        return thumb_id
    except Exception:
        logger.warning("Failed to create thumbnail for %s", upload_id, exc_info=True)
        return None


class HistoryStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = str(db_path or config.HISTORY_DB)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        # WAL mode: readers never block the single writer. Without this, the
        # /v2/history list endpoint stalls behind the queue worker's thumbnail
        # write. WAL is persisted in the DB header — running it on an existing
        # DELETE-journal DB performs an online conversion. Creates .db-wal and
        # .db-shm sidecar files (see .gitignore).
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()
        logger.info("History DB opened at %s", self._db_path)

    def save(
        self,
        job_id: str,
        api_key: str,
        job_type: str,
        prompt: str,
        model: str | None,
        width: int,
        height: int,
        turbo: bool,
        status: str,
        result_uri: str | None,
        result_bytes: bytes | None,
        created_at: float,
        completed_at: float | None,
        error: str | None = None,
    ) -> None:
        thumb_uri = None
        if result_bytes and result_uri:
            upload_id = result_uri.replace("storage://", "")
            thumb_id = _make_thumbnail(result_bytes, upload_id)
            if thumb_id:
                thumb_uri = f"thumb://{thumb_id}"

        self._conn.execute(
            """INSERT OR REPLACE INTO generations
               (id, api_key_hash, job_type, prompt, model, width, height, turbo,
                status, result_uri, thumbnail_uri, created_at, completed_at, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                _hash_key(api_key),
                job_type,
                prompt,
                model,
                width,
                height,
                1 if turbo else 0,
                status,
                result_uri,
                thumb_uri,
                created_at,
                completed_at,
                error,
            ),
        )
        self._conn.commit()

    def list(
        self, api_key: str, limit: int = 50, offset: int = 0, job_type: str | None = None
    ) -> list[dict]:
        key_hash = _hash_key(api_key)
        # Support category shortcuts: "image" → all image types, "video" → all video types
        if job_type == "image":
            rows = self._conn.execute(
                """SELECT * FROM generations
                   WHERE api_key_hash = ? AND job_type LIKE '%-image%'
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (key_hash, limit, offset),
            ).fetchall()
        elif job_type == "video":
            rows = self._conn.execute(
                """SELECT * FROM generations
                   WHERE api_key_hash = ? AND job_type LIKE '%-video%'
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (key_hash, limit, offset),
            ).fetchall()
        elif job_type:
            rows = self._conn.execute(
                """SELECT * FROM generations
                   WHERE api_key_hash = ? AND job_type = ?
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (key_hash, job_type, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM generations
                   WHERE api_key_hash = ?
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (key_hash, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, generation_id: str, api_key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM generations WHERE id = ? AND api_key_hash = ?",
            (generation_id, _hash_key(api_key)),
        ).fetchone()
        return dict(row) if row else None

    def cleanup(self, max_age_days: int | None = None) -> int:
        """Remove old entries and their files. Returns count removed."""
        days = max_age_days or config.HISTORY_RETENTION_DAYS
        cutoff = time.time() - (days * 86400)
        rows = self._conn.execute(
            "SELECT result_uri, thumbnail_uri FROM generations WHERE created_at < ?",
            (cutoff,),
        ).fetchall()

        count = 0
        for row in rows:
            if row["result_uri"]:
                uid = row["result_uri"].replace("storage://", "")
                p = config.UPLOAD_DIR / uid
                if p.exists():
                    p.unlink()
            if row["thumbnail_uri"]:
                tid = row["thumbnail_uri"].replace("thumb://", "")
                p = config.THUMBNAIL_DIR / tid
                if p.exists():
                    p.unlink()
            count += 1

        self._conn.execute("DELETE FROM generations WHERE created_at < ?", (cutoff,))
        self._conn.commit()
        if count:
            logger.info("Cleaned up %d history entries older than %d days", count, days)
        return count
