"""SQLite-backed generation history, keyed by hashed API key."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
import time
from pathlib import Path

from PIL import Image

import config

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 3

# Blob caps — a single rogue request can easily serialize multi-MB payloads
# (LoRA paths list, keyframe images-as-bytes if a caller mis-uses the field,
# gen_config dumps with giant sigma tables). Left uncapped, the history DB
# inflates until WAL sync pressure starves /v2/history reads.
_HISTORY_PARAMS_MAX_BYTES = 100_000  # 100 KB
_HISTORY_GEN_CONFIG_MAX_BYTES = 50_000  # 50 KB
# Truncation sentinel includes up to this many bytes of the original payload
# so the client still has *something* meaningful to display.
_HISTORY_TRUNCATED_PREVIEW_BYTES = 4096

# WAL checkpoint cadence. Without this, .db-wal grows unbounded between the
# nightly sqlite VACUUM cron; we've observed it past 4 GB on noisy days.
_HISTORY_WAL_CHECKPOINT_EVERY = 500
_HISTORY_WAL_WARN_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

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

# v3 (v1.17.0-rc1): five new tables created during _migrate() on the
# first user_version 2→3 bump. Each uses CREATE TABLE IF NOT EXISTS so
# re-running the migration after a partial failure is safe.
_SCHEMA_V3_TABLES = """
CREATE TABLE IF NOT EXISTS composition_clips (
    comp_id TEXT NOT NULL,
    clip_history_id TEXT,
    position INTEGER NOT NULL,
    was_final INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    PRIMARY KEY (comp_id, clip_history_id, position)
);
CREATE INDEX IF NOT EXISTS idx_comp_clips_clip ON composition_clips(clip_history_id);

CREATE TABLE IF NOT EXISTS validator_runs (
    run_id TEXT PRIMARY KEY,
    video_uri TEXT,
    video_sha256 TEXT,
    payload_json TEXT,
    latency_s REAL,
    validator_version TEXT,
    ran_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_validator_runs_video_version
    ON validator_runs(video_sha256, validator_version);

CREATE TABLE IF NOT EXISTS preference_pairs (
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chosen_clip_id TEXT,
    rejected_clip_id TEXT,
    signal_source TEXT,
    signal_strength REAL,
    used_in_training_run_id TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS training_runs (
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

CREATE TABLE IF NOT EXISTS api_key_metadata (
    api_key_hash TEXT PRIMARY KEY,
    training_opt_in INTEGER NOT NULL DEFAULT 1,
    tier TEXT DEFAULT 'pro',
    notes TEXT,
    created_at REAL,
    updated_at REAL
);
"""


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _path_to_uri(path: str | None) -> str | None:
    """Rewrite an on-disk uploads path to its canonical ``storage://<uuid>`` URI.

    Leaves paths outside ``config.UPLOAD_DIR`` verbatim (LoRA files under
    ``/mnt/nvme-1/servers/taco-backend/flux_loras``, ACE's ``/tmp/ace-*.wav``
    staging files, etc.) so the history row still records where the input
    came from even when it was never uploaded via the storage API.
    """
    if path is None:
        return None
    try:
        p = Path(path)
    except (TypeError, ValueError):
        return path
    if p.parent == config.UPLOAD_DIR:
        return f"storage://{p.name}"
    return path


def _sanitize_params_for_history(job_type: str, params: dict) -> dict:
    """Return a copy of ``params`` safe to persist as ``params_json``.

    - Drops ``_``-prefixed internal markers the dispatcher attaches.
    - Rewrites known on-disk path fields (``image_path``, ``audio_path``,
      ``video_path``, ``source_audio_path``, ``reference_audio_path``) to
      their ``*_uri`` counterparts via :func:`_path_to_uri`.
    - Rewrites ``image_paths`` list to ``image_uris``.
    - Rewrites each dict in a ``keyframes`` list, swapping ``image_path`` for
      ``image_uri``.
    - Leaves ``lora_path`` alone — LoRAs live outside ``UPLOAD_DIR``; the
      raw-request blob carries the stable ``{id, strength}`` shape.
    """
    out: dict = {}
    for key, value in params.items():
        if key.startswith("_"):
            continue
        out[key] = value

    for scalar_key in (
        "image_path",
        "audio_path",
        "video_path",
        "source_audio_path",
        "reference_audio_path",
    ):
        if scalar_key in out:
            uri_key = scalar_key.replace("_path", "_uri")
            out[uri_key] = _path_to_uri(out.pop(scalar_key))

    if "image_paths" in out:
        raw_list = out.pop("image_paths") or []
        out["image_uris"] = [_path_to_uri(p) for p in raw_list]

    if "keyframes" in out and isinstance(out["keyframes"], list):
        rewritten = []
        for kf in out["keyframes"]:
            if isinstance(kf, dict) and "image_path" in kf:
                kf_copy = dict(kf)
                kf_copy["image_uri"] = _path_to_uri(kf_copy.pop("image_path"))
                rewritten.append(kf_copy)
            else:
                rewritten.append(kf)
        out["keyframes"] = rewritten

    return out


def _is_mp4_bytes(data: bytes) -> bool:
    """Heuristic MP4 detection via the ISO base media `ftyp` box.

    Any ISO-BMFF container (MP4, MOV, M4A, etc.) has a 4-byte big-endian box
    size followed by the 4-byte type `ftyp` at offset 4. We don't care about
    the brand — just whether this is "a video-like container we should try
    to decode with PyAV rather than PIL".
    """
    return len(data) >= 12 and data[4:8] == b"ftyp"


def _probe_video_dims(video_bytes: bytes) -> tuple[int, int, float]:
    """Return ``(width, height, fps)`` from an MP4-like container via PyAV.

    Used by the v1.14.0 ``/v2/video-hdr`` endpoint to size the IC-LoRA
    reference latent to the exact source resolution (HDR transform preserves
    canvas; outpaint expands it). Frame count is intentionally not returned
    — callers must derive it from user-supplied ``duration * fps`` so the
    LTX 8k+1 quantization still applies.

    Raises:
        ValueError: when the bytes don't decode or have no video stream.
    """
    import av
    with av.open(io.BytesIO(video_bytes), mode="r") as container:
        if not container.streams.video:
            raise ValueError("no_video_stream")
        s = container.streams.video[0]
        return (int(s.width), int(s.height), float(s.average_rate or 24))


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


def _extract_frames_as_pils(video_bytes: bytes, indices: list[int]) -> list[Image.Image]:
    """Decode specified frame indices (sorted, deduped) in one pass.

    Single PyAV iteration over the video stream that collects only the
    requested frame indices and short-circuits once the largest index is
    reached. Frames are returned in the input-list order.

    Raises:
        IndexError: if any requested index exceeds the stream length.
        RuntimeError: on decode failure or when the container has no video stream.
    """
    import av
    import io as _io
    wanted = set(indices)
    max_idx = indices[-1]
    collected: dict[int, Image.Image] = {}
    try:
        with av.open(_io.BytesIO(video_bytes), mode="r") as container:
            if not container.streams.video:
                raise RuntimeError("no_video_stream")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    collected[i] = frame.to_image()
                    if i == max_idx:
                        break
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"decode_failed: {exc}") from exc
    missing = wanted - collected.keys()
    if missing:
        raise IndexError(f"frame_indices {sorted(missing)} exceed stream length")
    return [collected[i] for i in indices]


def _extract_segment_as_mp4(video_bytes: bytes, start_frame: int, num_frames: int) -> tuple[bytes, int, int, float]:
    """v1.12 — decode [start_frame, start_frame+num_frames) as a standalone MP4.

    Re-encodes the extracted frames as H.264 MP4 bytes (no audio track) for
    use as multi-frame chain conditioning input (see docs/debug-v1.11.3-chain-conditioning.md
    and the v1.12 plan). Output is video-only, same width/height/fps as source.

    Returns (mp4_bytes, width, height, fps).

    Raises:
        IndexError: if start_frame + num_frames exceeds the stream length.
        RuntimeError: on decode/encode failure or when the container has no video stream.
    """
    import av
    import io as _io
    from fractions import Fraction

    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    end_frame = start_frame + num_frames
    collected: list = []  # PyAV VideoFrame objects in decode order
    width = height = 0
    fps = 24.0
    try:
        with av.open(_io.BytesIO(video_bytes), mode="r") as container:
            if not container.streams.video:
                raise RuntimeError("no_video_stream")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            fps = float(stream.average_rate) if stream.average_rate else 24.0
            for i, frame in enumerate(container.decode(stream)):
                if i >= end_frame:
                    break
                if i >= start_frame:
                    collected.append(frame)
                    if not width:
                        width = frame.width
                        height = frame.height
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"decode_failed: {exc}") from exc

    if len(collected) < num_frames:
        raise IndexError(
            f"segment_out_of_range: requested frames [{start_frame}, {end_frame}) "
            f"but stream ended at frame {start_frame + len(collected)}"
        )

    try:
        out_buf = _io.BytesIO()
        with av.open(out_buf, mode="w", format="mp4") as out_container:
            time_base = Fraction(1, int(round(fps * 1000))) if fps > 0 else Fraction(1, 24000)
            out_stream = out_container.add_stream("h264", rate=Fraction(int(round(fps * 1000)), 1000))
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = "yuv420p"
            out_stream.time_base = time_base
            out_stream.options = {"crf": "18", "preset": "fast"}
            for i, frame in enumerate(collected):
                new_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                new_frame = new_frame.reformat(format="yuv420p")
                new_frame.pts = i
                new_frame.time_base = Fraction(1, int(round(fps))) if fps > 0 else Fraction(1, 24)
                for packet in out_stream.encode(new_frame):
                    out_container.mux(packet)
            for packet in out_stream.encode(None):
                out_container.mux(packet)
        return out_buf.getvalue(), width, height, fps
    except Exception as exc:
        raise RuntimeError(f"encode_failed: {exc}") from exc


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


def _truncate_json_blob(raw: str, max_bytes: int, job_id: str, label: str) -> str:
    """Return ``raw`` unchanged if it fits; otherwise return a sentinel.

    The sentinel is itself valid JSON — clients parsing ``params_json`` /
    ``gen_config_json`` can branch on the ``__truncated__`` key instead of
    failing to decode.
    """
    if len(raw) <= max_bytes:
        return raw
    preview = raw[:_HISTORY_TRUNCATED_PREVIEW_BYTES]
    logger.info(
        "history %s blob truncated for job=%s: %d > %d bytes",
        label, job_id, len(raw), max_bytes,
    )
    return json.dumps({
        "__truncated__": True,
        "original_bytes": len(raw),
        "preview": preview,
    })


class HistoryStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = str(db_path or config.HISTORY_DB)
        self._wal_path = Path(self._db_path + "-wal")
        self._write_count = 0
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
        self._migrate()
        logger.info("History DB opened at %s", self._db_path)

    def _migrate(self) -> None:
        """Run idempotent schema migrations keyed off ``PRAGMA user_version``.

        v2 adds four nullable columns to ``generations``: ``params_json``,
        ``gen_config_json``, ``seed``, ``enhanced_prompt``. Old rows remain
        valid (all NULL for the new fields) — no backfill.

        v3 (v1.17.0-rc1) adds:
          - 11 nullable columns on ``generations`` for validator scoring,
            shot/composition lineage, applied-LoRA tracking, and the
            (lazy-fill) prompt embedding.
          - 5 new tables: ``composition_clips`` (inverted-index of clips
            included in each composition export), ``validator_runs`` (cached
            validator output keyed by video_sha256+version), ``preference_pairs``
            (chosen-vs-rejected DPO training data), ``training_runs``
            (LoRA training-job ledger), ``api_key_metadata`` (per-key
            training_opt_in flag — defaults ON globally, opt-out only).
          - 3 indexes on the new generations columns for shot-config-key,
            parent-clip and composition lookups.
        Pre-v3 rows get NULL for new columns; no backfill. The
        ``api_key_metadata`` table is seeded once from ``.api_keys`` on
        initial migration; subsequent migrations are idempotent no-ops.

        v3 column writer status (audit-as-of v1.17.0-rc5; see
        ``docs/CAPTURE_VALIDATOR.md`` §7 for the canonical roadmap):

          WRITTEN (v1.17.0-rc1+):
            - ``parent_clip_id``        — server.py /v2/retake handler
            - ``shot_uuid``             — MCP v0.7.0 forwarding
                                          (orchestrator._normalize_input)
            - ``shot_config_key``       — MCP v0.7.0 forwarding (same)

          WRITTEN (v1.17.0-rc2+):
            - ``validator_score``       — passive dispatch via
                                          server._dispatch_validator
            - ``validator_payload_json`` — same
            - ``validator_version``     — same

          DEAD-LETTER as of rc5 (no writers; retained for forward-compat):
            - ``composition_id``        — denorm convenience; the canonical
                                          lineage lives in ``composition_clips``
                                          inverse-index. Safe to drop in
                                          v1.18 if a schema bump is acceptable.

          FORWARD-LOOKING (writers ship in Phase B / Phase C):
            - ``prompt_embedding``      — Phase B: lazy-fill via llama-swap
                                          /v1/embeddings once that endpoint
                                          is wired (~3584-dim float32).
            - ``validator_artifact_uri`` — populated when sapiens-sidecar
                                          real inference writes pose.npz /
                                          flow.npz / overlay.mp4 (currently
                                          stub mode → no artifacts).
            - ``lora_applied_id``       — populated by manager-side hook on
                                          LoRA fusion (currently the
                                          dispatcher reads ``lora.id`` from
                                          the request body but doesn't
                                          persist it post-fusion).
            - ``lora_applied_strength`` — same.
        """
        current = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if current >= CURRENT_SCHEMA_VERSION:
            return
        try:
            if current < 2:
                for stmt in (
                    "ALTER TABLE generations ADD COLUMN params_json TEXT",
                    "ALTER TABLE generations ADD COLUMN gen_config_json TEXT",
                    "ALTER TABLE generations ADD COLUMN seed INTEGER",
                    "ALTER TABLE generations ADD COLUMN enhanced_prompt TEXT",
                ):
                    self._conn.execute(stmt)
            if current < 3:
                for stmt in (
                    "ALTER TABLE generations ADD COLUMN validator_score REAL",
                    "ALTER TABLE generations ADD COLUMN validator_payload_json TEXT",
                    "ALTER TABLE generations ADD COLUMN validator_version TEXT",
                    "ALTER TABLE generations ADD COLUMN validator_artifact_uri TEXT",
                    "ALTER TABLE generations ADD COLUMN parent_clip_id TEXT",
                    "ALTER TABLE generations ADD COLUMN shot_uuid TEXT",
                    "ALTER TABLE generations ADD COLUMN shot_config_key TEXT",
                    "ALTER TABLE generations ADD COLUMN composition_id TEXT",
                    "ALTER TABLE generations ADD COLUMN lora_applied_id TEXT",
                    "ALTER TABLE generations ADD COLUMN lora_applied_strength REAL",
                    "ALTER TABLE generations ADD COLUMN prompt_embedding BLOB",
                ):
                    self._conn.execute(stmt)
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_gen_shot_config_key "
                    "ON generations(shot_config_key)"
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_gen_parent_clip_id "
                    "ON generations(parent_clip_id)"
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_gen_composition_id "
                    "ON generations(composition_id)"
                )
                self._conn.executescript(_SCHEMA_V3_TABLES)
                self._maybe_seed_api_key_metadata()
            self._conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
            self._conn.commit()
            logger.info(
                "History DB migrated from user_version=%d to %d",
                current,
                CURRENT_SCHEMA_VERSION,
            )
        except Exception:
            self._conn.rollback()
            raise

    def _maybe_seed_api_key_metadata(self) -> None:
        """Seed ``api_key_metadata`` from ``.api_keys`` on first v3 migration.

        Single-tenant deploy default: every key in ``.api_keys`` gets
        ``training_opt_in=1``. Subsequent calls are no-ops because
        ``_migrate()`` only runs when ``user_version < CURRENT_SCHEMA_VERSION``.
        Already-populated tables (e.g. operator pre-seeded a key) are
        respected via ``INSERT OR IGNORE``.
        """
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM api_key_metadata"
        ).fetchone()[0]
        if existing:
            return
        keys_file = Path(__file__).parent / ".api_keys"
        if not keys_file.exists():
            return
        try:
            lines = keys_file.read_text().splitlines()
        except OSError:
            logger.warning("api_key_metadata seed: failed to read .api_keys", exc_info=True)
            return
        now = time.time()
        rows = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append((_hash_key(line), 1, "pro", None, now, now))
        if not rows:
            return
        self._conn.executemany(
            """INSERT OR IGNORE INTO api_key_metadata
               (api_key_hash, training_opt_in, tier, notes, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        logger.info("api_key_metadata seeded with %d row(s) from .api_keys", len(rows))

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
        *,
        seed: int | None = None,
        enhanced_prompt: str | None = None,
        raw_request: dict | None = None,
        gen_config_snapshot: dict | None = None,
        dispatch_params: dict | None = None,
        parent_clip_id: str | None = None,
        shot_uuid: str | None = None,
        shot_config_key: str | None = None,
        composition_id: str | None = None,
        lora_applied_id: str | None = None,
        lora_applied_strength: float | None = None,
    ) -> None:
        thumb_uri = None
        if result_bytes and result_uri:
            upload_id = result_uri.replace("storage://", "")
            thumb_id = _make_thumbnail(result_bytes, upload_id)
            if thumb_id:
                thumb_uri = f"thumb://{thumb_id}"

        # Music dispatches go through the sanitizer because the request body
        # was already lowered to on-disk paths before we got here. Video/image
        # jobs ship the untouched raw request (already storage:// URIs from
        # Pydantic model_dump). dispatch_params wins when both are provided.
        if dispatch_params is not None:
            params_payload: dict | None = _sanitize_params_for_history(job_type, dispatch_params)
        else:
            params_payload = raw_request

        params_json = json.dumps(params_payload) if params_payload is not None else None
        gen_config_json = (
            json.dumps(gen_config_snapshot) if gen_config_snapshot is not None else None
        )
        if params_json is not None:
            params_json = _truncate_json_blob(
                params_json, _HISTORY_PARAMS_MAX_BYTES, job_id, "params_json",
            )
        if gen_config_json is not None:
            gen_config_json = _truncate_json_blob(
                gen_config_json, _HISTORY_GEN_CONFIG_MAX_BYTES, job_id, "gen_config_json",
            )

        self._conn.execute(
            """INSERT OR REPLACE INTO generations
               (id, api_key_hash, job_type, prompt, model, width, height, turbo,
                status, result_uri, thumbnail_uri, created_at, completed_at, error,
                params_json, gen_config_json, seed, enhanced_prompt,
                parent_clip_id, shot_uuid, shot_config_key, composition_id,
                lora_applied_id, lora_applied_strength)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?)""",
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
                params_json,
                gen_config_json,
                seed,
                enhanced_prompt,
                parent_clip_id,
                shot_uuid,
                shot_config_key,
                composition_id,
                lora_applied_id,
                lora_applied_strength,
            ),
        )
        self._conn.commit()
        self._write_count += 1
        if self._write_count % _HISTORY_WAL_CHECKPOINT_EVERY == 0:
            try:
                self.checkpoint_wal()
            except Exception:
                logger.warning("history WAL checkpoint failed", exc_info=True)

    def checkpoint_wal(self, mode: str = "TRUNCATE") -> None:
        """Run ``PRAGMA wal_checkpoint(<mode>)``.

        Call sites: the write counter in :py:meth:`save` triggers this every
        ``_HISTORY_WAL_CHECKPOINT_EVERY`` rows; external cron / admin hooks
        may call it manually for a forced flush. Logs at INFO when the WAL
        file was over the warn threshold so operators can see when the
        checkpoint actually mattered.
        """
        try:
            wal_bytes = self._wal_path.stat().st_size
        except OSError:
            wal_bytes = 0
        if wal_bytes > _HISTORY_WAL_WARN_BYTES:
            logger.info(
                "history WAL size %.1f MiB before checkpoint (mode=%s)",
                wal_bytes / (1024 * 1024), mode,
            )
        self._conn.execute(f"PRAGMA wal_checkpoint({mode})")
        self._conn.commit()

    def list(
        self, api_key: str, limit: int = 50, offset: int = 0, job_type: str | None = None
    ) -> list[dict]:
        key_hash = _hash_key(api_key)
        # Support category shortcuts: "image" → all image types, "video" → all video types
        if job_type == "image":
            # NOTE: `LIKE '%-image%'` (the previous filter) silently dropped
            # every `image-edit` row because that type doesn't contain a
            # dash before "image" — only `text-to-image` and `image-to-image`
            # match the leading-dash pattern. With the joyai-edit char
            # rotation work that's thousands of missing rows per user.
            #
            # Correct intent: "all image-producing types, excluding video".
            # `image-to-video` also mentions "image", so we have to exclude
            # video explicitly rather than assume "image in the name = still".
            rows = self._conn.execute(
                """SELECT * FROM generations
                   WHERE api_key_hash = ?
                     AND job_type LIKE '%image%'
                     AND job_type NOT LIKE '%video%'
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
        elif job_type == "music":
            rows = self._conn.execute(
                """SELECT * FROM generations
                   WHERE api_key_hash = ? AND job_type LIKE '%music%'
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

    def delete(self, generation_id: str, api_key: str) -> bool:
        """Delete a single history entry and its on-disk files.

        Returns True if a row was deleted, False if the entry doesn't exist
        or belongs to a different API key. The caller is responsible for
        surfacing 404 in the latter case — we don't distinguish "not yours"
        from "not there" so keys can't probe each other's IDs.
        """
        key_hash = _hash_key(api_key)
        row = self._conn.execute(
            "SELECT result_uri, thumbnail_uri FROM generations WHERE id = ? AND api_key_hash = ?",
            (generation_id, key_hash),
        ).fetchone()
        if row is None:
            return False

        # Unlink the result file (completed generations only — failed rows
        # have no result_uri). Best-effort: missing files are fine, the row
        # still gets removed.
        if row["result_uri"]:
            uid = row["result_uri"].replace("storage://", "")
            p = config.UPLOAD_DIR / uid
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    logger.warning("Failed to unlink result file %s", p, exc_info=True)
        if row["thumbnail_uri"]:
            tid = row["thumbnail_uri"].replace("thumb://", "")
            p = config.THUMBNAIL_DIR / tid
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    logger.warning("Failed to unlink thumbnail file %s", p, exc_info=True)

        self._conn.execute(
            "DELETE FROM generations WHERE id = ? AND api_key_hash = ?",
            (generation_id, key_hash),
        )
        self._conn.commit()
        return True

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

    # ------------------------------------------------------------------
    # v3 (v1.17.0-rc1) helpers — composition lineage + retake provenance
    # ------------------------------------------------------------------

    def record_composition_clips(self, comp_id: str, clips: list[dict]) -> int:
        """Insert one ``composition_clips`` row per clip in a comp export.

        Each entry in ``clips`` may carry ``historyId`` (LTX-generated clips
        resolved through ``generations``) OR ``storage_uri`` (synthetic
        flash inserts that never had an LTX job, per v1.16.3). Both forms
        get a row; flash inserts store ``clip_history_id=NULL`` but still
        record their position in the comp.

        Best-effort: returns the number of rows actually inserted.
        ``INSERT OR IGNORE`` so re-export of the same composition doesn't
        explode on the (comp_id, clip_history_id, position) PK.
        """
        if not clips:
            return 0
        now = time.time()
        rows = []
        for idx, clip in enumerate(clips):
            if not isinstance(clip, dict):
                continue
            hist_id = clip.get("historyId") or clip.get("history_id")
            rows.append((comp_id, hist_id, idx, 1, now))
        if not rows:
            return 0
        before = self._conn.execute(
            "SELECT COUNT(*) FROM composition_clips WHERE comp_id = ?",
            (comp_id,),
        ).fetchone()[0]
        self._conn.executemany(
            """INSERT OR IGNORE INTO composition_clips
               (comp_id, clip_history_id, position, was_final, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        after = self._conn.execute(
            "SELECT COUNT(*) FROM composition_clips WHERE comp_id = ?",
            (comp_id,),
        ).fetchone()[0]
        return after - before

    def find_id_by_result_uri(self, result_uri: str) -> str | None:
        """Return the most recent history-row id whose ``result_uri`` matches.

        Used by ``/v2/retake`` to populate ``parent_clip_id`` on the new
        retake row. Bypasses api_key scoping intentionally — the caller
        already trusts the path it resolved from the bearer's URI.
        Returns ``None`` when no row matches (legacy clip, expired
        retention, or storage_uri pointing at an upload that was never
        a generation result).
        """
        if not result_uri:
            return None
        row = self._conn.execute(
            """SELECT id FROM generations WHERE result_uri = ?
               ORDER BY created_at DESC LIMIT 1""",
            (result_uri,),
        ).fetchone()
        return row["id"] if row else None
