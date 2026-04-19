"""FFmpeg-based composition export — stitches clips with xfade transitions."""

from __future__ import annotations

import logging
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import config
from upload_store import UploadStore

logger = logging.getLogger(__name__)

# 🔌 EXTENSION POINT: map TransitionType → ffmpeg xfade transition name
TRANSITION_MAP: dict[str, str] = {
    "crossfade": "fade",
    # Future: "wipe": "wipeleft", "dissolve": "dissolve", "glitch": "...", etc.
}


def _resolve_clip_path(history_id: str, uploads: UploadStore) -> Path:
    """Look up a generation's result file by history ID."""
    conn = sqlite3.connect(str(config.HISTORY_DB))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT result_uri FROM generations WHERE id = ?", (history_id,)
    ).fetchone()
    conn.close()
    if not row or not row["result_uri"]:
        raise FileNotFoundError(f"Clip {history_id} not found in history")
    path = uploads.resolve(row["result_uri"])
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def export_composition(
    clips: list[dict],
    transitions: list[dict],
    uploads: UploadStore,
    audio_uri: str | None = None,
) -> bytes:
    """Stitch clips with xfade transitions, optional audio overlay, return MP4 bytes.

    audio_uri: optional `storage://<upload_id>` pointing at an audio file
    (wav/mp3/etc.) to overlay as the output audio track. When set, the audio is
    fed as an extra ffmpeg input, mapped to the output, and truncated to the
    video length via ``-shortest``.

    v1.9.3 per-clip segmentation (MusicVideo mode): when every clip carries a
    numeric ``audioStart`` field (seconds into the source song where that
    clip's audio window begins), the song is sliced per-clip via ``atrim`` and
    concatenated 1:1 with the video concat — so audio stays beat-aligned even
    with LTX's 8k+1 frame quantization drift. Fallback to the legacy full-song
    overlay when any clip lacks ``audioStart`` (timeline-mode compositions
    pre-dating the field) or xfade transitions are in play.
    """
    clip_paths: list[Path] = []
    clip_durations: list[float] = []

    for clip in clips:
        path = _resolve_clip_path(clip["historyId"], uploads)
        clip_paths.append(path)
        clip_durations.append(clip.get("duration", 6.0))

    if not clip_paths:
        raise ValueError("No clips to export")

    audio_path: Path | None = None
    if audio_uri:
        # uploads.resolve() validates the URI and raises FileNotFoundError
        # if the file is missing — no redundant exists() check needed.
        audio_path = uploads.resolve(audio_uri)

    # Single clip + no audio — return raw bytes, skip ffmpeg entirely.
    # With audio we still need ffmpeg to mux the track.
    if len(clip_paths) == 1 and audio_path is None:
        return clip_paths[0].read_bytes()

    # Build ffmpeg command with xfade chain (or concat + audio overlay)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = Path(tmp.name)

    inputs: list[str] = []
    for p in clip_paths:
        inputs.extend(["-i", str(p)])
    if audio_path is not None:
        inputs.extend(["-i", str(audio_path)])

    # Build xfade filter chain
    # Check if any transitions have actual duration
    has_xfade = any(
        (t.get("durationSec", t.get("duration_sec", 0)) or 0) > 0
        for t in transitions
    )

    if not has_xfade:
        if len(clip_paths) == 1:
            # Single clip + audio overlay — pass video through unchanged.
            filter_complex = "[0:v]null[vout]"
        else:
            # Simple concat — no xfade needed
            filter_parts = []
            for i in range(len(clip_paths)):
                filter_parts.append(f"[{i}:v]")
            filter_complex = f"{''.join(filter_parts)}concat=n={len(clip_paths)}:v=1:a=0[vout]"
    else:
        # xfade chain
        filter_parts: list[str] = []
        cumulative_offset = 0.0
        prev_label = "[0:v]"

        for i in range(1, len(clip_paths)):
            trans = next(
                (t for t in transitions if t.get("clipBIndex", t.get("clip_b_index")) == i),
                None,
            )
            trans_duration = trans.get("durationSec", trans.get("duration_sec", 0)) if trans else 0
            if trans_duration <= 0:
                trans_duration = 0.5  # minimum crossfade for xfade mode
            trans_type_key = trans.get("type", "crossfade") if trans else "crossfade"
            ffmpeg_transition = TRANSITION_MAP.get(trans_type_key, "fade")

            cumulative_offset += clip_durations[i - 1] - trans_duration
            out_label = f"[v{i:02d}]" if i < len(clip_paths) - 1 else "[vout]"

            filter_parts.append(
                f"{prev_label}[{i}:v]xfade=transition={ffmpeg_transition}"
                f":duration={trans_duration}:offset={cumulative_offset}{out_label}"
            )
            prev_label = out_label

        filter_complex = ";".join(filter_parts)

    # v1.9.3: per-clip audio segmentation trigger. When every clip carries a
    # numeric audioStart (bools rejected — isinstance(True, int) is True in
    # Python), slice the song via atrim per clip and concat-align with video.
    # Falls back to the legacy full-song overlay otherwise.
    #
    # xfade mode intentionally stays on the legacy path — crossfading video
    # while slicing audio needs its own alignment design; deferred. Frontend
    # doesn't allow xfade in MusicVideo mode anyway.
    clip_audio_starts = [c.get("audioStart") for c in clips]
    beat_synced = (
        audio_path is not None
        and not has_xfade
        and bool(clip_paths)
        and all(
            isinstance(s, (int, float)) and not isinstance(s, bool)
            for s in clip_audio_starts
        )
    )

    if beat_synced:
        audio_idx = len(clip_paths)  # song is the last -i input
        audio_parts: list[str] = []
        audio_labels: list[str] = []
        for i, (start, dur) in enumerate(zip(clip_audio_starts, clip_durations)):
            label = f"[a{i:02d}]"
            # asetpts=N/SR/TB resets timestamps so concat doesn't choke on
            # non-monotonic PTS across the sliced segments.
            audio_parts.append(
                f"[{audio_idx}:a]atrim=start={start}:duration={dur},asetpts=N/SR/TB{label}"
            )
            audio_labels.append(label)
        audio_concat = (
            f"{''.join(audio_labels)}concat=n={len(clip_paths)}:v=0:a=1[aout]"
        )
        filter_complex = ";".join([filter_complex, *audio_parts, audio_concat])
        audio_map = ["-map", "[aout]"]
        # No -shortest needed: audio is exactly the same length as video by
        # construction (atrim duration per clip == video clip duration).
        audio_codec = ["-strict", "-2", "-c:a", "aac", "-b:a", "192k"]
    elif audio_path is not None:
        # Legacy full-song overlay — unchanged behavior for pre-audioStart
        # comps (timeline mode) or xfade compositions.
        audio_map = ["-map", f"{len(clip_paths)}:a:0"]
        # -strict -2 enables native AAC on ffmpeg builds where it's flagged
        # experimental (avcodec_open2(aac) EINVAL). No-op on builds where
        # AAC is already stable (e.g. this box's ffmpeg 6.1.1).
        audio_codec = ["-strict", "-2", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        audio_map = []
        audio_codec = []

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        *audio_map,
        "-c:v", "libopenh264",
        *audio_codec,
        str(output_path),
    ]

    logger.info("FFmpeg export: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        # Get the actual error from the end of stderr (skip version banner)
        err_lines = [l for l in result.stderr.splitlines() if l.strip() and not l.startswith("  ")]
        err_msg = "\n".join(err_lines[-5:]) if err_lines else result.stderr[-300:]
        logger.error("FFmpeg failed:\n%s", err_msg)
        logger.error("FFmpeg command: %s", " ".join(cmd))
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg failed: {err_msg[:300]}")

    video_bytes = output_path.read_bytes()
    output_path.unlink(missing_ok=True)
    return video_bytes
