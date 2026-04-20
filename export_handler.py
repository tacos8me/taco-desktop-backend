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
    clip_fps: list[float] = []
    clip_tail_trim: list[int] = []

    for clip in clips:
        path = _resolve_clip_path(clip["historyId"], uploads)
        clip_paths.append(path)
        duration = float(clip.get("duration", 6.0))
        clip_durations.append(duration)
        fps = float(clip.get("fps", 24.0))
        clip_fps.append(fps)
        clip_tail_trim.append(int(clip.get("tailTrimFrames", 0) or 0))

    if not clip_paths:
        raise ValueError("No clips to export")

    # v1.10.0: effective_durations cascade = raw duration − trimmed tail.
    # Silent guards per spec:
    #   - Last clip always tailTrimFrames=0 (no follower, no continuity seam).
    #   - Single-clip exports zero out unconditionally.
    #   - Over-trim (tail >= declared frames) clamps to declared-1 with WARN.
    # xfade path skips trim entirely — handled at filter-build time below.
    effective_durations: list[float] = []
    for i in range(len(clip_paths)):
        tail = clip_tail_trim[i]
        fps = clip_fps[i]
        declared_frames = max(1, int(round(clip_durations[i] * fps)))
        if len(clip_paths) == 1 or i == len(clip_paths) - 1:
            tail = 0
            clip_tail_trim[i] = 0
        elif tail >= declared_frames:
            logger.warning(
                "export_composition: clip %d tailTrimFrames=%d >= declared_frames=%d; clamping to %d",
                i, tail, declared_frames, declared_frames - 1,
            )
            tail = declared_frames - 1
            clip_tail_trim[i] = tail
        effective_durations.append(clip_durations[i] - tail / fps)

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

    # v1.9.8: per-input video normalization before concat/xfade. The concat
    # filter assumes all inputs have identical parameters AND monotonic PTS
    # across the boundary — with no setpts reset, any clip that doesn't start
    # at PTS=0 (common for re-encoded intermediates) produces a visible seam
    # glitch (single-frame freeze or jump) at every join.
    #
    #   setpts=PTS-STARTPTS — reset each input's timeline to 0 so concat sees
    #                         a monotonic stream across the join.
    #   format=yuv420p       — normalize pixel format for libopenh264 + MP4,
    #                         preventing format-change stalls at the seam.
    #
    # LTX clips come out at matching resolutions for a given composition so
    # no scale/setsar is needed here. Frame rate is left alone so the first
    # clip's fps wins (concat filter's default behavior) — don't force a
    # target fps because it'd stutter if the user mixes sources.
    # v1.10.0: per-input `trim=end_frame=<kept>` BEFORE `setpts=PTS-STARTPTS`.
    # Order matters — trim preserves original PTS (it selects a frame window
    # on the input timeline), and setpts then rebases to zero. Reversing the
    # order would trim after rebasing, which drops the WRONG frames.
    # xfade path skips trim (xfade already overlaps — can't stack the two).
    def _norm(i: int, out_label: str) -> str:
        tail = clip_tail_trim[i]
        if tail > 0 and not has_xfade:
            fps = clip_fps[i]
            declared_frames = max(1, int(round(clip_durations[i] * fps)))
            kept = max(1, declared_frames - tail)
            return (
                f"[{i}:v]trim=end_frame={kept},setpts=PTS-STARTPTS,"
                f"format=yuv420p{out_label}"
            )
        return f"[{i}:v]setpts=PTS-STARTPTS,format=yuv420p{out_label}"

    if not has_xfade:
        if len(clip_paths) == 1:
            # Single clip + audio overlay — still normalize format so the
            # output's stream params are stable regardless of source codec.
            filter_complex = _norm(0, "[vout]")
        else:
            # Simple concat — no xfade needed. Normalize each input, then
            # concat the normalized streams.
            norm_parts = [_norm(i, f"[v{i:02d}n]") for i in range(len(clip_paths))]
            concat_inputs = "".join(f"[v{i:02d}n]" for i in range(len(clip_paths)))
            filter_complex = ";".join([
                *norm_parts,
                f"{concat_inputs}concat=n={len(clip_paths)}:v=1:a=0[vout]",
            ])
    else:
        # xfade chain — also normalize each input before xfade chains them.
        # xfade requires identical parameters on both sides of the transition
        # too, so the pre-normalization is equally important here.
        norm_parts: list[str] = [_norm(i, f"[v{i:02d}n]") for i in range(len(clip_paths))]
        filter_parts: list[str] = []
        cumulative_offset = 0.0
        prev_label = "[v00n]"

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
                f"{prev_label}[v{i:02d}n]xfade=transition={ffmpeg_transition}"
                f":duration={trans_duration}:offset={cumulative_offset}{out_label}"
            )
            prev_label = out_label

        filter_complex = ";".join([*norm_parts, *filter_parts])

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

        # v1.9.6: slice duration must be the BEAT GAP (audioStart[i+1] -
        # audioStart[i]), NOT the LTX-quantized clip_duration. Clip durations
        # round up to 8k+1 frames (e.g. 2.04 s instead of 2.00 s), so using
        # clip_duration as the atrim span overlaps adjacent slices by ~40 ms
        # at every seam — audible repeat of the last fraction of each beat.
        # The beat-gap approach pulls non-overlapping song ranges. For the
        # LAST clip there is no next beat, so fall back to clip_duration[N-1].
        # Zero / non-monotonic gaps fall back to clip_duration[i] — defensive
        # against clients that pass garbage audioStart values.
        # v1.10.0: non-last slice clamps to effective_durations[i] (defensive —
        # prevents atrim past the trimmed EOF when a clip's tail was cut).
        # Last clip uses effective_durations[-1] (== clip_durations[-1] because
        # last clip's tailTrimFrames is always zeroed above).
        # v1.11.2: FE can pass explicit `audioDurationSec` per clip to decouple
        # audio-side slice from video-side effective_duration — lets tail=6
        # chain conditioning give a 0 ms visual seam AND full-song audio
        # continuity (at the cost of a progressive video-cut-before-beat drift
        # equal to `audioDurationSec - effective_duration` per seam). When the
        # field is absent the v1.11.1 clamp behavior is preserved exactly.
        slice_durations: list[float] = []
        for i, start in enumerate(clip_audio_starts):
            explicit = clips[i].get("audioDurationSec")
            if (
                isinstance(explicit, (int, float))
                and not isinstance(explicit, bool)
                and explicit > 0
            ):
                slice_durations.append(float(explicit))
                continue
            if i < len(clip_paths) - 1:
                gap = clip_audio_starts[i + 1] - start
                if gap <= 0:
                    gap = clip_durations[i]
                gap = min(float(gap), effective_durations[i])
                slice_durations.append(float(gap))
            else:
                slice_durations.append(float(effective_durations[i]))

        audio_parts: list[str] = []
        audio_labels: list[str] = []
        for i, (start, slice_dur) in enumerate(zip(clip_audio_starts, slice_durations)):
            label = f"[a{i:02d}]"
            # asetpts=N/SR/TB resets timestamps so concat doesn't choke on
            # non-monotonic PTS across the sliced segments.
            audio_parts.append(
                f"[{audio_idx}:a]atrim=start={start}:duration={slice_dur},asetpts=N/SR/TB{label}"
            )
            audio_labels.append(label)
        audio_concat = (
            f"{''.join(audio_labels)}concat=n={len(clip_paths)}:v=0:a=1[aout]"
        )
        filter_complex = ";".join([filter_complex, *audio_parts, audio_concat])
        audio_map = ["-map", "[aout]"]
        # No -shortest needed: each atrim slice is bounded by the next beat's
        # audioStart (or by clip_duration for the last clip), so the audio
        # total length is (sum of beat gaps) + last clip_duration — within
        # one LTX-quantization step of the video length.
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

    # v1.9.9: force IDR frames at seam positions so each clip starts from a
    # clean intra frame rather than being P-predicted from the previous clip's
    # last frame. Without this, libopenh264 produces a single long GOP (1 IDR
    # for the whole output) and the first frame of each downstream clip is a
    # motion-vector interpolation from unrelated source content — visible as
    # a "smeared" / "bled" seam glitch at every cut.
    #
    # v1.9.8's `setpts=PTS-STARTPTS` made this worse by also eliminating the
    # implicit PTS-gap scene-change signal the encoder was getting. Keep the
    # setpts reset (correct for timing) and add `-force_key_frames` to make
    # scene boundaries explicit.
    #
    # Seam times are the cumulative sum of clip_durations, excluding the
    # final total (which IS the end of stream — no seam there).
    # v1.10.0: seam cumsum uses effective_durations (== clip_durations when
    # no tails are trimmed — byte-identical to v1.9.9 in that case).
    force_keyframe_args: list[str] = []
    if len(clip_paths) > 1:
        seam_times: list[float] = []
        acc = 0.0
        for dur in effective_durations[:-1]:
            acc += float(dur)
            seam_times.append(acc)
        if seam_times:
            # Format with enough precision to avoid rounding onto the wrong
            # side of an output frame boundary.
            force_keyframe_args = [
                "-force_key_frames",
                ",".join(f"{t:.6f}" for t in seam_times),
            ]

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        *audio_map,
        "-c:v", "libopenh264",
        *force_keyframe_args,
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
