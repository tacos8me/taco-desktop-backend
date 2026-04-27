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
    # v1.15.0 B2: per-clip playback speed (default 1.0 = unchanged). Speed is
    # video-only: slowing the picture does NOT slow the song. Audio decoupling
    # in the beat_synced branch below uses pre-speed durations so the song
    # stays the song. See docs/MV_EDITING.md §3.13 / §5.7. Speed ≤ 0 is
    # invalid; we WARN + default to 1.0 instead of raising (forgiving FE
    # contract — drift is visible, hard 422 mid-export is not).
    clip_speed: list[float] = []

    for clip in clips:
        path = _resolve_clip_path(clip["historyId"], uploads)
        clip_paths.append(path)
        duration = float(clip.get("duration", 6.0))
        clip_durations.append(duration)
        fps = float(clip.get("fps", 24.0))
        clip_fps.append(fps)
        clip_tail_trim.append(int(clip.get("tailTrimFrames", 0) or 0))
        speed_raw = clip.get("speed", 1.0)
        try:
            speed = float(speed_raw) if speed_raw is not None else 1.0
        except (TypeError, ValueError):
            speed = 1.0
        if not (speed > 0):
            logger.warning(
                "export_composition: clip speed=%r invalid (must be > 0); defaulting to 1.0",
                speed_raw,
            )
            speed = 1.0
        clip_speed.append(speed)

    if not clip_paths:
        raise ValueError("No clips to export")

    # v1.10.0: effective_durations cascade = raw duration − trimmed tail.
    # Silent guards per spec:
    #   - Last clip always tailTrimFrames=0 (no follower, no continuity seam).
    #   - Single-clip exports zero out unconditionally.
    #   - Over-trim (tail >= declared frames) clamps to declared-1 with WARN.
    # xfade path skips trim entirely — handled at filter-build time below.
    # v1.15.0 B2: divide by speed so post-speed playback length is what every
    # downstream consumer (xfade offset, force-keyframe seams, video-side
    # beat_synced clamp) sees. Pre-speed length lives on at
    # ``pre_speed_durations`` for audio-side decoupling below.
    effective_durations: list[float] = []
    pre_speed_durations: list[float] = []  # = (clip_duration - tail/fps), no /speed
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
        pre_speed = clip_durations[i] - tail / fps
        pre_speed_durations.append(pre_speed)
        effective_durations.append(pre_speed / clip_speed[i])

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
    # v1.15.0 B2: per-clip speed support. setpts=(PTS-STARTPTS)/speed scales
    # the playback rate (speed=2.0 → half-duration, speed=0.5 → double). The
    # divisor must be applied AFTER trim+rebase so the rebased PTS=0 origin
    # is preserved. When speed==1.0 the emitted filter is byte-identical to
    # v1.10.0 (additive invariant — see CLAUDE.md and PR body).
    def _norm(i: int, out_label: str) -> str:
        tail = clip_tail_trim[i]
        speed = clip_speed[i]
        speed_clause = (
            f"setpts=(PTS-STARTPTS)/{speed:.6f}"
            if speed != 1.0
            else "setpts=PTS-STARTPTS"
        )
        if tail > 0 and not has_xfade:
            fps = clip_fps[i]
            declared_frames = max(1, int(round(clip_durations[i] * fps)))
            kept = max(1, declared_frames - tail)
            return (
                f"[{i}:v]trim=end_frame={kept},{speed_clause},"
                f"format=yuv420p{out_label}"
            )
        return f"[{i}:v]{speed_clause},format=yuv420p{out_label}"

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

            # v1.15.0 B2: xfade offset must track POST-speed playback length so
            # a slow-mo clip's xfade seam lands at the correct rendered second.
            # effective_durations[i-1] == clip_durations[i-1] when speed==1.0.
            cumulative_offset += effective_durations[i - 1] - trans_duration
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
        # audioStart[i]), NOT the LTX-quantized clip_duration.
        # v1.10.0: non-last slice clamps to effective_durations[i] (defensive —
        # prevents atrim past the trimmed EOF when a clip's tail was cut).
        # v1.11.2: explicit `audioDurationSec` overrides the clamp.
        #
        # v1.15.0 B1 — TWO-PASS refactor for transition.audioLeadFrames (J/L
        # cuts). The single-pass loop CANNOT compute slice[i] correctly because
        # clip i's slice end depends on `audio_lead[i+1]`. The math:
        #
        #   audio_lead_secs[i] = transition[i].audioLeadFrames / clip_fps[i]
        #     (transition i = boundary INTO clip i; clip 0 has no boundary in
        #      → lead 0)
        #   audio_start_adj[i] = max(0, clip_audio_starts[i] - audio_lead_secs[i])
        #   slice_durations[i] = original_gap + audio_lead_secs[i] - audio_lead_secs[i+1]
        #
        # Negative audioLeadFrames is L-cut (audio of A continues past picture
        # cut); it's the same math with a negative sign — no special-casing.
        #
        # v1.15.0 B2 audio decoupling — clip.speed is video-only. The slice
        # clamp must use PRE-speed durations (`pre_speed_durations[i]` =
        # `clip_durations[i] - tail/fps`) so the song doesn't get clipped
        # short when speed > 1.0. The video side already shrank via
        # `effective_durations`; the audio side stays at song-time.
        #
        # Pass 1: gather per-boundary audio_lead_secs.
        audio_lead_secs: list[float] = []
        for i in range(len(clip_paths)):
            if i == 0:
                audio_lead_secs.append(0.0)
                continue
            trans = next(
                (t for t in transitions if t.get("clipBIndex", t.get("clip_b_index")) == i),
                None,
            )
            lead_frames_raw = trans.get("audioLeadFrames", 0) if trans else 0
            try:
                lead_frames = int(lead_frames_raw or 0)
            except (TypeError, ValueError):
                lead_frames = 0
            audio_lead_secs.append(lead_frames / clip_fps[i] if lead_frames else 0.0)
        # Sentinel for the i+1 lookup at the last clip: no follower → 0.0.
        audio_lead_secs_with_sentinel = audio_lead_secs + [0.0]

        # Pass 2: build adjusted starts + slice_durations cross-referencing
        # leads on both sides of each slice.
        audio_start_adj: list[float] = []
        slice_durations: list[float] = []
        min_slice = 1.0 / max(clip_fps)  # sub-frame slice clamp (1/fps min)
        for i, start in enumerate(clip_audio_starts):
            adj = max(0.0, float(start) - audio_lead_secs[i])
            audio_start_adj.append(adj)

            explicit = clips[i].get("audioDurationSec")
            if (
                isinstance(explicit, (int, float))
                and not isinstance(explicit, bool)
                and explicit > 0
            ):
                # Caller-takes-the-wheel — no lead math, no clamp, no decouple.
                slice_durations.append(float(explicit))
                continue
            if i < len(clip_paths) - 1:
                original_gap = clip_audio_starts[i + 1] - float(start)
                if original_gap <= 0:
                    original_gap = clip_durations[i]
                # B1 cross-boundary correction:
                #   slice = original_gap + lead[i] - lead[i+1]
                # When lead[i] > 0 (J-cut into i): slice grows so clip i's audio
                # window fully covers the video-i picture window plus the lead.
                # When lead[i+1] > 0 (J-cut into i+1): slice shrinks because
                # clip (i+1) starts its audio earlier.
                slice_dur = (
                    original_gap
                    + audio_lead_secs[i]
                    - audio_lead_secs_with_sentinel[i + 1]
                )
                # B2 audio decoupling: clamp against PRE-speed video duration,
                # not effective_durations[i] (which has been /=speed).
                slice_dur = min(float(slice_dur), pre_speed_durations[i])
                slice_dur = max(min_slice, slice_dur)
                slice_durations.append(float(slice_dur))
            else:
                # Last clip: no follower lead. Pre-speed duration + own lead.
                slice_dur = pre_speed_durations[i] + audio_lead_secs[i]
                slice_durations.append(max(min_slice, float(slice_dur)))

        audio_parts: list[str] = []
        audio_labels: list[str] = []
        for i, slice_dur in enumerate(slice_durations):
            label = f"[a{i:02d}]"
            # B1: emit atrim with the LEAD-ADJUSTED start, not raw audioStart.
            # asetpts=N/SR/TB resets timestamps so concat doesn't choke on
            # non-monotonic PTS across the sliced segments.
            audio_parts.append(
                f"[{audio_idx}:a]atrim=start={audio_start_adj[i]}:duration={slice_dur},asetpts=N/SR/TB{label}"
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
