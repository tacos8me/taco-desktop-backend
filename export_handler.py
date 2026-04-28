"""FFmpeg-based composition export — stitches clips with xfade transitions."""

from __future__ import annotations

import logging
import os
import shutil
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

# v1.16.2: encoder validation tables. Keys also gate the request-side allowlist
# in server.py so the two stay in sync.
SUPPORTED_ENCODERS = ("libx264", "libx265", "libopenh264")
SUPPORTED_PRESETS = (
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow", "placebo",
)
SUPPORTED_PROFILES = ("baseline", "main", "high", "high10", "high422", "high444")


def _resolve_ffmpeg_binary(preferred_encoder: str | None = None) -> tuple[str, list[str]]:
    """Pick the ffmpeg binary + the list of encoders it supports.

    v1.16.2 quality fix. Many conda-installed ffmpeg builds (e.g. the
    ``--disable-gpl`` recipe shipped with miniconda's anaconda channel) ship
    WITHOUT libx264/libx265 — they only have libopenh264, which can't honor
    ``-crf`` and produces visibly blocky output at default bitrate. The system
    ffmpeg at ``/usr/bin/ffmpeg`` typically has libx264 from the Ubuntu
    libx264-* packages.

    Strategy: probe the conda/PATH ffmpeg first, then ``/usr/bin/ffmpeg``,
    return the first one that supports the requested encoder. Falls back to
    ``ffmpeg`` on PATH (whatever it is) when nothing matches.

    Override via ``TACO_FFMPEG_BIN`` env var to pin a specific binary.
    """
    override = os.environ.get("TACO_FFMPEG_BIN", "").strip()
    candidates: list[str] = []
    if override:
        candidates.append(override)
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg and path_ffmpeg not in candidates:
        candidates.append(path_ffmpeg)
    if "/usr/bin/ffmpeg" not in candidates and Path("/usr/bin/ffmpeg").exists():
        candidates.append("/usr/bin/ffmpeg")

    def _probe(binary: str) -> list[str]:
        try:
            out = subprocess.run(
                [binary, "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        # `-encoders` lines look like:
        #   "  V....D libx264              libx264 H.264 / AVC ..."
        #   "  V....D libx264rgb           libx264 H.264 ... RGB ..."
        # Match by token boundary so libx264 doesn't also count libx264rgb.
        encs: set[str] = set()
        for line in out.stdout.splitlines():
            tokens = line.split()
            for enc in SUPPORTED_ENCODERS:
                if enc in tokens:
                    encs.add(enc)
        return sorted(encs)

    # First pass: prefer a binary that supports the requested encoder.
    if preferred_encoder:
        for c in candidates:
            encs = _probe(c)
            if preferred_encoder in encs:
                return c, encs
    # Second pass: pick any binary that supports libx264 (best default).
    for c in candidates:
        encs = _probe(c)
        if "libx264" in encs:
            return c, encs
    # Last resort: first probeable binary, whatever it has.
    for c in candidates:
        encs = _probe(c)
        if encs:
            return c, encs
    return (candidates[0] if candidates else "ffmpeg"), []


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
    quality: dict | None = None,
) -> bytes:
    """Stitch clips with xfade transitions, optional audio overlay, return MP4 bytes.

    audio_uri: optional `storage://<upload_id>` pointing at an audio file
    (wav/mp3/etc.) to overlay as the output audio track. When set, the audio is
    fed as an extra ffmpeg input, mapped to the output, and truncated to the
    video length via ``-shortest``.

    quality: optional dict of v1.16.2 encoder knobs:

      ``output_encoder``       — ``"libx264"`` (default) | ``"libx265"`` | ``"libopenh264"``
      ``output_crf``           — int 0..51 (default 18 for x264, 22 for x265)
      ``output_preset``        — libx264-style preset name (default ``"medium"``)
      ``output_profile``       — H.264 profile (default ``"high"``)
      ``output_video_bitrate`` — e.g. ``"12M"``; switches CRF encoders to 1-pass ABR
      ``output_audio_bitrate`` — e.g. ``"256k"`` (default ``"256k"``, was 192k pre-v1.16.2)

    All knobs are validated at the request layer. ``None`` (or absent keys)
    means "use defaults".

    v1.9.3 per-clip segmentation (MusicVideo mode): when every clip carries a
    numeric ``audioStart`` field (seconds into the source song where that
    clip's audio window begins), the song is sliced per-clip via ``atrim`` and
    concatenated 1:1 with the video concat — so audio stays beat-aligned even
    with LTX's 8k+1 frame quantization drift. Fallback to the legacy full-song
    overlay when any clip lacks ``audioStart`` (timeline-mode compositions
    pre-dating the field) or xfade transitions are in play.
    """
    quality_params = quality or {}

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

    for clip_idx, clip in enumerate(clips):
        # v1.16.3: clips may carry EITHER historyId (LTX-generated) OR
        # storage_uri (synthetic flash inserts minted by the MCP orchestrator
        # — they don't ride history.db at all). Honor both per the long-
        # standing MCP contract; pre-v1.16.3 only handled historyId and
        # KeyError'd on flash-only compositions.
        hist_id = clip.get("historyId")
        storage_uri = clip.get("storage_uri")
        if hist_id:
            path = _resolve_clip_path(hist_id, uploads)
        elif storage_uri:
            path = uploads.resolve(storage_uri)
            if not path.exists():
                raise FileNotFoundError(
                    f"Clip {clip_idx} storage_uri not found on disk: {storage_uri}"
                )
        else:
            raise ValueError(
                f"Clip {clip_idx} missing both historyId and storage_uri"
            )
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

    # Single clip + no audio + no per-clip transforms — return raw bytes,
    # skip ffmpeg entirely. With audio we still need ffmpeg to mux the track.
    # v1.15.1 fix: also fall through to ffmpeg whenever any per-clip transform
    # is requested (speed != 1.0, tailTrimFrames > 0). Previously the shortcut
    # returned raw bytes regardless, silently dropping the transform — a
    # speed=1000 export produced an MP4 byte-identical to speed=1.0.
    if (
        len(clip_paths) == 1
        and audio_path is None
        and clip_speed[0] == 1.0
        and clip_tail_trim[0] == 0
    ):
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
        audio_bitrate = quality_params.get("output_audio_bitrate") or "256k"
        audio_codec = ["-strict", "-2", "-c:a", "aac", "-b:a", audio_bitrate]
    elif audio_path is not None:
        # Legacy full-song overlay — unchanged behavior for pre-audioStart
        # comps (timeline mode) or xfade compositions.
        audio_map = ["-map", f"{len(clip_paths)}:a:0"]
        # -strict -2 enables native AAC on ffmpeg builds where it's flagged
        # experimental (avcodec_open2(aac) EINVAL). No-op on builds where
        # AAC is already stable (e.g. this box's ffmpeg 6.1.1).
        # v1.16.2: bitrate default 192k → 256k; overridable via quality dict.
        audio_bitrate = quality_params.get("output_audio_bitrate") or "256k"
        audio_codec = ["-strict", "-2", "-c:a", "aac", "-b:a", audio_bitrate, "-shortest"]
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

    # v1.16.2: industry-standard quality defaults. Pre-v1.16.2 was hardcoded
    # `-c:v libopenh264` with no CRF/bitrate flags = ~4-8 Mbps default and
    # visible blocking on 1080p+. Switch to libx264 + CRF 18 + profile=high +
    # preset=medium for visually-transparent quality. All knobs overridable
    # per-call via quality dict (see docstring).
    encoder = (quality_params.get("output_encoder") or "").strip() or None
    crf = quality_params.get("output_crf")
    preset = quality_params.get("output_preset") or "medium"
    profile = quality_params.get("output_profile") or "high"
    video_bitrate = quality_params.get("output_video_bitrate")

    ffmpeg_bin, available_encoders = _resolve_ffmpeg_binary(encoder)

    if encoder is None:
        # Auto-pick: prefer libx264 for the CRF default; fall back to whatever
        # the binary has if libx264 isn't compiled in.
        if "libx264" in available_encoders:
            encoder = "libx264"
        elif "libx265" in available_encoders:
            encoder = "libx265"
        else:
            encoder = "libopenh264"
            logger.warning(
                "export_composition: ffmpeg %s lacks libx264/libx265; "
                "falling back to libopenh264 (lower quality at default bitrate). "
                "Install x264 (apt: libx264-*) or set TACO_FFMPEG_BIN to a build "
                "that includes it.",
                ffmpeg_bin,
            )
    elif available_encoders and encoder not in available_encoders:
        raise RuntimeError(
            f"requested encoder {encoder!r} not available in {ffmpeg_bin}; "
            f"available: {available_encoders}"
        )

    video_codec_args: list[str] = ["-c:v", encoder]
    if encoder == "libx264":
        video_codec_args.extend(["-crf", str(crf if crf is not None else 18)])
        video_codec_args.extend(["-preset", preset])
        video_codec_args.extend(["-profile:v", profile])
        # yuv420p maximizes player compatibility (e.g. iOS QuickTime, Chrome
        # mobile); avoids 4:4:4 / yuv444p output that some clients won't decode.
        video_codec_args.extend(["-pix_fmt", "yuv420p"])
    elif encoder == "libx265":
        # x265 CRF runs ~4 colder than x264 for similar perceptual quality.
        video_codec_args.extend(["-crf", str(crf if crf is not None else 22)])
        video_codec_args.extend(["-preset", preset])
        video_codec_args.extend(["-pix_fmt", "yuv420p"])
    elif encoder == "libopenh264":
        # Legacy/fallback path — libopenh264 doesn't honor `-crf`. Use bitrate.
        video_codec_args.extend(["-b:v", video_bitrate or "12M"])
    else:
        raise ValueError(f"unsupported encoder: {encoder}")

    if video_bitrate and encoder in ("libx264", "libx265"):
        # Caller asked for a specific bitrate target on a CRF encoder — switch
        # to 1-pass ABR. Strip the CRF arg pair we just added; -b:v + -maxrate
        # + -bufsize give a constrained ABR mode that's better than CRF when
        # the operator has a hard size budget.
        if "-crf" in video_codec_args:
            crf_idx = video_codec_args.index("-crf")
            del video_codec_args[crf_idx:crf_idx + 2]
        video_codec_args.extend([
            "-b:v", video_bitrate,
            "-maxrate", video_bitrate,
            "-bufsize", "24M",
        ])

    cmd = [
        ffmpeg_bin, "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        *audio_map,
        *video_codec_args,
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
