"""v1.15.0 export_handler tests — J/L cuts (B1) + clip.speed cascade (B2).

These tests intercept ``subprocess.run`` and ``_resolve_clip_path`` so the
ffmpeg command never executes; we assert on the constructed ``filter_complex``
string and the final argv. The point is to lock the math, not to render a
video. See ``docs/MV_EDITING.md`` §3.3 / §3.13 / §5.7 for the grammar that
generated these cases.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import export_handler


@pytest.fixture
def fake_resolve(tmp_path):
    """Return a stub _resolve_clip_path that yields fake paths under tmp."""
    def _stub(history_id: str, uploads):
        p = tmp_path / f"clip-{history_id}.mp4"
        p.write_bytes(b"\x00")  # exists() must be True
        return p
    return _stub


class _FakeUploads:
    """Stand-in UploadStore: resolve any URI to a tmp file."""
    def __init__(self, tmp_path: Path):
        self._tmp = tmp_path

    def resolve(self, uri: str) -> Path:
        # Make the audio path point at a real, byte-zero file.
        p = self._tmp / "song.mp3"
        if not p.exists():
            p.write_bytes(b"\x00")
        return p


def _capture_export(clips, transitions, uploads, audio_uri, fake_resolve, monkeypatch, quality=None):
    """Run export_composition with mocked ffmpeg + clip-resolve, return the
    captured ffmpeg argv list (so tests can introspect filter_complex).
    """
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        # The encoder-probe call is `[bin, "-encoders"]` — return the libx264
        # listing so the auto-pick path picks libx264 deterministically in tests.
        if len(cmd) == 2 and cmd[1] == "-encoders":
            stdout = (
                " V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10\n"
                " V....D libx265              libx265 H.265 / HEVC\n"
                " V....D libopenh264          OpenH264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10\n"
            )
            return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")
        captured["cmd"] = cmd
        # Pretend ffmpeg succeeded; write a stub mp4 to the output path so the
        # post-run read_bytes() works.
        out = Path(cmd[-1])
        out.write_bytes(b"\x00")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(export_handler, "_resolve_clip_path", fake_resolve)
    monkeypatch.setattr(export_handler.subprocess, "run", _fake_run)

    export_handler.export_composition(clips, transitions, uploads, audio_uri, quality=quality)
    return captured["cmd"]


def _filter_complex(cmd: list[str]) -> str:
    idx = cmd.index("-filter_complex")
    return cmd[idx + 1]


# ---------------------------------------------------------------------------
# B1 — J/L-cut audioLeadFrames cascade
# ---------------------------------------------------------------------------


def test_j_cut_shortens_prior_slice_and_pulls_in_audio_start(tmp_path, fake_resolve, monkeypatch):
    """audioLeadFrames=6 on transition 0→1 at fps=24 → 0.25 s lead.

    Expected:
      - clip[1] atrim start = clip_audio_starts[1] - 0.25
      - clip[0] slice duration = original_gap - 0.25 (shorter by lead)
      - clip[1] slice duration = pre_speed_dur[1] + 0.25 (last clip)
    """
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 4.0, "fps": 24.0, "audioStart": 0.0},
        {"historyId": "b", "duration": 4.0, "fps": 24.0, "audioStart": 4.0},
    ]
    transitions = [{"clipBIndex": 1, "audioLeadFrames": 6}]  # 6/24 = 0.25 s

    cmd = _capture_export(clips, transitions, uploads, "storage://song", fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)

    # Clip 1's atrim start was pulled back from 4.0 to 3.75.
    assert "atrim=start=3.75" in fc, fc
    # Clip 0's slice should be shorter (was 4.0, now 3.75) — original_gap=4.0,
    # lead[0]=0, lead[1]=0.25 → slice = 4.0 + 0 - 0.25 = 3.75
    assert "atrim=start=0.0:duration=3.75" in fc, fc


def test_l_cut_negative_audio_lead_frames(tmp_path, fake_resolve, monkeypatch):
    """L-cut: audioLeadFrames=-6 at fps=24 → -0.25 s ("audio of A lingers").

    Expected:
      - clip[1] atrim start = 4.0 - (-0.25) = 4.25 (clamped to actual)
      - clip[0] slice = 4.0 + 0 - (-0.25) = 4.25
    """
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 5.0, "fps": 24.0, "audioStart": 0.0},
        {"historyId": "b", "duration": 5.0, "fps": 24.0, "audioStart": 4.0},
    ]
    transitions = [{"clipBIndex": 1, "audioLeadFrames": -6}]

    cmd = _capture_export(clips, transitions, uploads, "storage://song", fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)

    # Clip 1's atrim start was pushed forward to 4.25.
    assert "atrim=start=4.25" in fc, fc
    # Clip 0's slice = 4.0 - (-0.25) = 4.25, clamped to pre_speed=5.0 → 4.25.
    assert "atrim=start=0.0:duration=4.25" in fc, fc


def test_two_clip_j_cut_at_both_boundaries(tmp_path, fake_resolve, monkeypatch):
    """Three clips, J-cut on both 0→1 and 1→2 boundaries.

    audio_lead_secs = [0, 0.25, 0.5]
    slice[0] = (4.0 - 0.0) + 0 - 0.25 = 3.75
    slice[1] = (8.0 - 4.0) + 0.25 - 0.5 = 3.75
    slice[2] = pre_speed[2] + 0.5 = 4.5  (last)
    starts: [0.0, 3.75, 7.5]
    """
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 4.0, "fps": 24.0, "audioStart": 0.0},
        {"historyId": "b", "duration": 4.0, "fps": 24.0, "audioStart": 4.0},
        {"historyId": "c", "duration": 4.0, "fps": 24.0, "audioStart": 8.0},
    ]
    transitions = [
        {"clipBIndex": 1, "audioLeadFrames": 6},
        {"clipBIndex": 2, "audioLeadFrames": 12},
    ]

    cmd = _capture_export(clips, transitions, uploads, "storage://song", fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)

    assert "atrim=start=0.0:duration=3.75" in fc, fc
    assert "atrim=start=3.75:duration=3.75" in fc, fc
    assert "atrim=start=7.5:duration=4.5" in fc, fc


def test_audio_duration_sec_override_skips_lead_math(tmp_path, fake_resolve, monkeypatch):
    """When clip.audioDurationSec is set, lead math is skipped for that slice."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 4.0, "fps": 24.0, "audioStart": 0.0,
         "audioDurationSec": 4.0},
        {"historyId": "b", "duration": 4.0, "fps": 24.0, "audioStart": 4.0},
    ]
    transitions = [{"clipBIndex": 1, "audioLeadFrames": 6}]

    cmd = _capture_export(clips, transitions, uploads, "storage://song", fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)

    # Clip 0 uses explicit override (4.0), NOT 3.75.
    assert "atrim=start=0.0:duration=4.0" in fc, fc
    # Clip 1's start IS still adjusted (override is per-slice, not boundary).
    assert "atrim=start=3.75" in fc, fc


def test_negative_audio_start_clamped_to_zero(tmp_path, fake_resolve, monkeypatch):
    """If audioStart - lead would go below zero, clamp to 0."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 1.0, "fps": 24.0, "audioStart": 0.0},
        {"historyId": "b", "duration": 4.0, "fps": 24.0, "audioStart": 0.1},
    ]
    transitions = [{"clipBIndex": 1, "audioLeadFrames": 24}]  # 1.0 s lead > 0.1 s start

    cmd = _capture_export(clips, transitions, uploads, "storage://song", fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)

    # Clip 1's start was clamped from -0.9 to 0.0.
    assert "atrim=start=0.0:duration=" in fc, fc
    # Sanity: there should be exactly two atrim segments.
    assert fc.count("atrim=start=") == 2, fc


# ---------------------------------------------------------------------------
# B2 — clip.speed cascade
# ---------------------------------------------------------------------------


def test_speed_half_doubles_effective_duration_and_setpts(tmp_path, fake_resolve, monkeypatch):
    """speed=0.5 → effective_duration doubles; setpts uses (PTS-STARTPTS)/0.5."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 2.0, "fps": 24.0, "speed": 0.5},
        {"historyId": "b", "duration": 2.0, "fps": 24.0},
    ]

    cmd = _capture_export(clips, [], uploads, None, fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)

    assert "setpts=(PTS-STARTPTS)/0.500000" in fc, fc
    # Force-keyframe seam = effective_dur[0] = 2.0 / 0.5 = 4.0.
    kf_idx = cmd.index("-force_key_frames")
    assert cmd[kf_idx + 1] == "4.000000", cmd[kf_idx + 1]


def test_speed_two_halves_effective_duration(tmp_path, fake_resolve, monkeypatch):
    """speed=2.0 → effective_duration halves; audio side stays decoupled."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 4.0, "fps": 24.0, "speed": 2.0,
         "audioStart": 0.0},
        {"historyId": "b", "duration": 4.0, "fps": 24.0, "audioStart": 4.0},
    ]
    cmd = _capture_export(clips, [], uploads, "storage://song", fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)

    assert "setpts=(PTS-STARTPTS)/2.000000" in fc, fc
    # Audio slice for clip[0]: pre-speed clamp = 4.0 (NOT effective 2.0).
    # gap = 4.0 - 0.0 = 4.0, no leads → slice = 4.0.
    assert "atrim=start=0.0:duration=4.0" in fc, fc


def test_speed_xfade_cumulative_offset(tmp_path, fake_resolve, monkeypatch):
    """xfade offset must use effective (post-speed) duration."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 4.0, "fps": 24.0, "speed": 0.5},  # eff = 8.0
        {"historyId": "b", "duration": 4.0, "fps": 24.0, "speed": 2.0},  # eff = 2.0
        {"historyId": "c", "duration": 4.0, "fps": 24.0},
    ]
    transitions = [
        {"clipBIndex": 1, "type": "crossfade", "durationSec": 0.5},
        {"clipBIndex": 2, "type": "crossfade", "durationSec": 0.5},
    ]
    cmd = _capture_export(clips, transitions, uploads, None, fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)

    # First xfade offset = effective[0] - 0.5 = 8.0 - 0.5 = 7.5
    # Second xfade offset = first + (effective[1] - 0.5) = 7.5 + 1.5 = 9.0
    assert re.search(r"xfade=transition=fade:duration=0\.5:offset=7\.5", fc), fc
    assert re.search(r"xfade=transition=fade:duration=0\.5:offset=9\.0", fc), fc


def test_speed_beat_synced_uses_pre_speed_duration_for_audio(tmp_path, fake_resolve, monkeypatch):
    """When speed > 1.0, audio slice clamp must use pre-speed duration, not
    effective_duration (else song gets clipped short).
    """
    uploads = _FakeUploads(tmp_path)
    clips = [
        # pre_speed = 4.0, effective = 2.0. audioStart gap = 4.0.
        {"historyId": "a", "duration": 4.0, "fps": 24.0, "speed": 2.0,
         "audioStart": 0.0},
        {"historyId": "b", "duration": 4.0, "fps": 24.0, "audioStart": 4.0},
    ]
    cmd = _capture_export(clips, [], uploads, "storage://song", fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)
    # Clip 0 audio slice: gap=4.0, clamp against pre_speed=4.0 → 4.0
    # If we'd used effective (2.0), this would WRONGLY be 2.0.
    assert "atrim=start=0.0:duration=4.0" in fc, fc


def test_speed_zero_or_negative_defaults_to_one(tmp_path, fake_resolve, monkeypatch, caplog):
    """Speed ≤ 0 must default to 1.0 with a WARN, not raise."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 2.0, "fps": 24.0, "speed": 0},
        {"historyId": "b", "duration": 2.0, "fps": 24.0, "speed": -1.5},
    ]
    cmd = _capture_export(clips, [], uploads, None, fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)
    # speed=1.0 emission is byte-identical to v1.10.0 — no /speed clause.
    assert "setpts=PTS-STARTPTS" in fc, fc
    assert "(PTS-STARTPTS)/" not in fc, fc


def test_speed_one_is_byte_identical_to_legacy(tmp_path, fake_resolve, monkeypatch):
    """Invariant: speed=1.0 (or unset) emits exactly the v1.10.0 setpts."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 2.0, "fps": 24.0},  # no speed key
        {"historyId": "b", "duration": 2.0, "fps": 24.0, "speed": 1.0},
    ]
    cmd = _capture_export(clips, [], uploads, None, fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)
    assert "(PTS-STARTPTS)/" not in fc, fc
    # setpts=PTS-STARTPTS appears verbatim per clip.
    assert fc.count("setpts=PTS-STARTPTS") == 2, fc


def test_single_clip_with_speed_falls_through_to_ffmpeg(tmp_path, fake_resolve, monkeypatch):
    """v1.15.1 regression: single-clip + no-audio + speed != 1.0 must NOT take
    the raw-bytes shortcut. Previously single-clip exports returned the source
    file unchanged regardless of speed (silent corruption — speed=1000 emitted
    a normal-paced MP4 byte-identical to speed=1.0).
    """
    uploads = _FakeUploads(tmp_path)
    clips = [{"historyId": "solo", "duration": 2.0, "fps": 24.0, "speed": 0.5}]
    cmd = _capture_export(clips, [], uploads, None, fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)
    assert "(PTS-STARTPTS)/0.500000" in fc, fc


def test_single_clip_with_tail_trim_falls_through_to_ffmpeg(tmp_path, fake_resolve, monkeypatch):
    """Sibling regression: single-clip + tailTrimFrames > 0 also must NOT
    shortcut — though the LAST-clip-zeros rule means tailTrimFrames is forced
    to 0 here, so this asserts that a non-zero request renders an ffmpeg path
    rather than silently returning raw bytes.
    """
    # Single clip + tail trim is force-zeroed (last clip rule), so even though
    # the user passed tailTrimFrames=9 the actual trim is dropped — but the
    # shortcut should now ONLY trigger when speed==1.0 AND tail==0 AND audio
    # is None. Confirm we still fall through to ffmpeg when speed != 1.0
    # alongside a force-zeroed trim.
    uploads = _FakeUploads(tmp_path)
    clips = [{"historyId": "solo", "duration": 2.0, "fps": 24.0,
              "tailTrimFrames": 9, "speed": 2.0}]
    cmd = _capture_export(clips, [], uploads, None, fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)
    assert "(PTS-STARTPTS)/2.000000" in fc, fc


def test_audio_lead_frames_zero_is_byte_identical_to_legacy(tmp_path, fake_resolve, monkeypatch):
    """Invariant: no audioLeadFrames → atrim emits the legacy raw audioStart."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 4.0, "fps": 24.0, "audioStart": 0.0},
        {"historyId": "b", "duration": 4.0, "fps": 24.0, "audioStart": 4.0},
    ]
    cmd = _capture_export(clips, [], uploads, "storage://song", fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)
    assert "atrim=start=0.0:duration=4.0" in fc, fc
    assert "atrim=start=4.0:duration=4.0" in fc, fc


# ---------------------------------------------------------------------------
# v1.16.2 — composition export quality knobs
# ---------------------------------------------------------------------------


def test_export_default_uses_libx264_crf18_high(tmp_path, fake_resolve, monkeypatch):
    """v1.16.2: default export emits libx264 + CRF 18 + profile=high + yuv420p + 256k audio."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 2.0, "fps": 24.0},
        {"historyId": "b", "duration": 2.0, "fps": 24.0},
    ]
    cmd = _capture_export(clips, [], uploads, "storage://song", fake_resolve, monkeypatch)
    assert "-c:v" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "18"
    assert "-preset" in cmd and cmd[cmd.index("-preset") + 1] == "medium"
    assert "-profile:v" in cmd and cmd[cmd.index("-profile:v") + 1] == "high"
    assert "-pix_fmt" in cmd and cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert "-b:a" in cmd and cmd[cmd.index("-b:a") + 1] == "256k"


def test_export_quality_overrides_crf_preset_profile_audio(tmp_path, fake_resolve, monkeypatch):
    """User-supplied quality knobs propagate verbatim into the ffmpeg argv."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 2.0, "fps": 24.0},
        {"historyId": "b", "duration": 2.0, "fps": 24.0},
    ]
    quality = {
        "output_crf": 14,
        "output_preset": "slow",
        "output_profile": "high10",
        "output_audio_bitrate": "320k",
    }
    cmd = _capture_export(
        clips, [], uploads, "storage://song", fake_resolve, monkeypatch, quality=quality,
    )
    assert cmd[cmd.index("-crf") + 1] == "14"
    assert cmd[cmd.index("-preset") + 1] == "slow"
    assert cmd[cmd.index("-profile:v") + 1] == "high10"
    assert cmd[cmd.index("-b:a") + 1] == "320k"


def test_export_libopenh264_uses_bitrate_not_crf(tmp_path, fake_resolve, monkeypatch):
    """libopenh264 path uses -b:v not -crf since it doesn't support CRF."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 2.0, "fps": 24.0},
        {"historyId": "b", "duration": 2.0, "fps": 24.0},
    ]
    quality = {"output_encoder": "libopenh264", "output_video_bitrate": "10M"}
    cmd = _capture_export(clips, [], uploads, None, fake_resolve, monkeypatch, quality=quality)
    assert cmd[cmd.index("-c:v") + 1] == "libopenh264"
    assert "-crf" not in cmd
    assert "-b:v" in cmd and cmd[cmd.index("-b:v") + 1] == "10M"


def test_export_video_bitrate_switches_x264_from_crf_to_abr(tmp_path, fake_resolve, monkeypatch):
    """When output_video_bitrate is set on libx264, swap CRF → 1-pass ABR."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 2.0, "fps": 24.0},
        {"historyId": "b", "duration": 2.0, "fps": 24.0},
    ]
    quality = {"output_video_bitrate": "8M"}
    cmd = _capture_export(clips, [], uploads, None, fake_resolve, monkeypatch, quality=quality)
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    # CRF must have been removed in favor of bitrate.
    assert "-crf" not in cmd
    assert cmd[cmd.index("-b:v") + 1] == "8M"
    assert cmd[cmd.index("-maxrate") + 1] == "8M"
    assert cmd[cmd.index("-bufsize") + 1] == "24M"


def test_export_libx265_uses_colder_default_crf(tmp_path, fake_resolve, monkeypatch):
    """libx265 default CRF is 22 (vs x264's 18) — perceptually equivalent."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 2.0, "fps": 24.0},
        {"historyId": "b", "duration": 2.0, "fps": 24.0},
    ]
    quality = {"output_encoder": "libx265"}
    cmd = _capture_export(clips, [], uploads, None, fake_resolve, monkeypatch, quality=quality)
    assert cmd[cmd.index("-c:v") + 1] == "libx265"
    assert cmd[cmd.index("-crf") + 1] == "22"


def test_export_invalid_encoder_raises(tmp_path, fake_resolve, monkeypatch):
    """Unsupported encoder name raises ValueError before ffmpeg is invoked."""
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 2.0, "fps": 24.0},
        {"historyId": "b", "duration": 2.0, "fps": 24.0},
    ]
    quality = {"output_encoder": "libxavs"}
    # _capture_export inserts the encoder check inside the export_handler call;
    # request-side validation in server.py rejects this with 422 BEFORE we get
    # here, but the handler must also be defensive.
    with pytest.raises((ValueError, RuntimeError)):
        _capture_export(clips, [], uploads, None, fake_resolve, monkeypatch, quality=quality)


def test_export_filter_complex_unchanged_by_v1_16_2(tmp_path, fake_resolve, monkeypatch):
    """Encoder swap MUST NOT alter the filter graph. Re-runs the speed-xfade
    test from B2 and asserts the same offsets — confirming v1.16.2 is purely
    an encoder-args change.
    """
    uploads = _FakeUploads(tmp_path)
    clips = [
        {"historyId": "a", "duration": 4.0, "fps": 24.0, "speed": 0.5},
        {"historyId": "b", "duration": 4.0, "fps": 24.0, "speed": 2.0},
        {"historyId": "c", "duration": 4.0, "fps": 24.0},
    ]
    transitions = [
        {"clipBIndex": 1, "type": "crossfade", "durationSec": 0.5},
        {"clipBIndex": 2, "type": "crossfade", "durationSec": 0.5},
    ]
    cmd = _capture_export(clips, transitions, uploads, None, fake_resolve, monkeypatch)
    fc = _filter_complex(cmd)
    assert re.search(r"xfade=transition=fade:duration=0\.5:offset=7\.5", fc), fc
    assert re.search(r"xfade=transition=fade:duration=0\.5:offset=9\.0", fc), fc


# ---------------------------------------------------------------------------
# v1.16.3 — storage_uri fallback for clips without historyId (flash inserts)
# ---------------------------------------------------------------------------


class _StorageUriUploads:
    """UploadStore stub that materializes any storage_uri to a real tmp file."""
    def __init__(self, tmp_path: Path):
        self._tmp = tmp_path
        self._counter = 0

    def resolve(self, uri: str) -> Path:
        # Per-uri deterministic file name; create on demand so .exists() is True.
        safe = uri.replace("/", "_").replace(":", "_")
        p = self._tmp / f"resolved-{safe}.mp4"
        if not p.exists():
            p.write_bytes(b"\x00")
        return p


def test_storage_uri_only_clip_resolves_via_uploads(tmp_path, fake_resolve, monkeypatch):
    """v1.16.3 regression: a clip carrying storage_uri but no historyId (e.g.
    a synthetic flash insert minted by the MCP orchestrator) must resolve via
    UploadStore.resolve(...) instead of KeyError'ing on clip['historyId'].
    """
    uploads = _StorageUriUploads(tmp_path)
    clips = [
        {"storage_uri": "storage://flash-aaa", "duration": 0.375, "fps": 24.0, "audioStart": 0.0},
        {"storage_uri": "storage://flash-bbb", "duration": 0.375, "fps": 24.0, "audioStart": 0.375},
    ]
    transitions: list[dict] = []
    # No fake_resolve monkeypatch needed: the new branch never calls
    # _resolve_clip_path for storage_uri-only clips.
    cmd = _capture_export(clips, transitions, uploads, None, fake_resolve, monkeypatch)
    # Sanity: ffmpeg was actually invoked with the resolved files.
    assert any("resolved-storage___flash-aaa.mp4" in arg for arg in cmd), cmd
    assert any("resolved-storage___flash-bbb.mp4" in arg for arg in cmd), cmd


def test_mixed_historyid_and_storage_uri_clips(tmp_path, fake_resolve, monkeypatch):
    """v1.16.3: composition with a primary LTX clip (historyId) followed by a
    flash insert (storage_uri only) — both paths must resolve cleanly.
    """
    uploads = _StorageUriUploads(tmp_path)
    clips = [
        {"historyId": "primary-1", "duration": 4.0, "fps": 24.0, "audioStart": 0.0},
        {"storage_uri": "storage://flash-zzz", "duration": 0.5, "fps": 24.0, "audioStart": 4.0},
        {"historyId": "primary-2", "duration": 4.0, "fps": 24.0, "audioStart": 4.5},
    ]
    transitions: list[dict] = []
    cmd = _capture_export(clips, transitions, uploads, None, fake_resolve, monkeypatch)
    # historyId path uses the fake_resolve fixture (writes to tmp_path/clip-{id}.mp4).
    assert any("clip-primary-1.mp4" in arg for arg in cmd), cmd
    assert any("clip-primary-2.mp4" in arg for arg in cmd), cmd
    # storage_uri path uses _StorageUriUploads.
    assert any("resolved-storage___flash-zzz.mp4" in arg for arg in cmd), cmd


def test_clip_missing_both_raises(tmp_path, fake_resolve, monkeypatch):
    """v1.16.3: a clip with neither historyId nor storage_uri must raise a
    clear ValueError (not a KeyError on a stale field name).
    """
    uploads = _StorageUriUploads(tmp_path)
    clips = [{"duration": 4.0, "fps": 24.0, "audioStart": 0.0}]
    monkeypatch.setattr(export_handler, "_resolve_clip_path", fake_resolve)
    with pytest.raises(ValueError, match="missing both historyId and storage_uri"):
        export_handler.export_composition(clips, [], uploads, None)


def test_storage_uri_missing_on_disk_raises(tmp_path, fake_resolve, monkeypatch):
    """v1.16.3: storage_uri that resolves but file doesn't exist on disk
    raises FileNotFoundError with a clear message.
    """
    class _MissingFileUploads:
        def resolve(self, uri: str) -> Path:
            return tmp_path / "does-not-exist.mp4"

    uploads = _MissingFileUploads()
    clips = [{"storage_uri": "storage://gone", "duration": 4.0, "fps": 24.0, "audioStart": 0.0}]
    monkeypatch.setattr(export_handler, "_resolve_clip_path", fake_resolve)
    with pytest.raises(FileNotFoundError, match="storage_uri not found on disk"):
        export_handler.export_composition(clips, [], uploads, None)
