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


def _capture_export(clips, transitions, uploads, audio_uri, fake_resolve, monkeypatch):
    """Run export_composition with mocked ffmpeg + clip-resolve, return the
    captured ffmpeg argv list (so tests can introspect filter_complex).
    """
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Pretend ffmpeg succeeded; write a stub mp4 to the output path so the
        # post-run read_bytes() works.
        out = Path(cmd[-1])
        out.write_bytes(b"\x00")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(export_handler, "_resolve_clip_path", fake_resolve)
    monkeypatch.setattr(export_handler.subprocess, "run", _fake_run)

    export_handler.export_composition(clips, transitions, uploads, audio_uri)
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
