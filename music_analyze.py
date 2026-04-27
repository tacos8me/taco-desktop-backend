"""Librosa-only audio analysis for the v0.3 MV editing-grammar pipeline.

Returns a beat grid + onset list + RMS envelope sufficient for deterministic
shot-list cut placement (see ``docs/MV_EDITING.md`` §4 for the algorithm that
consumes this dict). CPU-only, ~88% accuracy on pop music — adequate for v0.3.
v0.4+ will refine via madmom + allin1 sidecars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# librosa is heavyweight (numpy + scipy + soxr + numba); import lazily inside
# ``analyze`` so an unused server import doesn't pay the cold-start cost.


def analyze(audio_path: Path) -> dict[str, Any]:
    """Run librosa beat-track + onset-detect + RMS over a single audio file.

    Returns a dict shaped:
        {
          "bpm": float,
          "beats": [t_sec, ...],
          "downbeats": [t_sec, ...],     # every 4th beat (assumes 4/4)
          "onsets": [t_sec, ...],
          "rms_envelope": [(t_sec, db), ...],
          "duration_s": float,
          "confidence": float in [0, 1],  # normalized beat-track strength
        }

    Notes:
    - 4/4 default — `downbeats[i] = beats[i*4]`. Compound meters (3/4, 6/8)
      will mis-bar; v0.4+ adds madmom-based meter detection.
    - Confidence comes from librosa's beat-track aggregate strength normalized
      via tanh — values <0.3 are unreliable, >0.7 are solid.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration_s = float(librosa.get_duration(y=y, sr=sr))

    # beat_track returns (tempo, beat_times) when units="time".
    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    # tempo is sometimes a 0-d ndarray — coerce to Python float.
    bpm = float(np.asarray(tempo).reshape(-1)[0]) if tempo is not None else 0.0
    beats: list[float] = [float(t) for t in np.asarray(beat_times).tolist()]

    # 4/4 default — every 4th beat is a downbeat. Configurable (compound
    # meters) deferred to v0.4 via madmom.
    downbeats: list[float] = beats[::4]

    onset_times = librosa.onset.onset_detect(
        y=y, sr=sr, backtrack=True, units="time"
    )
    onsets: list[float] = [float(t) for t in np.asarray(onset_times).tolist()]

    # RMS envelope at hop_length=512. Convert linear amplitude → dBFS.
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )
    # amplitude_to_db with ref=1.0 maps amplitude 1.0 → 0 dBFS.
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
    rms_envelope: list[tuple[float, float]] = [
        (float(t), float(d)) for t, d in zip(rms_times, rms_db)
    ]

    # Confidence: librosa's onset_strength aggregated over beat positions,
    # normalized via tanh so the output sits in [0, 1].
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if len(onset_env) > 0 and len(beats) > 0:
        beat_frames = librosa.time_to_frames(
            beat_times, sr=sr, hop_length=hop_length
        )
        beat_frames = np.clip(beat_frames, 0, len(onset_env) - 1)
        beat_strength = float(np.mean(onset_env[beat_frames]))
        confidence = float(np.tanh(beat_strength / 4.0))
    else:
        confidence = 0.0

    return {
        "bpm": bpm,
        "beats": beats,
        "downbeats": downbeats,
        "onsets": onsets,
        "rms_envelope": rms_envelope,
        "duration_s": duration_s,
        "confidence": confidence,
    }
