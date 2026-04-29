"""v1.19.0-rc1 — real Sapiens tier-2 payloads exercise the composite branch.

The rc2-shipped composite() already accepts the real shape (`pose_temporal_stability`
key present, `tier2_skipped` absent), but no test verified the real path: every
existing tier-2 test fed it stubs. These tests close that gap by feeding
real-shape payloads from the v1.19.0 sidecar and asserting the composite
arithmetic uses `0.2 × stability`, not `0.2 × 1.0`.
"""

import pytest


def test_composite_real_tier2_payload():
    """Real-shape tier2 with pose_temporal_stability=0.6 contributes 0.2*0.6 = 0.12,
    NOT 0.2*1.0 = 0.20. The composite must reflect that delta vs the stub path."""
    from validator import composite

    tier1 = {"dynamic_degree": 4.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2_real = {
        "pose_temporal_stability": 0.6,
        "pose_temporal_variance": 12.5,
        "human_detected": True,
        "identity_drift_frames": [],
        "frame_count": 14,
        "latency_s": 2.3,
        "stub": False,
    }
    tier3 = {"verdict": "pass", "score": 0.85, "judge_score": 0.85}

    out_real = composite(tier1, tier2_real, tier3)

    # The same call with a stub tier2 (1.0 contribution) for comparison.
    tier2_stub = {"tier2_skipped": True, "status": "stub"}
    out_stub = composite(tier1, tier2_stub, tier3)

    # Real path must be NUMERICALLY LOWER than stub path because 0.6 < 1.0.
    assert out_real["composite_score"] < out_stub["composite_score"]

    # The exact delta is the tier2 weight × (1.0 - 0.6) = 0.2 × 0.4 = 0.08.
    delta = out_stub["composite_score"] - out_real["composite_score"]
    assert abs(delta - 0.08) < 1e-6, f"unexpected delta {delta}; expected 0.08"

    # Real path is *not* flagged as stub or unreachable.
    assert "tier2_stub" not in out_real["reasoning_summary"]
    assert "tier2_unreachable" not in out_real["reasoning_summary"]


def test_composite_real_tier2_low_stability_pushes_to_warn():
    """Real tier2 stability low enough to drop composite from pass → warn.

    With tier1=0.8, tier3=0.7, baseline (tier2=1.0) → 0.4*0.8 + 0.2*1.0 + 0.4*0.7 = 0.80 (pass).
    With tier2_stability=0.1 → 0.4*0.8 + 0.2*0.1 + 0.4*0.7 = 0.62 (warn).
    """
    from validator import composite

    tier1 = {"dynamic_degree": 4.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2 = {
        "pose_temporal_stability": 0.1,
        "pose_temporal_variance": 9999.0,
        "human_detected": True,
        "identity_drift_frames": [3, 7, 11],
        "frame_count": 14,
        "stub": False,
    }
    tier3 = {"verdict": "pass", "score": 0.7, "judge_score": 0.7}

    out = composite(tier1, tier2, tier3)
    assert out["composite_score"] == pytest.approx(0.62, abs=1e-3)
    assert out["recommendation"] == "warn"


def test_composite_real_tier2_falls_back_to_variance():
    """If a sidecar version omits pose_temporal_stability but provides
    pose_temporal_variance, composite falls back to 1/(1+var). The fallback
    branch already exists in validator.composite — guard it with a real-shape
    test so a future schema change can't silently regress it."""
    from validator import composite

    tier1 = {"dynamic_degree": 4.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2 = {
        # No pose_temporal_stability key.
        "pose_temporal_variance": 1.0,  # 1/(1+1) = 0.5
        "human_detected": True,
        "stub": False,
    }
    tier3 = {"verdict": "pass", "score": 0.85, "judge_score": 0.85}

    out = composite(tier1, tier2, tier3)
    # 0.4*0.8 + 0.2*0.5 + 0.4*0.85 = 0.32 + 0.10 + 0.34 = 0.76
    assert out["composite_score"] == pytest.approx(0.76, abs=1e-3)


def test_composite_real_tier2_clamped_to_unit_range():
    """A bogus stability value outside [0, 1] is clamped — composite should
    not blow up if a future sidecar version emits raw scores."""
    from validator import composite

    tier1 = {"dynamic_degree": 4.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2_high = {"pose_temporal_stability": 1.5, "stub": False}
    tier2_neg = {"pose_temporal_stability": -0.5, "stub": False}
    tier3 = {"verdict": "pass", "score": 0.85, "judge_score": 0.85}

    out_high = composite(tier1, tier2_high, tier3)
    out_neg = composite(tier1, tier2_neg, tier3)
    # 1.5 clamps to 1.0 → 0.32 + 0.20 + 0.34 = 0.86
    assert out_high["composite_score"] == pytest.approx(0.86, abs=1e-3)
    # -0.5 clamps to 0.0 → 0.32 + 0.00 + 0.34 = 0.66
    assert out_neg["composite_score"] == pytest.approx(0.66, abs=1e-3)
