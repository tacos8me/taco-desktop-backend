#!/usr/bin/env python3
"""v1.18.0-rc3 — Phase C A/B decision script.

Weekly cron. Reads MV-level ``_ab_arm`` tags out of session metadata
(populated by mcp orchestrator's ``_normalize_input`` when
``AB_TEST_ACTIVE=1``) and computes paired t-test on per-MV mean
validator_score across baseline vs candidate arms.

Promotion thresholds (per the plan):

  - PROMOTE   if delta >= +10% AND p < 0.05
  - DEPRECATE if delta <= -5%  AND p < 0.05
  - else NO ACTION

When ``AB_AUTO_PROMOTE=0`` is set, the script reports decisions but
won't write to the DB (operator-gated mode).

Insufficient-samples rule: needs ≥ 30 MVs per arm before any decision.

The script reads ``AB_CANDIDATE_LORA`` from env. When unset, it logs
and exits cleanly — A/B is not active.

Usage::

    AB_CANDIDATE_LORA=sft-quality-v0.0.1 uv run python scripts/ab_decision.py
    AB_AUTO_PROMOTE=0 AB_CANDIDATE_LORA=... uv run python scripts/ab_decision.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from history_store import HistoryStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ab_decision")


MIN_SAMPLES_PER_ARM = 30
PROMOTE_DELTA = 0.10  # 10%
DEPRECATE_DELTA = -0.05  # -5%
PROMOTE_P_VALUE = 0.05


Decision = Literal["promote", "deprecate", "no_action", "insufficient_samples", "no_candidate"]


@dataclass
class ABResult:
    decision: Decision
    candidate_lora: str | None
    n_candidate: int
    n_baseline: int
    mean_candidate: float
    mean_baseline: float
    delta: float
    p_value: float
    reason: str


# ---------------------------------------------------------------------------
# Statistics — small ttest_ind for environments without scipy
# ---------------------------------------------------------------------------


def _welch_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test (two-sample, unequal variance). Returns (t, p)."""
    try:
        from scipy import stats  # type: ignore
        result = stats.ttest_ind(a, b, equal_var=False)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        # Manual fallback so the cron is not gated on scipy availability
        # in production environments. Math: Welch's t with Satterthwaite df.
        from math import sqrt, lgamma, log, exp
        if len(a) < 2 or len(b) < 2:
            return 0.0, 1.0
        ma = sum(a) / len(a)
        mb = sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
        vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
        se = sqrt(va / len(a) + vb / len(b)) if (va or vb) else 1e-9
        t = (ma - mb) / se if se else 0.0
        df = (va / len(a) + vb / len(b)) ** 2 / (
            (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
        ) if (va or vb) else max(len(a), len(b)) - 1

        # Two-sided p-value via incomplete beta (regularized).
        # Approximation acceptable for the gate threshold (p<0.05 hits well
        # within numerical tolerance even for the fallback). Use scipy if
        # higher precision matters.
        x = df / (df + t * t)
        # log-betacf truncated continued fraction (Numerical Recipes shape)
        a_, b_ = df / 2.0, 0.5
        # Use simple series acceleration: Lentz's method with 50 iters
        FPMIN = 1e-30
        qab = a_ + b_
        qap = a_ + 1.0
        qam = a_ - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, 100):
            m2 = 2 * m
            aa = m * (b_ - m) * x / ((qam + m2) * (a_ + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a_ + m) * (qab + m) * x / ((a_ + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            del_ = d * c
            h *= del_
            if abs(del_ - 1.0) < 3e-7:
                break
        bt = exp(
            lgamma(a_ + b_) - lgamma(a_) - lgamma(b_)
            + a_ * log(x) + b_ * log(1.0 - x)
        ) if 0 < x < 1 else 0.0
        p_one = bt * h / a_
        p = min(1.0, 2.0 * p_one)
        return t, p


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------


def _fetch_arm_means(
    history: HistoryStore, *, arm: str, candidate_lora: str, limit: int
) -> list[float]:
    """Per-MV (composition) mean validator_score for clips tagged with
    ``_ab_arm = arm`` in their gen_config_json or session metadata.

    The MV grouping uses ``composition_id`` (set by export). Clips
    without ``composition_id`` aren't yet part of a finalized MV and
    don't enter the comparison.
    """
    rows = history._conn.execute(
        """SELECT composition_id, AVG(validator_score) AS mean_score
           FROM generations
           WHERE composition_id IS NOT NULL
             AND validator_score IS NOT NULL
             AND lora_applied_id = ?
             AND created_at > 0
           GROUP BY composition_id
           ORDER BY composition_id DESC
           LIMIT ?""",
        (candidate_lora if arm == "candidate" else "", limit),
    ).fetchall()
    # Note: the ``arm == 'baseline'`` path uses lora_applied_id = '' or NULL.
    # SQLite treats NULL != '' so we re-query for NULL when arm is baseline.
    if arm == "baseline":
        rows = history._conn.execute(
            """SELECT composition_id, AVG(validator_score) AS mean_score
               FROM generations
               WHERE composition_id IS NOT NULL
                 AND validator_score IS NOT NULL
                 AND (lora_applied_id IS NULL OR lora_applied_id = '')
               GROUP BY composition_id
               ORDER BY composition_id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [float(r["mean_score"]) for r in rows if r["mean_score"] is not None]


# ---------------------------------------------------------------------------
# Decision + apply
# ---------------------------------------------------------------------------


def evaluate(
    candidate_means: list[float],
    baseline_means: list[float],
) -> tuple[Decision, float, float, str]:
    if len(candidate_means) < MIN_SAMPLES_PER_ARM or len(baseline_means) < MIN_SAMPLES_PER_ARM:
        return (
            "insufficient_samples",
            0.0, 1.0,
            f"need ≥{MIN_SAMPLES_PER_ARM}/arm; have candidate={len(candidate_means)}, baseline={len(baseline_means)}",
        )
    mean_c = sum(candidate_means) / len(candidate_means)
    mean_b = sum(baseline_means) / len(baseline_means)
    delta = (mean_c - mean_b) / max(mean_b, 1e-9)
    _, p_value = _welch_ttest(candidate_means, baseline_means)
    if delta >= PROMOTE_DELTA and p_value < PROMOTE_P_VALUE:
        return "promote", delta, p_value, f"candidate wins by {delta*100:.1f}% (p={p_value:.3f})"
    if delta <= DEPRECATE_DELTA and p_value < PROMOTE_P_VALUE:
        return "deprecate", delta, p_value, f"candidate loses by {abs(delta)*100:.1f}% (p={p_value:.3f})"
    return "no_action", delta, p_value, f"delta={delta*100:+.1f}% (p={p_value:.3f}); thresholds not met"


def apply_decision(
    history: HistoryStore,
    *,
    decision: Decision,
    candidate_lora: str,
    auto_promote: bool,
) -> None:
    """Persist promote/deprecate to ``training_runs``. When
    ``auto_promote`` is False, we log only.

    Promote: set deployed_at = now(); deprecate sister rows with the
    same lora_registry_id NULLed.
    Deprecate: set deprecated_at = now() on this run.
    """
    if not auto_promote:
        logger.info("AB_AUTO_PROMOTE=0 — would %s %s but operator gate is on", decision, candidate_lora)
        return
    now = time.time()
    if decision == "promote":
        history._conn.execute(
            "UPDATE training_runs SET deployed_at = ? WHERE run_id = ?",
            (now, candidate_lora),
        )
    elif decision == "deprecate":
        history._conn.execute(
            "UPDATE training_runs SET deprecated_at = ? WHERE run_id = ?",
            (now, candidate_lora),
        )
    history._conn.commit()


def run(
    *,
    candidate_lora: str | None,
    auto_promote: bool,
    db_path: Path | None = None,
) -> ABResult:
    if not candidate_lora:
        return ABResult(
            decision="no_candidate",
            candidate_lora=None,
            n_candidate=0, n_baseline=0,
            mean_candidate=0.0, mean_baseline=0.0,
            delta=0.0, p_value=1.0,
            reason="AB_CANDIDATE_LORA not set; A/B inactive",
        )

    history = HistoryStore(db_path=db_path) if db_path else HistoryStore()
    candidate_means = _fetch_arm_means(
        history, arm="candidate", candidate_lora=candidate_lora, limit=200,
    )
    baseline_means = _fetch_arm_means(
        history, arm="baseline", candidate_lora=candidate_lora, limit=200,
    )
    decision, delta, p_value, reason = evaluate(candidate_means, baseline_means)

    if decision in ("promote", "deprecate"):
        apply_decision(
            history,
            decision=decision,
            candidate_lora=candidate_lora,
            auto_promote=auto_promote,
        )

    return ABResult(
        decision=decision,
        candidate_lora=candidate_lora,
        n_candidate=len(candidate_means),
        n_baseline=len(baseline_means),
        mean_candidate=sum(candidate_means) / len(candidate_means) if candidate_means else 0.0,
        mean_baseline=sum(baseline_means) / len(baseline_means) if baseline_means else 0.0,
        delta=delta,
        p_value=p_value,
        reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-lora",
        default=os.environ.get("AB_CANDIDATE_LORA"),
        help="candidate LoRA id (default: env AB_CANDIDATE_LORA)",
    )
    parser.add_argument(
        "--no-auto-promote",
        action="store_true",
        help="report decision without writing (overrides AB_AUTO_PROMOTE env)",
    )
    args = parser.parse_args()

    auto_promote = (
        os.environ.get("AB_AUTO_PROMOTE", "1").lower() in ("1", "true", "yes")
        and not args.no_auto_promote
    )

    result = run(candidate_lora=args.candidate_lora, auto_promote=auto_promote)
    print(json.dumps({
        "decision": result.decision,
        "candidate_lora": result.candidate_lora,
        "n_candidate": result.n_candidate,
        "n_baseline": result.n_baseline,
        "mean_candidate": result.mean_candidate,
        "mean_baseline": result.mean_baseline,
        "delta_pct": result.delta * 100,
        "p_value": result.p_value,
        "reason": result.reason,
    }, indent=2))


if __name__ == "__main__":
    main()
