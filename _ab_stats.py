"""Shared A/B statistics helpers.

Extracted from ``scripts/ab_decision.py`` so both the cron script and
the ``GET /v1/system/ab-status`` endpoint can compute the same Welch's
t-test result without duplicating the ~70 LOC numerical fallback.
"""
from __future__ import annotations


def welch_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test (two-sample, unequal variance). Returns (t, p).

    Uses scipy when available; falls back to a manual incomplete-beta
    approximation that's accurate well within the p<0.05 gate.
    """
    try:
        from scipy import stats  # type: ignore
        result = stats.ttest_ind(a, b, equal_var=False)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
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

        x = df / (df + t * t)
        a_, b_ = df / 2.0, 0.5
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


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy default).

    Empty input returns 0.0.
    """
    if not values:
        return 0.0
    arr = sorted(values)
    if len(arr) == 1:
        return float(arr[0])
    idx = (pct / 100.0) * (len(arr) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    frac = idx - lo
    return float(arr[lo] + (arr[hi] - arr[lo]) * frac)
