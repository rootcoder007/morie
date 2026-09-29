"""Tests for volcorpst.vol_corradi_swan_persistence: statistics and Monte Carlo p-values recomputed."""

import math

from morie.fn._rng import random_normal
from morie.fn.volcorpst import vol_corradi_swan_persistence

R = [0.01 * math.sin(1.7 * t) + 0.004 * math.cos(0.3 * t * t) for t in range(120)]


def _fks(z):
    z = sorted(z)
    n = len(z)
    m = sum(z) / n
    s = math.sqrt(sum((v - m) ** 2 for v in z) / (n - 1))
    c = [0.5 * math.erfc(-(v - m) / (s * math.sqrt(2))) for v in z]
    return max(max((i + 1) / n - ci for i, ci in enumerate(c)), max(ci - i / n for i, ci in enumerate(c)))


def test_fitted_normal_path():
    out = vol_corradi_swan_persistence(R, horizons=(1, 4), n_mc=39, seed=3)
    for i, h in enumerate((1, 4)):
        agg = [sum(R[j * h : (j + 1) * h]) for j in range(120 // h)]
        d = _fks(agg)
        e = out["per_horizon"][i]
        assert abs(e["statistic"] - d) < 1e-12
        cnt = sum(
            1 for k in range(39) if _fks([float(v) for v in random_normal(len(agg), seed=3, stream=i * 39 + k)]) >= d
        )
        assert abs(e["p_value"] - (1 + cnt) / 40) < 1e-15
    assert abs(out["p_value"] - min(1.0, 2 * min(e["p_value"] for e in out["per_horizon"]))) < 1e-15


def test_supplied_cdf_uses_exact_kolmogorov():
    from morie.fn import _stats_core as stats

    def cdf(x, h):
        return 0.5 * math.erfc(-x / (0.01 * math.sqrt(h) * math.sqrt(2)))

    out = vol_corradi_swan_persistence(R, horizons=(4,), cdf=cdf)
    agg = sorted(sum(R[j * 4 : (j + 1) * 4]) for j in range(30))
    c = [cdf(v, 4) for v in agg]
    d = max(max((i + 1) / 30 - ci for i, ci in enumerate(c)), max(ci - i / 30 for i, ci in enumerate(c)))
    assert abs(out["statistic"] - d) < 1e-15
    assert abs(out["p_value"] - stats.kstwo.sf(d, 30)) < 1e-15
