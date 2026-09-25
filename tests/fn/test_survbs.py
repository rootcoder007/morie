"""Tests for survbs.survival_bootstrap_se (KM bootstrap beside Greenwood)."""

import math
import statistics

import pytest

from morie.fn import _array_core as np
from morie.fn.survbs import survival_bootstrap_se


def _km(t, e, v):
    """Product-limit S(v) and Greenwood's S^2 sum d/(n(n-d))."""
    s, acc = 1.0, 0.0
    for u in sorted({a for a, b in zip(t, e) if b and a <= v}):
        n = sum(1 for a in t if a >= u)
        d = sum(1 for a, b in zip(t, e) if a == u and b)
        s *= 1 - d / n
        if n > d:
            acc += d / (n * (n - d))
    return s, math.sqrt(s * s * acc)


def _data():
    t = [2.0, 3.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 14.0, 15.0]
    e = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
    return t, e


def test_survbs_basic():
    """Kaplan-Meier and Greenwood at each grid point match the hand
    product-limit; the bootstrap SE is the ddof-1 SD of the KM curves of
    subject-level resamples drawn with the seeded integers stream."""
    t, e = _data()
    grid = [3.0, 7.0, 12.0]
    r = survival_bootstrap_se(t, e, t_grid=grid, B=30, seed=4)
    for g, s, gw in zip(grid, r["survival"], r["greenwood_se"]):
        ks, kg = _km(t, e, g)
        assert float(s) == pytest.approx(ks, abs=1e-15)
        assert float(gw) == pytest.approx(kg, rel=1e-12)
    rng = np.random.default_rng(4)
    reps = []
    for _ in range(30):
        idx = [int(i) for i in rng.integers(0, 12, 12)]
        reps.append([_km([t[i] for i in idx], [e[i] for i in idx], g)[0] for g in grid])
    for j in range(3):
        assert float(r["bootstrap_se"][j]) == pytest.approx(statistics.stdev(row[j] for row in reps), rel=1e-12)
    assert r["n_events"] == 8


def test_survbs_edge():
    """Too few subjects or replicates and non-binary events raise."""
    t, e = _data()
    with pytest.raises(ValueError):
        survival_bootstrap_se(t[:4], e[:4])
    with pytest.raises(ValueError):
        survival_bootstrap_se(t, e, B=10)
    with pytest.raises(ValueError):
        survival_bootstrap_se(t, [2] * len(t))
