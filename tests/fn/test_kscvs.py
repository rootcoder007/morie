"""Tests for morie.fn.kscvs — kernel-smoothed KS test."""

import pytest

from morie.fn import _array_core as np
from morie.fn.kscvs import kscvs


class TestKscvs:
    def test_normal_not_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kscvs(data, cdf_func="normal", n_boot=199, seed=42)
        assert res["p_value"] > 0.01

    def test_nonnormal_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(1.0, 200)
        res = kscvs(data, cdf_func="normal", n_boot=199, seed=42)
        assert res["p_value"] < 0.1

    def test_statistic_positive(self):
        data = np.arange(1, 51, dtype=float)
        res = kscvs(data, n_boot=99, seed=42)
        assert res["statistic"] > 0

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kscvs(np.array([1.0, 2.0, 3.0]))

    def test_raises_unknown_cdf(self):
        with pytest.raises(ValueError):
            kscvs(np.arange(10, dtype=float), cdf_func="bogus")


def test_statistic_and_bootstrap_p_recomputed():
    """sup |F_h - Phi((x - xbar)/s)| on the grid, parametric-bootstrap p."""
    import math

    from morie.fn import _array_core as np

    x = np.array([2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 4.2, 2.5, 3.3, 2.7])
    n, G, B = 10, 32, 6
    Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))  # noqa: E731

    def summ(v):
        m = sum(v) / len(v)
        return m, math.sqrt(sum((t - m) ** 2 for t in v) / (len(v) - 1))

    xs = [float(v) for v in x]
    m, s = summ(xs)
    bw = 0.4
    grid = [min(xs) - 4 * bw + (max(xs) - min(xs) + 8 * bw) * i / (G - 1) for i in range(G)]

    def stat(v):
        mv, sv = summ(v)
        return max(abs(sum(Phi((g - t) / bw) for t in v) / len(v) - Phi((g - mv) / sv)) for g in grid)

    D = stat(xs)
    rng = np.random.default_rng(1)
    bs = [stat([float(t) for t in rng.normal(m, s, n)]) for _ in range(B)]
    res = kscvs(x, bw=bw, n_grid=G, n_boot=B, seed=1)
    assert res["statistic"] == pytest.approx(D, rel=1e-10)
    assert res["p_value"] == sum(b >= D for b in bs) / B
