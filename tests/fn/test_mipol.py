"""Tests for morie.fn.mipol — multiple imputation pooling (Rubin's rules)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.mipol import mi_pool


class TestMIPool:
    def test_pooled_is_mean(self):
        estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        res = mi_pool(estimates, variances)
        assert abs(res.extra["pooled_estimate"] - 3.0) < 1e-10

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            mi_pool(np.array([1.0]), np.array([0.1]))

    def test_se_positive(self):
        rng = np.random.default_rng(42)
        estimates = rng.standard_normal(10) + 5.0
        variances = rng.uniform(0.01, 0.5, size=10)
        res = mi_pool(estimates, variances)
        assert res.extra["se"] > 0
        assert res.extra["m"] == 10


def test_rubins_rules_recomputed():
    q = [1.2, 1.5, 0.9, 1.4]
    u = [0.10, 0.12, 0.09, 0.11]
    m = 4
    qb = sum(q) / m
    ub = sum(u) / m
    b = sum((t - qb) ** 2 for t in q) / (m - 1)
    T = ub + (1 + 1 / m) * b
    g = (1 + 1 / m) * b / T
    res = mi_pool(q, u)
    assert res.extra["total_variance"] == pytest.approx(T, rel=1e-13)
    assert res.extra["frac_missing_info"] == pytest.approx(g, rel=1e-13)
    assert res.extra["df"] == pytest.approx((m - 1) / g**2, rel=1e-12)
