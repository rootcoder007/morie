"""Tests for morie.fn.rubin -- Rubin's rules for pooling MI estimates."""

import pytest

from morie.fn.rubin import rubins_rules


class TestRubinsRules:
    def test_pooled_estimate_is_mean(self):
        """Pooled estimate should equal the mean of the input estimates."""
        result = rubins_rules([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
        assert result["pooled_estimate"] == pytest.approx(2.0)

    def test_ci_contains_estimate(self):
        """CI should bracket the pooled estimate."""
        result = rubins_rules([1.0, 1.1, 0.9, 1.05, 0.95], [0.2, 0.2, 0.2, 0.2, 0.2])
        assert result["ci_lower"] < result["pooled_estimate"] < result["ci_upper"]

    def test_identical_estimates_narrow_ci(self):
        """When all estimates agree, between-variance is 0 and CI is narrow."""
        result = rubins_rules([5.0, 5.0, 5.0, 5.0, 5.0], [0.1, 0.1, 0.1, 0.1, 0.1])
        assert result["between_var"] == pytest.approx(0.0)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width < 1.0

    def test_too_few_raises(self):
        """Fewer than 2 estimates should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            rubins_rules([1.0], [0.5])


def test_barnard_rubin_pool_recomputed():

    q = [1.2, 1.5, 0.9, 1.4]
    se = [0.3, 0.35, 0.28, 0.32]
    m = 4
    qb = sum(q) / m
    ub = sum(s * s for s in se) / m
    b = sum((t - qb) ** 2 for t in q) / (m - 1)
    T = ub + (1 + 1 / m) * b
    r_ = (1 + 1 / m) * b / ub
    res = rubins_rules(q, se)
    assert res["total_var"] == pytest.approx(T, rel=1e-13)
    assert res["df"] == pytest.approx((m - 1) * (1 + ub / ((1 + 1 / m) * b)) ** 2, rel=1e-12)
    assert res["fmi"] == pytest.approx((r_ + 2 / (m + 1)) / (1 + r_), rel=1e-12)
