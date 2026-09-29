"""Tests for morie.fn.t1smp -- One-sample t-test."""

from morie.fn.t1smp import one_sample_t_test


class TestOneSampleTTest:
    def test_known_mean(self):
        """Sample centered at 5 tested against mu0=0 should reject."""
        x = [4.5, 5.0, 5.5, 5.2, 4.8, 5.1, 5.3, 4.9]
        result = one_sample_t_test(x, mu0=0.0)
        assert isinstance(result, dict)
        assert "t" in result
        assert result["p_value"] < 0.001

    def test_correct_mu_not_significant(self):
        """Sample centered at 5 tested against mu0=5 should not reject."""
        x = [4.5, 5.0, 5.5, 5.2, 4.8, 5.1, 5.3, 4.9]
        result = one_sample_t_test(x, mu0=5.0)
        assert result["p_value"] > 0.05

    def test_returns_ci(self):
        """Result should contain CI bounds."""
        result = one_sample_t_test([1, 2, 3, 4, 5])
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["ci_upper"]


def test_t_statistic_and_interval_recomputed():
    import math

    import pytest

    from morie.fn import _stats_core as st

    x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1]
    n = 6
    m = sum(x) / n
    se = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)) / math.sqrt(n)
    tc = float(st.t(df=n - 1).ppf(0.975))
    r = one_sample_t_test(x, mu0=2.5)
    assert r["t"] == pytest.approx((m - 2.5) / se, rel=1e-12)
    assert (r["ci_lower"], r["ci_upper"]) == pytest.approx((m - tc * se, m + tc * se), rel=1e-12)
