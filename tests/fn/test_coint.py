"""Tests for morie.fn.coint — cointegration test."""

from morie.fn import _array_core as np

from morie.fn.coint import cointegration_test


class TestCointegration:
    def test_cointegrated_pair(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(200))
        y = 0.5 * x + rng.standard_normal(200) * 0.5
        res = cointegration_test(x, y)
        assert res.statistic < 0

    def test_returns_gamma(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(100))
        y = np.cumsum(rng.standard_normal(100))
        res = cointegration_test(x, y)
        assert "gamma" in res.extra
