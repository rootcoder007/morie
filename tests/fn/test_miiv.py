"""Tests for morie.fn.miiv."""

from morie.fn import _array_core as np

from morie.fn.miiv import miiv


class TestMiiv:
    def test_basic(self):
        np.random.seed(92)
        resid = np.random.randn(20)
        W = np.eye(20) * 0.3
        result = miiv(resid, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(92)
        resid = np.random.randn(20)
        W = np.eye(20) * 0.3
        result = miiv(resid, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(92)
        resid = np.random.randn(20)
        W = np.eye(20) * 0.3
        result = miiv(resid, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
