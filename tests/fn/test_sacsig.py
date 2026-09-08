"""Tests for morie.fn.sacsig."""

from morie.fn import _array_core as np

from morie.fn.sacsig import sacsig


class TestSacsig:
    def test_basic(self):
        np.random.seed(62)
        resid = np.random.randn(20)
        n = 20
        result = sacsig(resid, n)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(62)
        resid = np.random.randn(20)
        n = 20
        result = sacsig(resid, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(62)
        resid = np.random.randn(20)
        n = 20
        result = sacsig(resid, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
