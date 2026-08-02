"""Tests for morie.fn.lacvgm."""

from morie.fn import _array_core as np

from morie.fn.lacvgm import lacvgm


class TestLacvgm:
    def test_basic(self):
        np.random.seed(207)
        y = np.random.randn(15)
        coords = np.random.rand(15, 2)
        result = lacvgm(y, coords)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(207)
        y = np.random.randn(15)
        coords = np.random.rand(15, 2)
        result = lacvgm(y, coords)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(207)
        y = np.random.randn(15)
        coords = np.random.rand(15, 2)
        result = lacvgm(y, coords)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
