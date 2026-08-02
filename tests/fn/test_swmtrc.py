"""Tests for morie.fn.swmtrc."""

from morie.fn import _array_core as np

from morie.fn.swmtrc import swmtrc


class TestSwmtrc:
    def test_basic(self):
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        result = swmtrc(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        result = swmtrc(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        result = swmtrc(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
