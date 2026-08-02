"""Tests for morie.fn.swrank."""

from morie.fn import _array_core as np

from morie.fn.swrank import swrank


class TestSwrank:
    def test_basic(self):
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        result = swrank(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        result = swrank(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        result = swrank(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
