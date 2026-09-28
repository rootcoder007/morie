"""Tests for zsstk.st_kriging."""

from morie.fn.zsstk import st_kriging

M = {"type": "separable", "sill": 1.0, "space": {"model": "Exp", "range": 1.0}, "time": {"model": "Exp", "range": 2.0}}


class TestZsstk:
    def test_basic(self):
        result = st_kriging([1.0, 2.0, 1.5], [(0, 0), (1, 0), (0, 1)], [0, 0, 1], [(0.5, 0.5)], [0.5], M)
        assert result.statistic is not None

    def test_returns_spatial_result(self):
        result = st_kriging([1.0, 2.0, 3.0], [(0, 0), (1, 0), (2, 0)], [0, 1, 2], [(1, 0)], [1], M)
        assert hasattr(result, "statistic")
        assert abs(result.local_values[0] - 2.0) < 1e-12  # an observed point is reproduced
