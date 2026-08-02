"""Tests for morie.fn.zsgrz -- Zonal grid statistics"""

from morie.fn import _array_core as np

from morie.fn.zsgrz import grid_zonal


class TestGridZonal:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = grid_zonal(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = grid_zonal(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
