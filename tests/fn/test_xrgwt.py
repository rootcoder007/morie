"""Tests for morie.fn.xrgwt -- GWR local t-values"""

from morie.fn import _array_core as np

from morie.fn.xrgwt import gwr_tvalues


class TestGwrTvalues:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gwr_tvalues(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gwr_tvalues(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
