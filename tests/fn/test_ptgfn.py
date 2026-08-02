"""Tests for morie.fn.ptgfn -- Nearest-neighbor G-function"""

from morie.fn import _array_core as np

from morie.fn.ptgfn import g_function


class TestGFunction:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = g_function(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = g_function(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
