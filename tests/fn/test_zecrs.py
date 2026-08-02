"""Tests for morie.fn.zecrs -- Carstairs deprivation index"""

from morie.fn import _array_core as np

from morie.fn.zecrs import carstairs_index


class TestCarstairsIndex:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = carstairs_index(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = carstairs_index(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
