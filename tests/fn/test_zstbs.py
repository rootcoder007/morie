"""Tests for morie.fn.zstbs -- Turning bands simulation"""

from morie.fn import _array_core as np

from morie.fn.zstbs import turning_bands


class TestTurningBands:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = turning_bands(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = turning_bands(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
