"""Tests for morie.fn.svclh -- Heart of spatial game"""

from morie.fn import _array_core as np

from morie.fn.svclh import coalition_heart


class TestCoalitionHeart:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = coalition_heart(data)
        assert result.value is not None

    def test_output_type(self):
        result = coalition_heart(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
