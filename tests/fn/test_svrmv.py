"""Tests for morie.fn.svrmv -- Rabinowitz-Macdonald intensity component"""

from morie.fn import _array_core as np

from morie.fn.svrmv import rm_intensity


class TestRmIntensity:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rm_intensity(data)
        assert result.value is not None

    def test_output_type(self):
        result = rm_intensity(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
