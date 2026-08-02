"""Tests for morie.fn.mscnt -- Continuity metric"""

from morie.fn import _array_core as np

from morie.fn.mscnt import continuity


class TestContinuity:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = continuity(data)
        assert result.value is not None

    def test_output_type(self):
        result = continuity(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
