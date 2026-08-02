"""Tests for morie.fn.nmrcy -- Roll call yea/nay summary"""

from morie.fn import _array_core as np

from morie.fn.nmrcy import roll_call_yea_nay


class TestRollCallYeaNay:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_yea_nay(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_yea_nay(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
