"""Tests for morie.fn.svrcs -- Roll call simulation"""

from morie.fn import _array_core as np

from morie.fn.svrcs import roll_call_sim


class TestRollCallSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_sim(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
