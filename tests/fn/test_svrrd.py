"""Tests for morie.fn.svrrd -- Roemer party unanimity model"""

from morie.fn import _array_core as np

from morie.fn.svrrd import roemer_model


class TestRoemerModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roemer_model(data)
        assert result.value is not None

    def test_output_type(self):
        result = roemer_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
