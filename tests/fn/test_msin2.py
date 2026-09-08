"""Tests for morie.fn.msin2 -- INDSCAL subject weights"""

from morie.fn import _array_core as np

from morie.fn.msin2 import indscal_weights


class TestIndscalWeights:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = indscal_weights(data)
        assert result.value is not None

    def test_output_type(self):
        result = indscal_weights(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
