"""Tests for morie.fn.msshr -- Shepard residuals"""

from morie.fn import _array_core as np

from morie.fn.msshr import shepard_resid


class TestShepardResid:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = shepard_resid(data)
        assert result.value is not None

    def test_output_type(self):
        result = shepard_resid(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
