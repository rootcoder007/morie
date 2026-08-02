"""Tests for morie.fn.ptpoi -- Homogeneous Poisson point process"""

from morie.fn import _array_core as np

from morie.fn.ptpoi import poisson_process


class TestPoissonProcess:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = poisson_process(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = poisson_process(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
