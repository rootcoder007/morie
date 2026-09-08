"""Tests for morie.fn.zssgs -- Sequential Gaussian simulation"""

from morie.fn import _array_core as np

from morie.fn.zssgs import seq_gauss_sim


class TestSeqGaussSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = seq_gauss_sim(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = seq_gauss_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
