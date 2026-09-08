"""Tests for morie.fn.ptvor -- Point pattern Voronoi intensities"""

from morie.fn import _array_core as np

from morie.fn.ptvor import pp_voronoi


class TestPpVoronoi:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pp_voronoi(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = pp_voronoi(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
