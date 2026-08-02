"""Tests for morie.fn.pproc — point process intensity."""

from morie.fn import _array_core as np
import pytest

from morie.fn.pproc import point_process_intensity


class TestPointProcess:
    def test_basic(self):
        pts = np.random.default_rng(42).uniform(0, 10, (50, 2))
        res = point_process_intensity(pts, bandwidth=1.0)
        assert res.value > 0

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError):
            point_process_intensity(np.ones((5, 3)))
