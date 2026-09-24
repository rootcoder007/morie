"""Tests for krigsv.variogram_fit."""

from morie.fn import _array_core as np

from morie.fn.krigsv import variogram_fit


def test_krigsv_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = variogram_fit(coords, values)
    assert isinstance(result, dict)
    assert "c0" in result


def test_krigsv_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = variogram_fit(coords, values)
    assert isinstance(result, dict)
