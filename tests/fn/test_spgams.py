"""Tests for spgams.spatial_gams."""

from morie.fn import _array_core as np

from morie.fn.spgams import spatial_gams


def test_spgams_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = spatial_gams(y, x, coords)
    assert isinstance(result, dict)
    assert "fitted" in result


def test_spgams_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = spatial_gams(y, x, coords)
    assert isinstance(result, dict)
