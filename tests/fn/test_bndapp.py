"""Tests for bndapp.bound_application."""

from morie.fn import _array_core as np

from morie.fn.bndapp import bound_application


def test_bndapp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bound_application(y, z)
    assert isinstance(result, dict)
    assert "levels" in result
def test_bndapp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = bound_application(y, z)
    assert isinstance(result, dict)
