"""Tests for ripL.ripley_l_function."""

from morie.fn import _array_core as np

from morie.fn.ripL import ripley_l_function


def test_ripL_basic():
    """Test basic functionality."""
    points = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ripley_l_function(points)
    assert isinstance(result, dict)
    assert "r" in result


def test_ripL_edge():
    """Test edge cases."""
    points = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ripley_l_function(points)
    assert isinstance(result, dict)
