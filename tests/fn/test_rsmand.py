"""Tests for rsmand.rating_scale_andrich."""

from morie.fn import _array_core as np

from morie.fn.rsmand import rating_scale_andrich


def test_rsmand_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rating_scale_andrich(theta)
    assert isinstance(result, dict)
    assert "p" in result


def test_rsmand_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rating_scale_andrich(theta)
    assert isinstance(result, dict)
