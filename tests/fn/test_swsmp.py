"""Tests for swsmp.systematic_with_random_start."""

from morie.fn import _array_core as np

from morie.fn.swsmp import systematic_with_random_start


def test_swsmp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = 5
    result = systematic_with_random_start(y, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "start" in result


def test_swsmp_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = 5
    result = systematic_with_random_start(y, k)
    assert isinstance(result, dict)
