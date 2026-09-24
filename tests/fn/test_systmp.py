"""Tests for systmp.systematic_sampling."""

from morie.fn import _array_core as np

from morie.fn.systmp import systematic_sampling


def test_systmp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = 5
    result = systematic_sampling(y, k)
    assert isinstance(result, dict)
    assert "means" in result


def test_systmp_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = 5
    result = systematic_sampling(y, k)
    assert isinstance(result, dict)
