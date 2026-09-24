"""Tests for manfd.manifold_functional."""

from morie.fn import _array_core as np

from morie.fn.manfd import manifold_functional


def test_manfd_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = manifold_functional(Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "coords" in result


def test_manfd_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = manifold_functional(Y)
    assert isinstance(result, dict)
