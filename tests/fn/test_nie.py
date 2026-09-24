"""Tests for nie.natural_indirect_effect."""

from morie.fn import _array_core as np

from morie.fn.nie import natural_indirect_effect


def test_nie_basic():
    """Test basic functionality."""
    y11 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y10 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = natural_indirect_effect(y11, y10)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_nie_edge():
    """Test edge cases."""
    y11 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y10 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = natural_indirect_effect(y11, y10)
    assert isinstance(result, dict)
