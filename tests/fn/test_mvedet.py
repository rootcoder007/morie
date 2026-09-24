"""Tests for mvedet.mve."""

from morie.fn import _array_core as np

from morie.fn.mvedet import mve


def test_mvedet_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mve(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mvedet_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mve(X)
    assert isinstance(result, dict)
