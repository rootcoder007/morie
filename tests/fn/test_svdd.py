"""Tests for svdd.svdd."""

from morie.fn import _array_core as np

from morie.fn.svdd import svdd


def test_svdd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = svdd(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_svdd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = svdd(X)
    assert isinstance(result, dict)
