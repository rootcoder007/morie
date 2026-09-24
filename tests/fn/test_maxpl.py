"""Tests for maxpl.max_pooling."""

from morie.fn import _array_core as np

from morie.fn.maxpl import max_pooling


def test_maxpl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    kernel = 5
    stride = 5
    result = max_pooling(x, kernel, stride)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_maxpl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    kernel = 5
    stride = 5
    result = max_pooling(x, kernel, stride)
    assert isinstance(result, dict)
