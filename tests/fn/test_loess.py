"""Tests for loess.loess."""

from morie.fn import _array_core as np

from morie.fn.loess import loess


def test_loess_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = loess(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_loess_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = loess(x, y)
    assert isinstance(result, dict)
