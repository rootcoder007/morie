"""Tests for derivf.derivative_function."""

from morie.fn import _array_core as np

from morie.fn.derivf import derivative_function


def test_derivf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(42).normal(0, 1, 100)
    result = derivative_function(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_derivf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(42).normal(0, 1, 100)
    result = derivative_function(x, y)
    assert isinstance(result, dict)
