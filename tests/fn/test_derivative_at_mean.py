"""Tests for derivative_at_mean.derivative_at_mean."""

from morie.fn import _array_core as np

from morie.fn.derivative_at_mean import derivative_at_mean


def test_ca4e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = derivative_at_mean(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca4e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = derivative_at_mean(x)
    assert isinstance(result, dict)
