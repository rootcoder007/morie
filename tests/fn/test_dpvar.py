"""Tests for dpvar.dp_variance."""

from morie.fn import _array_core as np

from morie.fn.dpvar import dp_variance


def test_dpvar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = -1.0
    b = 1.0
    epsilon = 1.0
    result = dp_variance(x, a, b, epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_dpvar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = -1.0
    b = 1.0
    epsilon = 1.0
    result = dp_variance(x, a, b, epsilon)
    assert isinstance(result, dict)
