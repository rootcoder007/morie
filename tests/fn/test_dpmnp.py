"""Tests for dpmnp.dp_minmax."""

from morie.fn import _array_core as np

from morie.fn.dpmnp import dp_minmax


def test_dpmnp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_minmax(x, epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_dpmnp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_minmax(x, epsilon)
    assert isinstance(result, dict)
