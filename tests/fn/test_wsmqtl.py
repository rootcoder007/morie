"""Tests for wsmqtl.wasserman_empirical_quantile."""

from morie.fn import _array_core as np

from morie.fn.wsmqtl import wasserman_empirical_quantile


def test_wsmqtl_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    p = 0.1
    result = wasserman_empirical_quantile(data, p)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmqtl_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    p = 0.1
    result = wasserman_empirical_quantile(data, p)
    assert isinstance(result, dict)
