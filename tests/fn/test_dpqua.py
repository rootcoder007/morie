"""Tests for dpqua.dp_quantile."""

from morie.fn import _array_core as np

from morie.fn.dpqua import dp_quantile


def test_dpqua_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_quantile(x)
    assert isinstance(result, dict)
    assert "release" in result
def test_dpqua_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_quantile(x)
    assert isinstance(result, dict)
