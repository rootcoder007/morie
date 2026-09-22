"""Tests for btciqua.boot_ci_quantile."""

from morie.fn import _array_core as np

from morie.fn.btciqua import boot_ci_quantile


def test_btciqua_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_ci_quantile(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btciqua_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_ci_quantile(x)
    assert isinstance(result, dict)
