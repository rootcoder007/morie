"""Tests for hrzllqr.horowitz_local_linear_quantile."""

from morie.fn import _array_core as np

from morie.fn.hrzllqr import horowitz_local_linear_quantile


def test_hrzllqr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_local_linear_quantile(x, y)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzllqr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_local_linear_quantile(x, y)
    assert isinstance(result, dict)
