"""Tests for hrzkqre.horowitz_kernel_quantile_reg."""

from morie.fn import _array_core as np

from morie.fn.hrzkqre import horowitz_kernel_quantile_reg


def test_hrzkqre_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_kernel_quantile_reg(x, y)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzkqre_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_kernel_quantile_reg(x, y)
    assert isinstance(result, dict)
