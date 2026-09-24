"""Tests for quanrg.quantile_regression."""

from morie.fn import _array_core as np

from morie.fn.quanrg import quantile_regression


def test_quanrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = quantile_regression(y)
    assert isinstance(result, dict)
    assert "coefficients" in result


def test_quanrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = quantile_regression(y)
    assert isinstance(result, dict)
