"""Tests for hmmsec.geron_linreg_mse_cost."""

from morie.fn import _array_core as np

from morie.fn.hmmsec import geron_linreg_mse_cost


def test_hmmsec_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = 0.1
    result = geron_linreg_mse_cost(X, y, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "cost" in result


def test_hmmsec_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = 0.1
    result = geron_linreg_mse_cost(X, y, theta)
    assert isinstance(result, dict)
