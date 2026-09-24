"""Tests for krrFDA.kernel_ridge_regression."""

from morie.fn import _array_core as np

from morie.fn.krrFDA import kernel_ridge_regression


def test_krrFDA_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kernel_ridge_regression(X, y)
    assert isinstance(result, dict)
    assert "x_eval" in result


def test_krrFDA_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kernel_ridge_regression(X, y)
    assert isinstance(result, dict)
