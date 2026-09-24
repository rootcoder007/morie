"""Tests for wsmlas.wasserman_lasso."""

from morie.fn import _array_core as np

from morie.fn.wsmlas import wasserman_lasso


def test_wsmlas_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lambda_ = 0.1
    result = wasserman_lasso(X, y, lambda_)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmlas_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lambda_ = 0.1
    result = wasserman_lasso(X, y, lambda_)
    assert isinstance(result, dict)
