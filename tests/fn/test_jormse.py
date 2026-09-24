"""Tests for jormse.joseph_rmse."""

from morie.fn import _array_core as np

from morie.fn.jormse import joseph_rmse


def test_jormse_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_rmse(y, yhat)
    assert isinstance(result, dict)
    assert "rmse" in result


def test_jormse_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_rmse(y, yhat)
    assert isinstance(result, dict)
