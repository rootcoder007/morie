"""Tests for rmsef.rmse_metric."""

from morie.fn import _array_core as np

from morie.fn.rmsef import rmse_metric


def test_rmsef_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rmse_metric(y, yhat)
    assert isinstance(result, dict)
    assert "rmse" in result


def test_rmsef_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rmse_metric(y, yhat)
    assert isinstance(result, dict)
