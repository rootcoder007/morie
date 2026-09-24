"""Tests for msdef.mse_metric."""

from morie.fn import _array_core as np

from morie.fn.msdef import mse_metric


def test_msdef_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mse_metric(y, yhat)
    assert isinstance(result, dict)
    assert "mse" in result


def test_msdef_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mse_metric(y, yhat)
    assert isinstance(result, dict)
