"""Tests for maedf.mae_metric."""

from morie.fn import _array_core as np

from morie.fn.maedf import mae_metric


def test_maedf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mae_metric(y, yhat)
    assert isinstance(result, dict)
    assert "mae" in result


def test_maedf_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mae_metric(y, yhat)
    assert isinstance(result, dict)
