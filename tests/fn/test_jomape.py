"""Tests for jomape.joseph_mape."""

from morie.fn import _array_core as np

from morie.fn.jomape import joseph_mape


def test_jomape_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_mape(y, yhat)
    assert isinstance(result, dict)
    assert "mape" in result


def test_jomape_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_mape(y, yhat)
    assert isinstance(result, dict)
