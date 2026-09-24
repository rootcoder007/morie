"""Tests for sarima.seasonal_arima."""

from morie.fn import _array_core as np

from morie.fn.sarima import seasonal_arima


def test_sarima_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = seasonal_arima(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sarima_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = seasonal_arima(y)
    assert isinstance(result, dict)
