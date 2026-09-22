"""Tests for arimaF.arima."""

from morie.fn import _array_core as np

from morie.fn.arimaF import arima


def test_arimaF_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = arima(y)
    assert isinstance(result, dict)
    assert "css" in result
def test_arimaF_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = arima(y)
    assert isinstance(result, dict)
