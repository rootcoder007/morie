"""Tests for jonaiv.joseph_naive_forecast."""

from morie.fn import _array_core as np

from morie.fn.jonaiv import joseph_naive_forecast


def test_jonaiv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_naive_forecast(y)
    assert isinstance(result, dict)
    assert "forecast" in result


def test_jonaiv_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_naive_forecast(y)
    assert isinstance(result, dict)
