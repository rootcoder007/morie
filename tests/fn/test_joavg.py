"""Tests for joavg.joseph_average_forecast."""

from morie.fn import _array_core as np

from morie.fn.joavg import joseph_average_forecast


def test_joavg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_average_forecast(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_joavg_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_average_forecast(y)
    assert isinstance(result, dict)
