"""Tests for josnv.joseph_seasonal_naive."""

from morie.fn import _array_core as np

from morie.fn.josnv import joseph_seasonal_naive


def test_josnv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    season = 5
    horizon = 5
    result = joseph_seasonal_naive(x, season, horizon)
    assert isinstance(result, dict)
    assert "forecast" in result


def test_josnv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    season = 5
    horizon = 5
    result = joseph_seasonal_naive(x, season, horizon)
    assert isinstance(result, dict)
