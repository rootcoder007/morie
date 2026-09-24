"""Tests for naivef.naive_forecast."""

from morie.fn import _array_core as np

from morie.fn.naivef import naive_forecast


def test_naivef_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = naive_forecast(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_naivef_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = naive_forecast(y)
    assert isinstance(result, dict)
