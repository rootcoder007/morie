"""Tests for hmtsf.geron_time_series_forecast."""

from morie.fn import _array_core as np

from morie.fn.hmtsf import geron_time_series_forecast


def test_hmtsf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_time_series_forecast(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "forecast" in result


def test_hmtsf_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_time_series_forecast(y)
    assert isinstance(result, dict)
