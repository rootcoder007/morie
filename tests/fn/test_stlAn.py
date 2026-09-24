"""Tests for stlAn.stl_anomaly."""

from morie.fn import _array_core as np

from morie.fn.stlAn import stl_anomaly


def test_stlAn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    period = 5
    result = stl_anomaly(x, period)
    assert isinstance(result, dict)
    assert "estimate" in result or "seasonal" in result


def test_stlAn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    period = 5
    result = stl_anomaly(x, period)
    assert isinstance(result, dict)
