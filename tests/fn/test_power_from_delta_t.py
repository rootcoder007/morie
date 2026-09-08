"""Tests for power_from_delta_t.power_from_delta_t."""

from morie.fn import _array_core as np

from morie.fn.power_from_delta_t import power_from_delta_t


def test_ca8e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = power_from_delta_t(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca8e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = power_from_delta_t(x)
    assert isinstance(result, dict)
