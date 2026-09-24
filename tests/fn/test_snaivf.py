"""Tests for snaivf.seasonal_naive."""

from morie.fn import _array_core as np

from morie.fn.snaivf import seasonal_naive


def test_snaivf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    m = 5
    result = seasonal_naive(y, m)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_snaivf_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    m = 5
    result = seasonal_naive(y, m)
    assert isinstance(result, dict)
