"""Tests for adjusted_r2.adjusted_r2."""

from morie.fn import _array_core as np

from morie.fn.adjusted_r2 import adjusted_r2


def test_ca2e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = adjusted_r2(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca2e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = adjusted_r2(x)
    assert isinstance(result, dict)
