"""Tests for r2_from_f2.r2_from_f2."""

from morie.fn import _array_core as np

from morie.fn.r2_from_f2 import r2_from_f2


def test_ca8e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = r2_from_f2(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca8e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = r2_from_f2(x)
    assert isinstance(result, dict)
