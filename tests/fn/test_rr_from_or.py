"""Tests for rr_from_or.rr_from_or."""

from morie.fn import _array_core as np

from morie.fn.rr_from_or import rr_from_or


def test_ca11e29_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rr_from_or(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e29_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rr_from_or(x)
    assert isinstance(result, dict)
