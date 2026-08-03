"""Tests for or_from_rr.or_from_rr."""

from morie.fn import _array_core as np

from morie.fn.or_from_rr import or_from_rr


def test_ca11e28_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = or_from_rr(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e28_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = or_from_rr(x)
    assert isinstance(result, dict)
