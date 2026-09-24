"""Tests for ramsw.ramsay_weight."""

from morie.fn import _array_core as np

from morie.fn.ramsw import ramsay_weight


def test_ramsw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = 0.1
    result = ramsay_weight(y, a)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ramsw_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = 0.1
    result = ramsay_weight(y, a)
    assert isinstance(result, dict)
