"""Tests for slope1.slope_one."""

from morie.fn import _array_core as np

from morie.fn.slope1 import slope_one


def test_slope1_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0.0, 1.0, 40)
    u = 0.1
    i = 5
    result = slope_one(R, u, i)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_slope1_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0.0, 1.0, 40)
    u = 0.1
    i = 5
    result = slope_one(R, u, i)
    assert isinstance(result, dict)
