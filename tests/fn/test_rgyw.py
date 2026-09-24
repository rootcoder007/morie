"""Tests for rgyw.rangayyan_yule_walker."""

from morie.fn import _array_core as np

from morie.fn.bsaar import rangayyan_yule_walker


def test_rgyw_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_yule_walker(x)
    assert isinstance(result, dict)
    assert "a" in result


def test_rgyw_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_yule_walker(x)
    assert isinstance(result, dict)
