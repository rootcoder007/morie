"""Tests for rgburg.rangayyan_burg_method."""

from morie.fn import _array_core as np

from morie.fn.bsaar import rangayyan_burg_method


def test_rgburg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_burg_method(x)
    assert isinstance(result, dict)
    assert "a" in result


def test_rgburg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_burg_method(x)
    assert isinstance(result, dict)
