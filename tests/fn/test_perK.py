"""Tests for perK.periodic_kernel."""

from morie.fn import _array_core as np

from morie.fn.perK import periodic_kernel


def test_perK_basic():
    """Test basic functionality."""
    x1 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = periodic_kernel(x1)
    assert isinstance(result, dict)
    assert "K" in result


def test_perK_edge():
    """Test edge cases."""
    x1 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = periodic_kernel(x1)
    assert isinstance(result, dict)
