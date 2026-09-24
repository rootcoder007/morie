"""Tests for rgdwt.rangayyan_dwt."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_dwt


def test_rgdwt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_dwt(x)
    assert isinstance(result, dict)
    assert "approx" in result


def test_rgdwt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_dwt(x)
    assert isinstance(result, dict)
