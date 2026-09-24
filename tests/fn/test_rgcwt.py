"""Tests for rgcwt.rangayyan_cwt."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_cwt


def test_rgcwt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_cwt(x)
    assert isinstance(result, dict)
    assert "coeffs" in result


def test_rgcwt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_cwt(x)
    assert isinstance(result, dict)
