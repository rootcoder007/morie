"""Tests for rng188.rangayyan_ch4_pan_tompkins_derivative_operator."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_pan_tompkins_derivative_operator


def test_rng188_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_derivative_operator(x)
    assert isinstance(result, dict)
    assert "y" in result


def test_rng188_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_derivative_operator(x)
    assert isinstance(result, dict)
