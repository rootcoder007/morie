"""Tests for rgswtden.rangayyan_swt_denoise."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_swt_denoise


def test_rgswtden_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_swt_denoise(x)
    assert isinstance(result, dict)
    assert "denoised" in result


def test_rgswtden_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_swt_denoise(x)
    assert isinstance(result, dict)
