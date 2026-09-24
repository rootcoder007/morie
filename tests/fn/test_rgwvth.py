"""Tests for rgwvth.rangayyan_wavelet_threshold."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_wavelet_threshold


def test_rgwvth_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_threshold(x)
    assert isinstance(result, dict)
    assert "denoised" in result


def test_rgwvth_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_threshold(x)
    assert isinstance(result, dict)
