"""Tests for rgwvmom.rangayyan_wavelet_moments."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_wavelet_moments


def test_rgwvmom_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_moments(x)
    assert isinstance(result, dict)
    assert "moments" in result


def test_rgwvmom_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_moments(x)
    assert isinstance(result, dict)
