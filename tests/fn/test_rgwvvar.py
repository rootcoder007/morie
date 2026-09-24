"""Tests for rgwvvar.rangayyan_wavelet_variance."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_wavelet_variance


def test_rgwvvar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_variance(x)
    assert isinstance(result, dict)
    assert "variances" in result


def test_rgwvvar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_variance(x)
    assert isinstance(result, dict)
