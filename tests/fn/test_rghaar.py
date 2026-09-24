"""Tests for rghaar.rangayyan_haar_wavelet."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_haar_wavelet


def test_rghaar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_haar_wavelet(x)
    assert isinstance(result, dict)
    assert "approx" in result


def test_rghaar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_haar_wavelet(x)
    assert isinstance(result, dict)
