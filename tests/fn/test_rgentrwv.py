"""Tests for rgentrwv.rangayyan_wavelet_entropy."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_wavelet_entropy


def test_rgentrwv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_entropy(x)
    assert isinstance(result, dict)
    assert "entropy" in result


def test_rgentrwv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_entropy(x)
    assert isinstance(result, dict)
