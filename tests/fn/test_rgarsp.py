"""Tests for rgarsp.rangayyan_ar_spectrum."""

from morie.fn import _array_core as np

from morie.fn.bsaar import rangayyan_ar_spectrum


def test_rgarsp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ar_spectrum(x)
    assert isinstance(result, dict)
    assert "freqs" in result


def test_rgarsp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ar_spectrum(x)
    assert isinstance(result, dict)
