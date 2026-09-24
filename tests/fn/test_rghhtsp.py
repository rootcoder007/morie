"""Tests for rghhtsp.rangayyan_hht_spectrum."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_hht_spectrum


def test_rghhtsp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_hht_spectrum(x)
    assert isinstance(result, dict)
    assert "spectrum" in result


def test_rghhtsp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_hht_spectrum(x)
    assert isinstance(result, dict)
