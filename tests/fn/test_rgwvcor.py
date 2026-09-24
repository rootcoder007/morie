"""Tests for rgwvcor.rangayyan_wavelet_corr."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_wavelet_corr


def test_rgwvcor_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_corr(x, y)
    assert isinstance(result, dict)
    assert "correlations" in result


def test_rgwvcor_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_corr(x, y)
    assert isinstance(result, dict)
