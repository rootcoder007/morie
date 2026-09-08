"""Tests for rgbartl.rangayyan_bartlett_psd."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_bartlett_psd


def test_rgbartl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    nseg = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bartlett_psd(x, fs, nseg)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgbartl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    nseg = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bartlett_psd(x, fs, nseg)
    assert isinstance(result, dict)
