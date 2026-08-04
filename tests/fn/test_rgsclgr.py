"""Tests for rgsclgr.rangayyan_scalogram."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_scalogram


def test_rgsclgr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    scales = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = "morl"
    result = rangayyan_scalogram(x, fs, scales, wavelet)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgsclgr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    scales = np.random.default_rng(42).normal(0, 1, 100)
    wavelet = "morl"
    result = rangayyan_scalogram(x, fs, scales, wavelet)
    assert isinstance(result, dict)
