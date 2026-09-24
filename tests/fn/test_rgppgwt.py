"""Tests for rgppgwt.rangayyan_ppg_wavelet."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_ppg_wavelet


def test_rgppgwt_basic():
    """Test basic functionality."""
    ppg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    wavelet = "db4"
    levels = 4
    result = rangayyan_ppg_wavelet(ppg, fs, wavelet, levels)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rgppgwt_edge():
    """Test edge cases."""
    ppg = np.random.default_rng(42).normal(0, 1, 256)
    fs = 100.0
    wavelet = "db4"
    levels = 2
    result = rangayyan_ppg_wavelet(ppg, fs, wavelet, levels)
    assert isinstance(result, dict)
    assert len(result) > 0
