"""Tests for rgstfts.rangayyan_stft_spectrogram."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_stft_spectrogram


def test_rgstfts_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 128)
    fs = 100.0
    nperseg = 32
    noverlap = 16
    window = 'hann'
    result = rangayyan_stft_spectrogram(x, fs, nperseg, noverlap, window)
    assert isinstance(result, dict)
    assert any(k in result for k in ('f', 't', 'Sxx', 'sxx', 'spectrogram', 'SXX'))


def test_rgstfts_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 64)
    fs = 50.0
    nperseg = 16
    noverlap = 8
    window = 'hann'
    result = rangayyan_stft_spectrogram(x, fs, nperseg, noverlap, window)
    assert isinstance(result, dict)
