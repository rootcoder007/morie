"""Tests for rgeegar.rangayyan_eeg_autocorr."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_eeg_autocorr


def test_rgeegar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.5
    result = rangayyan_eeg_autocorr(x, fs)
    assert isinstance(result, dict)
    assert "acf" in result


def test_rgeegar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.5
    result = rangayyan_eeg_autocorr(x, fs)
    assert isinstance(result, dict)
