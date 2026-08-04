"""Tests for rgemgpk.rangayyan_emg_peak_freq."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_emg_peak_freq


def test_rgemgpk_basic():
    """Test basic functionality."""
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_emg_peak_freq(emg, fs)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgemgpk_edge():
    """Test edge cases."""
    emg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_emg_peak_freq(emg, fs)
    assert isinstance(result, dict)
