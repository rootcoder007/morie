"""Tests for rgeegsp.rangayyan_eeg_spectral."""

import pytest

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_eeg_spectral


def test_rgeegsp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_eeg_spectral(eeg, fs)
    assert isinstance(result, dict)
    assert "bands" in result
    assert "relative" in result
    assert "total_power" in result
    assert "freqs" in result
    assert "psd" in result
    assert "n_ch" in result
    assert "method" in result
    assert result["n_ch"] == 1
    for v in result["bands"].values():
        assert float(v) >= 0.0
    assert float(result["total_power"]) >= 0.0


def test_rgeegsp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 1, 512)
    with pytest.raises(ValueError):
        rangayyan_eeg_spectral(eeg, 50.0)
