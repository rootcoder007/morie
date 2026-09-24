"""Tests for rgpcgeeg.rangayyan_pcg_eeg_coupling."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_pcg_eeg_coupling


def test_rgpcgeeg_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pcg_eeg_coupling(pcg, eeg, fs)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rgpcgeeg_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 128)
    eeg = np.random.default_rng(42).normal(0, 1, 128)
    fs = 100.0
    result = rangayyan_pcg_eeg_coupling(pcg, eeg, fs)
    assert isinstance(result, dict)
    assert len(result) > 0
