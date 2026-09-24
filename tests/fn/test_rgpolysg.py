"""Tests for rgpolysg.rangayyan_polysomnography."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_polysomnography


def test_rgpolysg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 1024
    eeg = rng.normal(0, 1, n)
    eog = rng.normal(0, 1, n)
    emg = rng.normal(0, 1, n)
    fs = 100.0
    epoch_len = 2.0
    result = rangayyan_polysomnography(eeg, eog, emg, fs, epoch_len)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rgpolysg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 200
    eeg = rng.normal(0, 1, n)
    eog = rng.normal(0, 1, n)
    emg = rng.normal(0, 1, n)
    fs = 100.0
    epoch_len = 2.0
    result = rangayyan_polysomnography(eeg, eog, emg, fs, epoch_len)
    assert isinstance(result, dict)
    assert len(result) > 0
