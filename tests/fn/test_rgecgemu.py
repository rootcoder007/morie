"""Tests for rgecgemu.rangayyan_ecg_emg_coupling."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ecg_emg_coupling


def test_rgecgemu_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ecg = rng.normal(0, 1, 1024)
    emg = rng.normal(0, 1, 1024)
    qrs = [100, 300, 500, 700]
    fs = 500
    result = rangayyan_ecg_emg_coupling(ecg, emg, qrs, fs)
    assert isinstance(result, dict)


def test_rgecgemu_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    ecg = rng.normal(0, 1, 64)
    emg = rng.normal(0, 1, 64)
    qrs = [10, 30, 50]
    fs = 250
    result = rangayyan_ecg_emg_coupling(ecg, emg, qrs, fs)
    assert isinstance(result, dict)
