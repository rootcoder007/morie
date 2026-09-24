"""Tests for rgrpsig.rangayyan_resp_signal."""

from morie.fn import _array_core as np
from morie.fn.bsaqrs import rangayyan_resp_signal


def test_rgrpsig_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ecg = rng.normal(0, 1, 1024)
    r_peaks = np.arange(50, 1000, 50)
    fs_out = 4.0
    result = rangayyan_resp_signal(ecg, r_peaks, fs_out)
    assert isinstance(result, dict)


def test_rgrpsig_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    ecg = rng.normal(0, 1, 256)
    r_peaks = np.arange(20, 180, 20)
    fs_out = 4.0
    result = rangayyan_resp_signal(ecg, r_peaks, fs_out)
    assert isinstance(result, dict)
