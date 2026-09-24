"""Tests for rgtwamx.rangayyan_twa_spectral_mx."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_twa_spectral_mx


def test_rgtwamx_basic():
    """Test basic functionality."""
    ecg = 5
    fs = 5
    r_peaks = np.array([1, 0, 2, 2, 3, 3, 1, 0, 4, 3, 2, 3, 0, 1, 1, 3, 3, 0, 1, 1, 3, 2, 4, 0, 4, 3, 1, 3, 0, 2, 2, 4, 1, 4, 2, 2, 3, 0, 0, 3])
    result = rangayyan_twa_spectral_mx(ecg, fs, r_peaks)
    assert isinstance(result, dict)
    assert "alternans_voltage" in result


def test_rgtwamx_edge():
    """Test edge cases."""
    ecg = 5
    fs = 5
    r_peaks = np.array([1, 0, 2, 2, 3, 3, 1, 0, 4, 3, 2, 3, 0, 1, 1, 3, 3, 0, 1, 1, 3, 2, 4, 0, 4, 3, 1, 3, 0, 2, 2, 4, 1, 4, 2, 2, 3, 0, 0, 3])
    result = rangayyan_twa_spectral_mx(ecg, fs, r_peaks)
    assert isinstance(result, dict)
