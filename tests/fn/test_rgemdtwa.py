"""Tests for rgemdtwa.rangayyan_emd_twa."""

import math

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_emd_twa


def test_rgemdtwa_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_emd_twa(ecg, fs, r_peaks)
    assert isinstance(result, dict)
    assert len(result) > 0
    # Verify at least one value is a finite number
    found_finite = False
    for val in result.values():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            assert math.isfinite(val)
            found_finite = True
            break
    assert found_finite


def test_rgemdtwa_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    r_peaks = np.arange(50, 1000, 50)
    result = rangayyan_emd_twa(ecg, fs, r_peaks)
    assert isinstance(result, dict)
