"""Tests for rgfeatex.rangayyan_feature_extract_bci."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_feature_extract_bci


def test_rgfeatex_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 1, 2048)
    fs = 100.0
    ref_window = (0.0, 2.0)
    active_window = (4.0, 6.0)
    band = (1.0, 30.0)
    result = rangayyan_feature_extract_bci(eeg, fs, ref_window, active_window, band)
    assert isinstance(result, dict)
    assert result


def test_rgfeatex_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    eeg = rng.normal(0, 1, 512)
    fs = 100.0
    ref_window = (0.0, 0.5)
    active_window = (1.0, 1.5)
    band = (4.0, 13.0)
    result = rangayyan_feature_extract_bci(eeg, fs, ref_window, active_window, band)
    assert isinstance(result, dict)
    assert result
