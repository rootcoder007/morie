"""Tests for rng190.rangayyan_ch4_pan_tompkins_peak_classification."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_ch4_pan_tompkins_peak_classification


def test_rng190_basic():
    """Test basic functionality."""
    PEAKI = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = rangayyan_ch4_pan_tompkins_peak_classification(PEAKI)
    assert isinstance(result, dict)
    assert "SPKI" in result


def test_rng190_edge():
    """Test edge cases."""
    PEAKI = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = rangayyan_ch4_pan_tompkins_peak_classification(PEAKI)
    assert isinstance(result, dict)
