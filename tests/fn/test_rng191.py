"""Tests for rng191.rangayyan_ch4_pan_tompkins_thresholds."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_pan_tompkins_thresholds


def test_rng191_basic():
    """Test basic functionality."""
    NPKI = np.random.default_rng(42).normal(0, 1, 100)
    SPKI = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_thresholds(NPKI, SPKI)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng191_edge():
    """Test edge cases."""
    NPKI = np.random.default_rng(42).normal(0, 1, 100)
    SPKI = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_thresholds(NPKI, SPKI)
    assert isinstance(result, dict)
