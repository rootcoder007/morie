"""Tests for rgblwand.rangayyan_baseline_wander."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_baseline_wander


def test_rgblwand_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.5
    result = rangayyan_baseline_wander(ecg, fs)
    assert isinstance(result, dict)
    assert "ecg_detrended" in result


def test_rgblwand_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.5
    result = rangayyan_baseline_wander(ecg, fs)
    assert isinstance(result, dict)
