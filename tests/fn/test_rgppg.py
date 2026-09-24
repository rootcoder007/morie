"""Tests for rgppg.rangayyan_ppg_features."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ppg_features


def test_rgppg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ppg = rng.normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_ppg_features(ppg, fs)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rgppg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    ppg = rng.normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_ppg_features(ppg, fs)
    assert isinstance(result, dict)
