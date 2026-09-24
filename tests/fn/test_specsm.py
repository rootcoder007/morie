"""Tests for specsm.spectral_smoothed."""

from morie.fn import _array_core as np

from morie.fn.specsm import spectral_smoothed


def test_specsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = spectral_smoothed(y)
    assert isinstance(result, dict)
    assert "omega" in result


def test_specsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = spectral_smoothed(y)
    assert isinstance(result, dict)
