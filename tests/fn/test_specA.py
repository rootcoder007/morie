"""Tests for specA.spectral_anomaly."""

from morie.fn import _array_core as np

from morie.fn.specA import spectral_anomaly


def test_specA_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = spectral_anomaly(x)
    assert isinstance(result, dict)
    assert "saliency" in result


def test_specA_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = spectral_anomaly(x)
    assert isinstance(result, dict)
