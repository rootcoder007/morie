"""Tests for vae_an.vae_anomaly."""

from morie.fn import _array_core as np

from morie.fn.vae_an import vae_anomaly


def test_vae_an_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vae_anomaly(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vae_an_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vae_anomaly(X)
    assert isinstance(result, dict)
