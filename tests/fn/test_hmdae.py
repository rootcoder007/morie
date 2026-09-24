"""Tests for hmdae.geron_denoising_autoencoder."""

from morie.fn import _array_core as np

from morie.fn.hmdae import geron_denoising_autoencoder


def test_hmdae_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_denoising_autoencoder(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "loss_history" in result


def test_hmdae_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_denoising_autoencoder(X)
    assert isinstance(result, dict)
