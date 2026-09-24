"""Tests for hmvae.geron_vae."""

from morie.fn import _array_core as np

from morie.fn.hmvae import geron_vae


def test_hmvae_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_vae(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "mu" in result


def test_hmvae_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_vae(X)
    assert isinstance(result, dict)
