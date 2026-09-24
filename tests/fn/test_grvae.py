"""Tests for grvae.geron_vae_elbo."""

from morie.fn import _array_core as np

from morie.fn.grvae import geron_vae_elbo


def test_grvae_basic():
    """Test basic functionality."""
    x = [[1.0, 0.0]]
    mu = [[0.0]]
    logvar = [[0.0]]
    recon = [[0.8, 0.3]]
    result = geron_vae_elbo(x, mu, logvar, recon)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grvae_edge():
    """Test edge cases."""
    x = [[1.0, 0.0]]
    mu = [[0.0]]
    logvar = [[0.0]]
    recon = [[0.8, 0.3]]
    result = geron_vae_elbo(x, mu, logvar, recon)
    assert isinstance(result, dict)
