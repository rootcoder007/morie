"""Tests for grvae.geron_vae_elbo."""

from morie.fn import _array_core as np

from morie.fn.grvae import geron_vae_elbo


def test_grvae_basic():
    """Test basic functionality."""
    x = [[0.0, 0.0]]
    mu = np.array([[1.0, -0.5]])
    logvar = np.array([[0.0, 0.3]])
    recon = [[0.0, 0.0]]
    result = geron_vae_elbo(x, mu, logvar, recon)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grvae_edge():
    """Test edge cases."""
    x = [[0.0, 0.0]]
    mu = np.array([[1.0, -0.5]])
    logvar = np.array([[0.0, 0.3]])
    recon = [[0.0, 0.0]]
    result = geron_vae_elbo(x, mu, logvar, recon)
    assert isinstance(result, dict)
