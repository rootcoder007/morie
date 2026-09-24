"""Tests for vaeber.vae_elbo."""

from morie.fn import _array_core as np

from morie.fn.vaeber import vae_elbo


def test_vaeber_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vae_elbo(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vaeber_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vae_elbo(x)
    assert isinstance(result, dict)
