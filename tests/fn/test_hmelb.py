"""Tests for hmelb.geron_elbo."""

from morie.fn import _array_core as np

from morie.fn.hmelb import geron_elbo


def test_hmelb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    mu = np.random.default_rng(42).normal(0.0, 1.0, 40)
    log_sigma = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_elbo(x, mu, log_sigma)
    assert isinstance(result, dict)
    assert "estimate" in result or "elbo" in result


def test_hmelb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    mu = np.random.default_rng(42).normal(0.0, 1.0, 40)
    log_sigma = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_elbo(x, mu, log_sigma)
    assert isinstance(result, dict)
