"""Tests for mhmcmc.metropolis_hastings."""

import math

from morie.fn import _array_core as np

from morie.fn.mhmcmc import metropolis_hastings


def test_mhmcmc_basic():
    """Test basic functionality."""
    rng_u = np.random.default_rng(43)
    rng_z = np.random.default_rng(42)
    target = lambda x: math.exp(-0.5 * x * x)
    x0 = 0.0
    n_iter = 50
    u = rng_u.uniform(0.0, 1.0, n_iter)
    z = rng_z.normal(0.0, 1.0, n_iter)
    result = metropolis_hastings(target, x0, n_iter, u, z)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "sd" in result
    assert "accept_rate" in result
    assert "chain" in result
    assert "accepted" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == n_iter
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["sd"])
    assert math.isfinite(result["accept_rate"])
    assert 0.0 <= result["accept_rate"] <= 1.0


def test_mhmcmc_edge():
    """Test edge cases."""
    rng_u = np.random.default_rng(43)
    rng_z = np.random.default_rng(42)
    target = lambda x: math.exp(-0.5 * x * x)
    x0 = 1.0
    n_iter = 10
    u = rng_u.uniform(0.0, 1.0, n_iter)
    z = rng_z.normal(0.0, 1.0, n_iter)
    result = metropolis_hastings(target, x0, n_iter, u, z, burn=5)
    assert isinstance(result, dict)
    assert result["n"] == n_iter
    assert len(result["chain"]) == n_iter
    assert math.isfinite(result["estimate"])
