"""Tests for emaxr.em_step_random_effects."""

import math

from morie.fn import _array_core as np

from morie.fn.emaxr import em_step_random_effects


def test_emaxr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    N, p, J = 40, 3, 4
    y = rng.normal(0, 1, N)
    X = rng.normal(0, 1, (N, p))
    cluster = rng.integers(0, J, N)
    sigma2_u = 1.0
    sigma2_e = 1.0
    result = em_step_random_effects(y, X, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
    for key in ("estimate", "sigma2_u", "sigma2_e", "beta", "u_hat", "var_u", "J", "n", "method"):
        assert key in result
    assert math.isfinite(float(result["estimate"]))
    assert float(result["estimate"]) >= 0.0
    assert float(result["sigma2_e"]) >= 0.0


def test_emaxr_edge():
    """Test edge cases with small valid input."""
    rng = np.random.default_rng(42)
    N, p, J = 20, 2, 3
    y = rng.normal(0, 1, N)
    X = rng.normal(0, 1, (N, p))
    cluster = rng.integers(0, J, N)
    sigma2_u = 0.5
    sigma2_e = 0.5
    result = em_step_random_effects(y, X, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(float(result["estimate"]))
    assert float(result["estimate"]) >= 0.0
