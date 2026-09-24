"""Tests for dpepsm.epsilon_dp."""

import math

from morie.fn import _array_core as np

from morie.fn.dpepsm import epsilon_dp


def test_dpepsm_basic():
    """Test basic functionality."""
    D = [0.0] * 10
    D_prime = [0.0] * 9 + [1.0]

    def mech(X, rng):
        return float(np.sum(X) + rng.normal(0, 1.0))

    result = epsilon_dp(mech, D, D_prime, n_samples=2000, bins=50, seed=42)
    assert isinstance(result, dict)
    assert "epsilon_empirical" in result
    assert "max_log_ratio" in result
    assert "n_usable_bins" in result
    assert "n_excluded_bins" in result
    assert math.isfinite(float(result["epsilon_empirical"]))


def test_dpepsm_edge():
    """Test edge cases."""
    D = [0.0] * 10
    D_prime = [0.0] * 9 + [1.0]

    def mech(X, rng):
        # Use a noise level that yields a finite empirical epsilon.
        return float(np.sum(X) + rng.normal(0, 1.0))

    # Edge case: small number of bins.
    result = epsilon_dp(mech, D, D_prime, n_samples=2000, bins=10, seed=0)
    assert isinstance(result, dict)
    assert "epsilon_empirical" in result
    assert "max_log_ratio" in result
    assert "n_usable_bins" in result
    assert "n_excluded_bins" in result
    assert math.isfinite(float(result["epsilon_empirical"]))
