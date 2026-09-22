"""Tests for gh_c12_2.ghosal_dp_bvm."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_2 import ghosal_dp_bvm


def test_gh_c12_2_basic():
    """Test basic functionality."""
    n = 2000
    alpha = 2.0
    n_sim = 400
    seed = 42

    result = ghosal_dp_bvm(n=n, alpha=alpha, n_sim=n_sim, seed=seed)

    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # The function simulates the Bayesian posterior mean of a Bernoulli
    # proportion under a Beta(alpha, 1) prior with n samples, then compares
    # the empirical variance of sqrt(n) * (F_post - t) at t = 0.3 against
    # the Brownian-bridge variance t*(1-t) from GvdV (2017) sec. 12.2.
    t = 0.3
    expected_bridge_variance = t * (1.0 - t)
    assert result["bridge_variance"] == expected_bridge_variance

    # The estimate must be a non-negative finite scalar.
    est = float(result["estimate"])
    assert est >= 0.0


def test_gh_c12_2_edge():
    """Test edge cases."""
    result = ghosal_dp_bvm(n=1, alpha=2.0, n_sim=2, seed=42)

    # The result exposes 'estimate', not 'n'.
    assert "estimate" in result
    assert "bridge_variance" in result
    assert "gap" in result
    assert "method" in result
