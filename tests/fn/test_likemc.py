"""Tests for likemc.likelihood_mcmc_epi."""

import math

from morie.fn import _array_core as np
from morie.fn.likemc import likelihood_mcmc_epi


def test_likemc_basic():
    """Test basic functionality."""
    model = {
        "S0": 990.0,
        "I0": 10.0,
        "N": 1000.0,
        "dt": 1.0,
    }
    # Observed incidence counts over time (non-negative)
    data = [1, 3, 8, 15, 22, 17, 12, 8, 5, 3]
    priors = {
        "beta_mu": math.log(0.5),
        "beta_sigma": 1.0,
        "gamma_mu": math.log(0.2),
        "gamma_sigma": 1.0,
    }
    n_iter = 50
    result = likelihood_mcmc_epi(model, data, priors, n_iter)
    assert isinstance(result, dict)
    # Verify the keys named in the return statement are present
    for key in ("estimate", "beta_mean", "gamma_mean", "chain",
                "n_draws", "n_iter", "acceptance_rate", "R0_mean",
                "R0_q025", "R0_median", "R0_q975", "logpost_final",
                "seed", "step", "method"):
        assert key in result
    # Acceptance rate is a probability in [0, 1]
    assert 0.0 <= result["acceptance_rate"] <= 1.0
    # With burn=0, all draws are kept
    assert result["n_draws"] == n_iter
    assert len(result["chain"]) == n_iter
    # Posterior summary must be finite
    assert math.isfinite(result["logpost_final"])


def test_likemc_edge():
    """Test edge cases - minimal valid input."""
    model = {
        "S0": 990.0,
        "I0": 10.0,
        "N": 1000.0,
    }
    # Minimal data: exactly two observed counts
    data = [1.0, 2.0]
    priors = {
        "beta_mu": math.log(0.5),
        "beta_sigma": 1.0,
        "gamma_mu": math.log(0.2),
        "gamma_sigma": 1.0,
    }
    # Minimal n_iter: a single iteration
    n_iter = 1
    result = likelihood_mcmc_epi(model, data, priors, n_iter)
    assert isinstance(result, dict)
    assert result["n_iter"] == 1
    assert result["n_draws"] == 1
    assert len(result["chain"]) == 1
    assert math.isfinite(result["logpost_final"])
    assert 0.0 <= result["acceptance_rate"] <= 1.0
