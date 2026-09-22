"""Tests for gh_c8_3.ghosal_test_cond."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c8_3 import ghosal_test_cond


def test_gh_c8_3_basic():
    """Test basic functionality with a passing configuration."""
    prior_ball = 0.9
    log_entropy = 1.0
    sieve_mass = 0.05
    eps_bar = 0.2
    eps = 0.3
    n = 50
    Cconst = 1.5

    result = ghosal_test_cond(
        prior_ball, log_entropy, sieve_mass, eps_bar, eps, n, Cconst
    )

    # The function returns a dict-like RichResult
    assert isinstance(result, dict)

    # Required documented keys
    expected_keys = {
        "holds",
        "cond_prior",
        "cond_entropy",
        "cond_sieve",
        "slack_prior",
        "slack_entropy",
        "slack_sieve",
        "n_eps_bar_sq",
    }
    assert expected_keys.issubset(set(result.keys()))

    # Compute expected quantities independently from the documented formula
    neb = n * eps_bar * eps_bar
    expected_neb = neb
    expected_slack_prior = math.log(prior_ball) + Cconst * neb
    expected_slack_entropy = n * eps * eps - log_entropy
    expected_slack_sieve = -(Cconst + 4.0) * neb - math.log(sieve_mass)

    expected_cond_prior = 1.0 if expected_slack_prior >= 0.0 else 0.0
    expected_cond_entropy = 1.0 if expected_slack_entropy >= 0.0 else 0.0
    expected_cond_sieve = 1.0 if expected_slack_sieve >= 0.0 else 0.0
    expected_holds = (
        1.0 if (expected_cond_prior and expected_cond_entropy and expected_cond_sieve)
        else 0.0
    )

    assert result["n_eps_bar_sq"] == expected_neb
    assert result["slack_prior"] == expected_slack_prior
    assert result["slack_entropy"] == expected_slack_entropy
    assert result["slack_sieve"] == expected_slack_sieve
    assert result["cond_prior"] == expected_cond_prior
    assert result["cond_entropy"] == expected_cond_entropy
    assert result["cond_sieve"] == expected_cond_sieve
    assert result["holds"] == expected_holds


def test_gh_c8_3_edge():
    """Test edge cases: slack values and condition flags."""
    # Configuration with eps == eps_bar (boundary of allowed range)
    prior_ball = 0.5
    log_entropy = 0.0
    sieve_mass = 0.0  # -> cond_sieve must hold (s3 = +inf)
    eps_bar = 0.1
    eps = 0.1
    n = 10
    Cconst = 1.0

    result = ghosal_test_cond(
        prior_ball, log_entropy, sieve_mass, eps_bar, eps, n, Cconst
    )

    assert isinstance(result, dict)

    neb = n * eps_bar * eps_bar
    expected_slack_prior = math.log(prior_ball) + Cconst * neb
    expected_slack_entropy = n * eps * eps - log_entropy
    expected_slack_sieve = math.inf

    expected_cond_prior = 1.0 if expected_slack_prior >= 0.0 else 0.0
    expected_cond_entropy = 1.0 if expected_slack_entropy >= 0.0 else 0.0
    expected_cond_sieve = 1.0  # sieve slack is +inf when sieve_mass == 0

    assert result["n_eps_bar_sq"] == neb
    assert result["slack_prior"] == expected_slack_prior
    assert result["slack_entropy"] == expected_slack_entropy
    assert result["slack_sieve"] == expected_slack_sieve
    assert result["cond_prior"] == expected_cond_prior
    assert result["cond_entropy"] == expected_cond_entropy
    assert result["cond_sieve"] == expected_cond_sieve
