"""Tests for bicg.bayesian_information_criterion."""

from morie.fn import _array_core as np

from morie.fn.bicg import bayesian_information_criterion


def test_bicg_basic():
    """Test basic functionality."""
    log_lik = -150.0
    n_params = 3
    n_obs = 100
    result = bayesian_information_criterion(log_lik, n_params, n_obs)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    # Independent recomputation of the documented formula
    import math
    expected_bic = -2.0 * log_lik + n_params * math.log(n_obs)
    expected_aic = -2.0 * log_lik + 2.0 * n_params
    expected_penalty = n_params * math.log(n_obs)
    assert result["estimate"] == expected_bic
    assert result["aic"] == expected_aic
    assert result["penalty"] == expected_penalty
    assert result["log_lik"] == float(log_lik)
    assert result["n_params"] == float(n_params)
    assert result["n"] == float(n_obs)


def test_bicg_edge():
    """Test edge cases."""
    log_lik = -42.5
    n_params = 5
    n_obs = 1000
    result = bayesian_information_criterion(log_lik, n_params, n_obs)
    assert isinstance(result, dict)
    # n_obs <= 0 should yield a NaN penalty / BIC per the documented behaviour
    log_lik2 = -10.0
    n_params2 = 4
    n_obs2 = 0
    result_edge = bayesian_information_criterion(log_lik2, n_params2, n_obs2)
    import math
    assert math.isnan(result_edge["penalty"])
    assert math.isnan(result_edge["estimate"])
    # And the AIC branch should still be finite
    expected_aic_edge = -2.0 * log_lik2 + 2.0 * n_params2
    assert result_edge["aic"] == expected_aic_edge
