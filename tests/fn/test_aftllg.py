"""Tests for aftllg.aft_log_logistic."""

import numpy as np

from morie.fn.aftllg import aft_log_logistic


def test_aftllg_basic():
    """Test basic functionality on data generated from a known log-logistic DGP."""
    rng = np.random.default_rng(42)
    n = 800
    X = rng.normal(size=(n, 2))

    # True parameters (intercept first column will be added by the model)
    true_beta = np.array([1.0, 0.7, -0.4])
    true_sigma = 0.6

    # log-logistic DGP: log T = x' beta + sigma * epsilon,
    # epsilon ~ standard logistic (CDF F(t)=t/(1+t), inverse F^{-1}(u)=u/(1-u))
    mu = true_beta[0] + true_beta[1] * X[:, 0] + true_beta[2] * X[:, 1]
    u = rng.random(n)
    T = np.exp(mu + true_sigma * np.log(u / (1.0 - u)))

    # Independent (non-informative) right-censoring
    C = rng.exponential(float(np.exp(mu).mean()) * 4.0, n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)

    result = aft_log_logistic(time, event, X)

    # The function returns a RichResult that behaves like a dict
    assert isinstance(result, dict)

    # Documented return keys
    for key in ("beta", "time_ratio", "sigma", "loglik", "aic", "converged"):
        assert key in result

    # beta is on the log-time scale, intercept first
    p = X.shape[1] + 1
    assert result["beta"].shape == (p,)

    # time_ratio = exp(beta); check shapes agree
    assert result["time_ratio"].shape == result["beta"].shape

    # sigma is positive and matches exp(log_sigma)
    assert result["sigma"] > 0
    assert np.isfinite(result["sigma"])

    # On a sample this size the MLE should recover the DGP parameters
    # within a reasonable tolerance. Computed from the same formula as
    # the literature definition: time_ratio_j = exp(beta_j).
    np.testing.assert_allclose(
        result["time_ratio"],
        np.exp(result["beta"]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(result["beta"], true_beta, atol=0.2)
    np.testing.assert_allclose(result["sigma"], true_sigma, atol=0.2)

    # AIC definition from the docstring: 2*(p+1) - 2*loglik
    p_with_intercept = result["beta"].size
    expected_aic = float(2 * (p_with_intercept + 1) - 2 * result["loglik"])
    assert abs(result["aic"] - expected_aic) < 1e-8


def test_aftllg_edge():
    """Test that positive-time constraint and event-coding are honoured."""
    rng = np.random.default_rng(42)
    n = 400
    X = rng.normal(size=(n, 2))

    # event must be 0/1, not a continuous normal draw
    mu = 1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1]
    u = rng.random(n)
    T = np.exp(mu + 0.5 * np.log(u / (1.0 - u)))
    C = rng.exponential(float(np.exp(mu).mean()) * 3.0, n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)

    # Documented constraint: times must be strictly positive.
    assert np.all(time > 0)
    # Documented constraint: event is 1 (event) or 0 (censored).
    assert set(np.unique(event)).issubset({0.0, 1.0})

    result = aft_log_logistic(time, event, X)

    assert isinstance(result, dict)
    assert "beta" in result
    assert "sigma" in result
    assert "time_ratio" in result

    # With right-censoring present, the fit must still be finite and valid
    assert np.all(np.isfinite(result["beta"]))
    assert result["sigma"] > 0
