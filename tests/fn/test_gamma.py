"""Tests for morie.fn.gamma — Gamma GLM."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gamma import gamma_glm


def test_gamma_glm_positive_coef():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    mu = np.exp(1.0 + 0.5 * X[:, 0])
    # Sample y ~ Gamma(shape=k, scale=mu/k) pointwise: draw standard
    # exponentials and scale by mu/k, which avoids needing array-shaped
    # scale in rng.gamma (the shim's RNG only accepts scalar scale).
    k = 5.0
    e = rng.exponential(1.0, n)
    y = (mu / k) * e
    res = gamma_glm(y, X)
    assert res.coefficients["x0"] > 0
    # Independently check fitted values match the documented log-link
    # mean function mu = exp(beta0 + beta1*x) on the same inputs.
    beta0 = res.coefficients["(Intercept)"]
    beta1 = res.coefficients["x0"]
    mu_hat = np.exp(beta0 + beta1 * X[:, 0])
    assert np.all(mu_hat > 0)


def test_gamma_glm_negative_y_raises():
    with pytest.raises(ValueError, match="positive"):
        gamma_glm(np.array([-1.0, 2.0, 3.0]), np.ones((3, 1)))


def test_gamma_glm_fitted_positive():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    y = np.exp(0.5 + 0.3 * X[:, 0]) + rng.exponential(0.1, n)
    res = gamma_glm(y, X)
    assert np.all(res.fitted > 0)


def test_gamma_glm_deviance():
    rng = np.random.default_rng(42)
    n = 150
    X = rng.standard_normal((n, 1))
    # Same trick: generate Gamma variates pointwise as scale*Exp(1).
    k = 3.0
    mu = np.exp(0.5 * X[:, 0]) / k
    y = mu * rng.exponential(1.0, n)
    res = gamma_glm(y, X)
    assert res.extra["deviance"] > 0
    assert np.isfinite(res.extra["phi"])
    # Deviance is 2 * sum( -log(y/mu) + (y - mu)/mu ) with mu = exp(X@beta).
    beta = np.array([res.coefficients["(Intercept)"], res.coefficients["x0"]])
    Xfull = np.column_stack([np.ones(n), X])
    mu_f = np.exp(np.clip(Xfull @ beta, -20, 20))
    expected_deviance = 2.0 * float(np.sum(-np.log(y / mu_f) + (y - mu_f) / mu_f))
    assert abs(res.extra["deviance"] - expected_deviance) < 1e-6
