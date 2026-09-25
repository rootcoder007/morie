"""Tests for vinfer.variational_inference (mean-field Normal-Gamma VI)."""

import math

import pytest

from morie.fn.vinfer import variational_inference


X = [2.1, 1.4, 3.3, 2.8, 1.9, 2.5, 3.0, 1.2, 2.2, 2.7]


def test_vinfer_basic():
    """Improper limit (all hyperparameters 0): the fixed point is
    E[tau] = N / sum (x - xbar)^2 (Bishop 2006, eq. 10.33).  With a
    proper prior the result solves Bishop's 10.26-10.30 jointly,
    iterated here to convergence independently."""
    n, xb = len(X), sum(X) / len(X)
    ss = sum((v - xb) ** 2 for v in X)
    r = variational_inference(x=X)
    assert r["e_tau"] == pytest.approx(n / ss, rel=1e-10)
    assert r["mu_n"] == pytest.approx(xb, abs=1e-15)
    mu0, l0, a0, b0 = 1.0, 2.0, 3.0, 1.5
    mu = (l0 * mu0 + n * xb) / (l0 + n)
    a = a0 + (n + 1) / 2
    et = 1.0
    for _ in range(500):
        var = 1 / ((l0 + n) * et)
        b = b0 + 0.5 * (ss + n * (xb - mu) ** 2 + n * var + l0 * ((mu - mu0) ** 2 + var))
        et = a / b
    rp = variational_inference(x=X, mu0=mu0, lambda0=l0, a0=a0, b0=b0)
    assert rp["e_tau"] == pytest.approx(et, rel=1e-10)
    assert rp["mu_n"] == pytest.approx(mu, rel=1e-14)
    assert rp["a_n"] == pytest.approx(a, rel=1e-15)
    assert rp["var_mu"] == pytest.approx(1 / ((l0 + n) * et), rel=1e-10)
    assert rp["elbo_monotone"] == 1.0


def test_vinfer_edge():
    """Unknown model or family, one observation and negative
    hyperparameters raise."""
    with pytest.raises(ValueError):
        variational_inference(log_p="poisson", x=X)
    with pytest.raises(ValueError):
        variational_inference(q_family="fullrank", x=X)
    with pytest.raises(ValueError):
        variational_inference(x=[1.0])
    with pytest.raises(ValueError):
        variational_inference(x=X, b0=-1.0)


