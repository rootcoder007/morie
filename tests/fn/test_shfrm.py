"""Tests for shfrm.shared_frailty_marginal (shared gamma frailty by EM)."""

import math

import pytest

from morie.fn.shfrm import shared_frailty_marginal


def _data():
    """10 clusters of 4, cluster frailty exp(1.6 sin(1.3 k)), Weibull-free
    exponential times with administrative censoring."""
    T, E, X, C = [], [], [], []
    for k in range(10):
        w = math.exp(1.6 * math.sin(1.3 * k))
        for j in range(4):
            x = math.cos(0.7 * (4 * k + j)) + 0.2 * j
            u = ((37 * (4 * k + j) + 11) % 97 + 0.5) / 97.0
            t = -math.log(u) / (0.1 * w * math.exp(0.8 * x))
            c = 12.0 + (k % 5)
            T.append(min(t, c))
            E.append(1.0 if t <= c else 0.0)
            X.append([x])
            C.append(k)
    return T, E, X, C


def test_shfrm_basic():
    """At a fixed theta the EM solution is the penalised-likelihood one.
    survival::coxph(Surv(t, e) ~ x + frailty(c, "gamma", theta = 0.5),
    ties = "breslow", eps = 1e-12) gives beta = 0.86651524533273; the
    penalised fit stops on the change in its objective, within ~2e-7 of
    the optimum, hence rel 1e-6."""
    T, E, X, C = _data()
    r = shared_frailty_marginal(T, E, X, C, theta=0.5)
    assert float(r["estimate"][0]) == pytest.approx(0.86651524533273, rel=1e-6)
    assert r["kendall_tau"] == pytest.approx(0.5 / 2.5, rel=1e-15)


def test_shfrm_edge():
    """theta estimated by the marginal likelihood: coxph with a free gamma
    frailty (eps 1e-10) gives beta 0.80962645306458 and theta
    0.32204877785429 -- the likelihood is flat to ~1e-7 in theta, so
    rel 1e-6; singleton clusters cannot identify theta and are refused."""
    T, E, X, C = _data()
    r = shared_frailty_marginal(T, E, X, C)
    assert float(r["estimate"][0]) == pytest.approx(0.80962645306458, rel=1e-6)
    assert r["theta"] == pytest.approx(0.32204877785429, rel=1e-6)
    with pytest.raises(ValueError):
        shared_frailty_marginal(T[:5], E[:5], X[:5], list(range(5)))
