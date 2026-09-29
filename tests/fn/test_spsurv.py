"""Tests for spsurv: Weibull gamma-frailty survival and the SPDE precision."""

import math

from morie.fn.spsurv import _digamma, spde_precision_grid, weibull_frailty_fit

IDX = range(60)
X1 = [math.sin(i * 0.7) for i in IDX]
CL = [i // 6 for i in IDX]
T = [math.exp(1 + 0.5 * X1[i] + 0.4 * math.cos(CL[i] * 1.7) + 0.6 * math.sin(i * 2.3)) for i in IDX]
EV = [1 if i % 5 else 0 for i in IDX]


def _loglik(lam, rho, beta, th=None):
    Hi = [lam * T[i] ** rho * math.exp(beta * X1[i]) for i in IDX]
    f = sum(EV[i] * (math.log(lam * rho) + (rho - 1) * math.log(T[i]) + beta * X1[i]) for i in IDX)
    if th is None:
        return f - sum(Hi)
    a = 1 / th
    for g in set(CL):
        idx = [i for i in IDX if CL[i] == g]
        D = sum(EV[i] for i in idx)
        H = sum(Hi[i] for i in idx)
        f += math.lgamma(a + D) - math.lgamma(a) + D * math.log(th) - (a + D) * math.log(1 + th * H)
    return f


def test_digamma():
    assert abs(_digamma(1.0) + 0.5772156649015329) <= 1e-13
    assert abs(_digamma(0.5) - (-0.5772156649015329 - 2 * math.log(2))) <= 1e-13


def test_weibull_maximises_the_likelihood():
    r = weibull_frailty_fit(T, EV, [[v] for v in X1])
    assert abs(r.loglik - _loglik(r.lambda_, r.rho, r.beta[0])) <= 1e-9
    for d in (1e-4, -1e-4):
        assert _loglik(r.lambda_ * (1 + d), r.rho, r.beta[0]) <= r.loglik
        assert _loglik(r.lambda_, r.rho * (1 + d), r.beta[0]) <= r.loglik
        assert _loglik(r.lambda_, r.rho, r.beta[0] + d) <= r.loglik


def test_gamma_frailty_marginal_likelihood():
    r = weibull_frailty_fit(T, EV, [[v] for v in X1], CL)
    assert abs(r.loglik - _loglik(r.lambda_, r.rho, r.beta[0], r.theta)) <= 1e-9
    for d in (1e-3, -1e-3):
        assert _loglik(r.lambda_, r.rho, r.beta[0], r.theta * (1 + d)) <= r.loglik


def test_spde_precision_structure():
    q = spde_precision_grid(5, 4, 0.8, 1.2, h=0.5).Q
    n = 20
    assert all(abs(q[a][b] - q[b][a]) <= 1e-15 for a in range(n) for b in range(n))
    # every row sums to tau^2 kappa^4 h^2 (G and G G have zero row sums)
    for a in range(n):
        assert abs(sum(q[a]) - 1.2**2 * 0.8**4 * 0.25) <= 1e-12
