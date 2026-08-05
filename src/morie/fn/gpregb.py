# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Gaussian process regression with the hyperparameters integrated out.

Rasmussen and Williams (2006), *Gaussian Processes for Machine
Learning*, MIT Press, equations (2.23)-(2.26) for the posterior and
(5.8) for the marginal likelihood; Murray and Adams (2010),
"Slice sampling covariance hyperparameters of latent Gaussian
models", NIPS 23, arXiv:1006.0868, for the idea of averaging the
predictive distribution over the hyperparameter posterior rather than
fixing it at a point estimate.

Conditional on (lengthscale, noise) the predictive mean and variance
are

    mu*  = k*' (K + s2n I)^{-1} y,
    var* = k** - k*' (K + s2n I)^{-1} k*,

and the log marginal likelihood is

    -0.5 y'(K + s2n I)^{-1} y - 0.5 log|K + s2n I| - (n/2) log 2 pi.

The hyperparameter average is taken here by DETERMINISTIC quadrature
over a log-spaced grid weighted by that marginal likelihood, not by
slice sampling: a fixed grid gives both language arms the same
numbers, and with a one-point grid it collapses to the plain
point-estimate GP, which is the identity the tests check.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_regression_bayes"]


def _sqd(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s


def _k(A, B, ell, var):
    return [[var * math.exp(-0.5 * _sqd(A[i], B[j]) / (ell * ell)) for j in range(len(B))] for i in range(len(A))]


def _fit(A, y, Xs, ell, var, s2n):
    n = len(A)
    K = _k(A, A, ell, var)
    for i in range(n):
        K[i][i] += s2n
    alpha = core.cholsolve(K, y)
    L = core.chol(K)
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    ll = -0.5 * sum(y[i] * alpha[i] for i in range(n)) - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)
    Ks = _k(Xs, A, ell, var)
    mu = [sum(Ks[j][i] * alpha[i] for i in range(n)) for j in range(len(Xs))]
    sd = []
    for j in range(len(Xs)):
        v = core.cholsolve(K, Ks[j])
        q = sum(Ks[j][i] * v[i] for i in range(n))
        sd.append(max(var - q, 0.0))
    return mu, sd, ll


def gp_regression_bayes(X, y, kernel=None, X_test=None, lengthscales=None, noises=None, variance=1.0):
    """Predictive mean and variance averaged over a hyperparameter grid."""
    A = core.mat(X)
    yv = core.vec(y)
    n = len(A)
    if n == 0:
        raise ValueError("gp_regression_bayes: X is empty")
    if len(yv) != n:
        raise ValueError("gp_regression_bayes: X and y have different lengths")
    Xs = A if X_test is None else core.mat(X_test)
    ells = [0.5, 1.0, 2.0] if lengthscales is None else core.vec(lengthscales)
    s2s = [0.01, 0.1] if noises is None else core.vec(noises)
    var = float(variance)
    if var <= 0:
        raise ValueError("gp_regression_bayes: variance must be positive")
    for v in list(ells) + list(s2s):
        if v <= 0:
            raise ValueError("gp_regression_bayes: hyperparameters must be positive")
    mus = []
    sds = []
    lls = []
    for e in ells:
        for s in s2s:
            mu, sd, ll = _fit(A, yv, Xs, e, var, s)
            mus.append(mu)
            sds.append(sd)
            lls.append(ll)
    mx = max(lls)
    w = [math.exp(v - mx) for v in lls]
    tot = sum(w)
    w = [v / tot for v in w]
    m = len(Xs)
    mean = [sum(w[g] * mus[g][j] for g in range(len(w))) for j in range(m)]
    # law of total variance over the hyperparameter posterior
    varp = [sum(w[g] * (sds[g][j] + mus[g][j] ** 2) for g in range(len(w))) - mean[j] ** 2 for j in range(m)]
    return RichResult(
        title="GP regression, hyperparameters marginalised",
        summary_lines=[("n", n), ("grid", len(w))],
        payload={
            "estimate": mean[0],
            "mean": mean,
            "variance": varp,
            "weights": w,
            "loglik": lls,
            "n": n,
            "method": "GP posterior (R&W eqs. 2.23-2.26) averaged over a marginal-likelihood-weighted hyperparameter grid; Murray & Adams (2010)",
        },
    )


def cheatsheet():
    return "gpregb: GP regression with hyperparameters marginalised"
