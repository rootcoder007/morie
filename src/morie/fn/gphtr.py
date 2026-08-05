# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Heteroscedastic Gaussian process regression.

Goldberg, Williams and Bishop (1998), "Regression with input-dependent
noise: a Gaussian process treatment", NIPS 10 (1997 proceedings) --
title verified against the NIPS record.  Two processes are used: one
for the mean, one for the LOG noise variance, so the noise level is
itself a smooth function of the input and cannot go negative:

    y(x) = f(x) + eps(x),   f ~ GP,   log var(eps(x)) ~ GP.

The alternation used here is the deterministic analogue of the
paper's sampler: fit the mean GP under the current noise, form the
log squared residuals, smooth them with the second GP, exponentiate,
refit.  Zero iterations leaves the plain homoscedastic GP exactly,
which is the identity the tests check; the noise process is what makes
the predictive band widen where the data are noisy.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_heteroscedastic"]


def _k(A, B, ell, var):
    out = []
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            s = 0.0
            for c in range(len(A[i])):
                d = A[i][c] - B[j][c]
                s += d * d
            row.append(var * math.exp(-0.5 * s / (ell * ell)))
        out.append(row)
    return out


def _post(K, noise, y, Ks, n):
    M = [[K[i][j] + (noise[i] if i == j else 0.0) for j in range(n)] for i in range(n)]
    al = core.cholsolve(M, y)
    mu = [sum(Ks[j][i] * al[i] for i in range(n)) for j in range(len(Ks))]
    return M, al, mu


def gp_heteroscedastic(X, y, X_test=None, lengthscale=1.0, variance=1.0,
                       noise0=0.1, noise_lengthscale=1.0, iters=3, floor=1e-6):
    """Mean and input-dependent noise, by alternating two GPs."""
    A = core.mat(X)
    n = len(A)
    if n == 0:
        raise ValueError("gp_heteroscedastic: X is empty")
    yv = core.vec(y)
    if len(yv) != n:
        raise ValueError("gp_heteroscedastic: X and y have different lengths")
    ell = float(lengthscale)
    var = float(variance)
    if ell <= 0 or var <= 0:
        raise ValueError("gp_heteroscedastic: lengthscale and variance must be positive")
    s0 = float(noise0)
    if s0 <= 0:
        raise ValueError("gp_heteroscedastic: noise0 must be positive")
    if int(iters) < 0:
        raise ValueError("gp_heteroscedastic: iters must be non-negative")
    Xs = A if X_test is None else core.mat(X_test)
    K = _k(A, A, ell, var)
    Ks = _k(Xs, A, ell, var)
    Kn = _k(A, A, float(noise_lengthscale), 1.0)
    Kns = _k(Xs, A, float(noise_lengthscale), 1.0)
    noise = [s0] * n
    for _ in range(int(iters)):
        M, al, _mu = _post(K, noise, yv, Ks, n)
        fit = [sum(K[i][j] * al[j] for j in range(n)) for i in range(n)]
        z = [math.log(max((yv[i] - fit[i]) ** 2, float(floor))) for i in range(n)]
        zbar = sum(z) / n
        Mn = [[Kn[i][j] + (0.25 if i == j else 0.0) for j in range(n)] for i in range(n)]
        an = core.cholsolve(Mn, [z[i] - zbar for i in range(n)])
        noise = [math.exp(zbar + sum(Kn[i][j] * an[j] for j in range(n))) for i in range(n)]
    M, al, mu = _post(K, noise, yv, Ks, n)
    sd = []
    for j in range(len(Xs)):
        v = core.cholsolve(M, Ks[j])
        sd.append(max(var - sum(Ks[j][i] * v[i] for i in range(n)), 0.0))
    if int(iters) > 0:
        zb = sum(math.log(max(v, float(floor))) for v in noise) / n
        an2 = core.cholsolve([[Kn[i][j] + (0.25 if i == j else 0.0) for j in range(n)] for i in range(n)],
                             [math.log(noise[i]) - zb for i in range(n)])
        noise_test = [math.exp(zb + sum(Kns[j][i] * an2[i] for i in range(n))) for j in range(len(Xs))]
    else:
        noise_test = [s0] * len(Xs)
    L = core.chol(M)
    ll = -0.5 * sum(yv[i] * al[i] for i in range(n)) - sum(math.log(L[i][i]) for i in range(n)) - 0.5 * n * math.log(2.0 * math.pi)
    return RichResult(
        title="Heteroscedastic GP",
        summary_lines=[("n", n), ("iterations", int(iters))],
        payload={
            "estimate": mu[0],
            "mean": mu,
            "variance": sd,
            "noise": noise,
            "noise_test": noise_test,
            "loglik": ll,
            "n": n,
            "method": "alternating mean GP and log-variance GP, Goldberg, Williams & Bishop (1998)",
        },
    )


def cheatsheet():
    return "gphtr: heteroscedastic Gaussian process"
