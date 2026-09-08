# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Gaussian process classification by the Laplace approximation.

Williams and Barber (1998), "Bayesian classification with Gaussian
processes", IEEE Trans. Pattern Analysis and Machine Intelligence
20(12):1342-1351, doi:10.1109/34.735807; Rasmussen and Williams
(2006), *Gaussian Processes for Machine Learning*, MIT Press, chapter
3, Algorithm 3.1 and equation (3.82).

With the probit likelihood p(y = 1 | f) = Phi(f), Newton's method on

    Psi(f) = log p(y | f) - 0.5 f' K^{-1} f

gives the posterior mode f-hat and W = -grad^2 log p(y | f); the
averaged predictive probability at a test point is

    p* = Phi( mu* / sqrt(1 + var*) ).

The mode is where the gradient vanishes, and the tests check that
directly by differentiating Psi numerically rather than trusting the
iteration.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_classification"]

_SQ2PI = math.sqrt(2.0 * math.pi)


def _sqd(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s


def _k(A, B, ell, var):
    return [[var * math.exp(-0.5 * _sqd(A[i], B[j]) / (ell * ell)) for j in range(len(B))] for i in range(len(A))]


def _dphi(z):
    return math.exp(-0.5 * z * z) / _SQ2PI


def _grad_hess(yv, f):
    """First and second derivatives of the probit log-likelihood."""
    g = []
    w = []
    for i in range(len(f)):
        s = 1.0 if yv[i] == 1 else -1.0
        z = s * f[i]
        P = core.pnorm(z)
        if P < 1e-300:
            P = 1e-300
        r = _dphi(z) / P
        g.append(s * r)
        w.append(r * r + z * r)
    return g, w


def gp_classification(X, y, X_test=None, kernel=None, lengthscale=1.0, variance=1.0, iters=40):
    """Laplace-approximate GP classifier with the probit likelihood."""
    A = core.mat(X)
    n = len(A)
    if n == 0:
        raise ValueError("gp_classification: X is empty")
    yv = [int(v) for v in core.vec(y)]
    if len(yv) != n:
        raise ValueError("gp_classification: X and y have different lengths")
    for v in yv:
        if v not in (0, 1):
            raise ValueError("gp_classification: labels must be 0 or 1")
    ell = float(lengthscale)
    var = float(variance)
    if ell <= 0 or var <= 0:
        raise ValueError("gp_classification: lengthscale and variance must be positive")
    Xs = A if X_test is None else core.mat(X_test)
    K = _k(A, A, ell, var)
    f = [0.0] * n
    obj = []
    for _ in range(int(iters)):
        g, w = _grad_hess(yv, f)
        B = [[K[i][j] * w[j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
        rhs = [sum(K[i][j] * (w[j] * f[j] + g[j]) for j in range(n)) for i in range(n)]
        f = [v for v in np.linalg.solve(B, rhs)]
        ll = 0.0
        for i in range(n):
            s = 1.0 if yv[i] == 1 else -1.0
            ll += math.log(max(core.pnorm(s * f[i]), 1e-300))
        a = core.cholsolve(K, f)
        obj.append(ll - 0.5 * sum(f[i] * a[i] for i in range(n)))
    g, w = _grad_hess(yv, f)
    alpha = core.cholsolve(K, f)
    Ks = _k(Xs, A, ell, var)
    mu = [sum(Ks[j][i] * alpha[i] for i in range(n)) for j in range(len(Xs))]
    M = [[K[i][j] + (1.0 / w[i] if i == j and w[i] > 0 else (1e12 if i == j else 0.0)) for j in range(n)] for i in range(n)]
    sd = []
    for j in range(len(Xs)):
        v = core.cholsolve(M, Ks[j])
        q = sum(Ks[j][i] * v[i] for i in range(n))
        sd.append(max(var - q, 0.0))
    p = [core.pnorm(mu[j] / math.sqrt(1.0 + sd[j])) for j in range(len(Xs))]
    pred = [1 if v >= 0.5 else 0 for v in p]
    return RichResult(
        title="GP classification (Laplace)",
        summary_lines=[("n", n), ("test", len(Xs))],
        payload={
            "estimate": p[0],
            "p": p,
            "predicted": pred,
            "f_mode": f,
            "latent_mean": mu,
            "latent_var": sd,
            "objective": obj,
            "n": n,
            "method": "Newton mode of Psi(f) with probit likelihood; averaged prediction R&W eq. (3.82)",
        },
    )


def cheatsheet():
    return "gpcla: GP classification by Laplace approximation"
