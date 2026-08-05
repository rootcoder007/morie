# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Spectral mixture kernel GP.

Wilson and Adams (2013), "Gaussian process kernels for pattern
discovery and extrapolation", ICML 2013 (PMLR 28(3):1067-1075),
arXiv:1302.4245.  Modelling the spectral density as a mixture of
Gaussians and inverting Bochner's theorem gives, in the paper's own
notation (equation on p. 3 of the arXiv version),

    k(tau) = sum_q w_q exp(-2 pi^2 tau^2 v_q) cos(2 pi tau mu_q).

The stub this module replaces printed the exponent as exp(-tau^2 v_q),
dropping the 2 pi^2; the rendered PDF was checked and the factor is
there.  It matters: without it the mapping to the squared exponential
(mu = 0 gives an SE kernel with lengthscale 1/(2 pi sqrt(v))) does not
hold, and that mapping is the anchor the tests use.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_spectral_mixture"]

_TWOPI2 = 2.0 * math.pi * math.pi


def _sm(t, w, v, mu):
    s = 0.0
    for q in range(len(w)):
        s += w[q] * math.exp(-_TWOPI2 * t * t * v[q]) * math.cos(2.0 * math.pi * t * mu[q])
    return s


def gp_spectral_mixture(X, y, X_test=None, Q=1, weights=None, variances=None, means=None, noise=0.01):
    """GP regression under a spectral mixture kernel on a one-dimensional input."""
    xs = core.vec(X)
    yv = core.vec(y)
    n = len(xs)
    if n == 0:
        raise ValueError("gp_spectral_mixture: X is empty")
    if len(yv) != n:
        raise ValueError("gp_spectral_mixture: X and y have different lengths")
    q = int(Q)
    if q < 1:
        raise ValueError("gp_spectral_mixture: Q must be at least 1")
    w = [1.0 / q] * q if weights is None else core.vec(weights)
    v = [1.0 / (2.0 * math.pi) ** 2] * q if variances is None else core.vec(variances)
    m = [0.1 * (k + 1) for k in range(q)] if means is None else core.vec(means)
    if not (len(w) == len(v) == len(m) == q):
        raise ValueError("gp_spectral_mixture: weights, variances and means must each have Q entries")
    for a in list(w) + list(v):
        if a <= 0:
            raise ValueError("gp_spectral_mixture: weights and variances must be positive")
    s2 = float(noise)
    if s2 < 0:
        raise ValueError("gp_spectral_mixture: noise must be non-negative")
    xt = xs if X_test is None else core.vec(X_test)
    K = [[_sm(xs[i] - xs[j], w, v, m) + (s2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    alpha = core.cholsolve(K, yv)
    Ks = [[_sm(xt[j] - xs[i], w, v, m) for i in range(n)] for j in range(len(xt))]
    mu = [sum(Ks[j][i] * alpha[i] for i in range(n)) for j in range(len(xt))]
    k0 = _sm(0.0, w, v, m)
    sd = []
    for j in range(len(xt)):
        z = core.cholsolve(K, Ks[j])
        sd.append(max(k0 - sum(Ks[j][i] * z[i] for i in range(n)), 0.0))
    L = core.chol(K)
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    ll = -0.5 * sum(yv[i] * alpha[i] for i in range(n)) - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)
    return RichResult(
        title="Spectral mixture GP",
        summary_lines=[("n", n), ("Q", q)],
        payload={
            "estimate": mu[0],
            "mean": mu,
            "variance": sd,
            "k_zero": k0,
            "loglik": ll,
            "n": n,
            "method": "k(tau) = sum_q w_q exp(-2 pi^2 tau^2 v_q) cos(2 pi tau mu_q), Wilson & Adams (2013)",
        },
    )


def cheatsheet():
    return "gpsps: spectral mixture kernel GP"
