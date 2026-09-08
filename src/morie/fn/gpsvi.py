# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Sparse variational GP regression.

Hensman, Fusi and Lawrence (2013), "Gaussian processes for big data",
UAI 2013, arXiv:1309.6835, with the collapsed bound of Titsias (2009),
AISTATS 2009 (PMLR 5:567-574).  For a Gaussian likelihood the optimal
q(u) is available in closed form, so the stochastic gradient loop of
the paper is unnecessary here and the bound can be evaluated exactly:

    Q  = K_nm K_mm^{-1} K_mn,
    L  = log N(y; 0, Q + s2 I) - (1 / 2 s2) tr(K_nn - Q),

the trace term being the price paid for the low-rank approximation.
When the inducing inputs ARE the training inputs the trace term is
zero and both the bound and the predictive mean coincide exactly with
full GP regression -- that identity is the anchor, and it is the one
thing a sparse approximation must satisfy.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_stochastic_vi"]


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


def gp_stochastic_vi(X, y, X_test=None, inducing=None, batch_size=None, lengthscale=1.0, variance=1.0, noise=0.1, jitter=1e-9):
    """Titsias/Hensman sparse GP: collapsed bound and predictive moments."""
    A = core.mat(X)
    yv = core.vec(y)
    n = len(A)
    if n == 0:
        raise ValueError("gp_stochastic_vi: X is empty")
    if len(yv) != n:
        raise ValueError("gp_stochastic_vi: X and y have different lengths")
    Z = A if inducing is None else core.mat(inducing)
    m = len(Z)
    if m == 0:
        raise ValueError("gp_stochastic_vi: no inducing inputs")
    if len(Z[0]) != len(A[0]):
        raise ValueError("gp_stochastic_vi: inducing inputs have the wrong dimension")
    Xs = A if X_test is None else core.mat(X_test)
    ell = float(lengthscale)
    var = float(variance)
    s2 = float(noise)
    if ell <= 0 or var <= 0 or s2 <= 0:
        raise ValueError("gp_stochastic_vi: lengthscale, variance and noise must be positive")
    Kmm = _k(Z, Z, ell, var)
    for i in range(m):
        Kmm[i][i] += float(jitter)
    Knm = _k(A, Z, ell, var)
    Q = [[0.0] * n for _ in range(n)]
    Kinv_rows = [core.cholsolve(Kmm, Knm[i]) for i in range(n)]
    for i in range(n):
        for j in range(n):
            Q[i][j] = sum(Knm[i][k] * Kinv_rows[j][k] for k in range(m))
    trace = 0.0
    for i in range(n):
        trace += var - Q[i][i]
    S = [[Q[i][j] + (s2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    a = core.cholsolve(S, yv)
    L = core.chol(S)
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    gauss = -0.5 * sum(yv[i] * a[i] for i in range(n)) - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)
    elbo = gauss - trace / (2.0 * s2)
    Ksm = _k(Xs, Z, ell, var)
    mu = []
    sd = []
    for j in range(len(Xs)):
        row = [sum(Ksm[j][k] * Kinv_rows[i][k] for k in range(m)) for i in range(n)]
        mu.append(sum(row[i] * a[i] for i in range(n)))
        z = core.cholsolve(S, row)
        sd.append(max(var - sum(row[i] * z[i] for i in range(n)), 0.0))
    return RichResult(
        title="Sparse variational GP",
        summary_lines=[("n", n), ("inducing", m), ("elbo", elbo)],
        payload={
            "estimate": mu[0],
            "mean": mu,
            "variance": sd,
            "elbo": elbo,
            "gaussian_term": gauss,
            "trace_term": trace,
            "n": n,
            "method": "collapsed sparse bound of Titsias (2009) as used by Hensman, Fusi & Lawrence (2013)",
        },
    )


def cheatsheet():
    return "gpsvi: sparse variational GP regression"


# compact alias per ledger/NAMING.md
gpstochasticvi = gp_stochastic_vi
