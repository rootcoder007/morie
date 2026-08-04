# morie.fn -- slice s03 (rootcoder007/morie)
"""Sparse variational Gaussian process and its ELBO.

Source consulted: Titsias, M. K. (2009).  Variational learning of
inducing variables in sparse Gaussian processes.  *AISTATS* 5, 567-574.
His bound, equation (9), is

    log p(y) >= log N( y | 0, Q_nn + sigma^2 I )
                - (1 / (2 sigma^2)) tr( K_nn - Q_nn )

with Q_nn = K_nm K_mm^(-1) K_mn.  The first term is the DTC marginal
likelihood; the second, the trace of the Nystrom residual, is the
*penalty* that makes the bound a bound -- it is what stops the inducing
points being chosen to overfit, and it is the entire difference between
Titsias's variational method and the earlier projected-process
approximations.  The AISTATS volume is free but was not retrievable
here; the bound is quoted in its standard published form.

Both terms are returned separately so the penalty is visible.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["variational_gp"]


def _rbf(x, y, gamma):
    s = 0.0
    for a in range(len(x)):
        d = x[a] - y[a]
        s += d * d
    return math.exp(-gamma * s)


def variational_gp(X, y, Z=None, gamma=1.0, sigma2=1e-2, jitter=1e-8,
                   X_test=None):
    """Titsias's collapsed ELBO and the corresponding predictions.

    Returns
    -------
    RichResult with payload:
        estimate : the ELBO
        elbo, fit_term, trace_penalty
        pred, var : predictions at the test inputs
    """
    Xm = k.mat(X)
    yv = k.vec(y)
    Zm = k.mat(Z) if Z is not None else Xm
    g = float(gamma)
    n = len(Xm)
    m = len(Zm)
    Kmm = [[_rbf(Zm[i], Zm[j], g) for j in range(m)] for i in range(m)]
    for i in range(m):
        Kmm[i][i] += float(jitter)
    Knm = [[_rbf(Xm[i], Zm[j], g) for j in range(m)] for i in range(n)]
    Q = [[0.0] * n for _ in range(n)]
    trace = 0.0
    for i in range(n):
        w = k.cholsolve(Kmm, Knm[i])
        for j in range(n):
            s = 0.0
            for t in range(m):
                s += Knm[j][t] * w[t]
            Q[i][j] = s
        trace += 1.0 - Q[i][i]
    S = [[Q[i][j] + (float(sigma2) if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    L = k.chol(S)
    logdet = 0.0
    for i in range(n):
        logdet += 2.0 * math.log(L[i][i]) if L[i][i] > 0.0 else 0.0
    sol = k.cholsolve(S, yv)
    quad = 0.0
    for i in range(n):
        quad += yv[i] * sol[i]
    fit = -0.5 * (n * math.log(2.0 * math.pi) + logdet + quad)
    pen = -0.5 * trace / float(sigma2)
    Xt = k.mat(X_test) if X_test is not None else Xm
    pred = []
    var = []
    for t in range(len(Xt)):
        ks = [_rbf(Xt[t], Xm[i], g) for i in range(n)]
        # project through the inducing set, as the variational posterior does
        kz = [_rbf(Xt[t], Zm[j], g) for j in range(m)]
        w = k.cholsolve(Kmm, kz)
        qs = []
        for i in range(n):
            s = 0.0
            for a in range(m):
                s += Knm[i][a] * w[a]
            qs.append(s)
        p = 0.0
        for i in range(n):
            p += qs[i] * sol[i]
        pred.append(p)
        u = k.cholsolve(S, qs)
        q = 0.0
        for i in range(n):
            q += qs[i] * u[i]
        var.append(1.0 - q)
    return RichResult(
        title="Sparse variational GP",
        summary_lines=[("ELBO", fit + pen), ("trace penalty", pen)],
        payload={
            "estimate": fit + pen,
            "elbo": fit + pen,
            "fit_term": fit,
            "trace_penalty": pen,
            "trace": trace,
            "pred": pred,
            "var": var,
            "method": "Titsias (2009) collapsed variational bound, eq. (9)",
        },
    )


def cheatsheet():
    return "varKf: Sparse variational GP"


variationalgp = variational_gp
