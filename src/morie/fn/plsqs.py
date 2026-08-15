# morie.fn -- function file (rootcoder007/morie)
r"""Partial least squares regression by NIPALS.

Ordinary least squares fails when the predictors are collinear or when
there are more of them than observations: X'X is singular. PLS builds
orthogonal components that maximise the COVARIANCE with the response
rather than the variance of X alone, which is what separates it from
principal component regression -- PCR can spend its first component on a
direction of X that has nothing to do with y.

Each NIPALS iteration takes the loading w proportional to X'y, scores
t = Xw, deflates X by its rank-one projection and repeats. The number of
components is the model: too few underfits, too many reproduces OLS
including its instability, so ``explained_x`` and ``explained_y`` are
reported per component rather than only in total.

References
----------
Wold, S., Sjostrom, M. and Eriksson, L. (2001) "PLS-regression: a basic
tool of chemometrics", *Chemometrics and Intelligent Laboratory Systems*
**58**(2), 109-130, doi:10.1016/S0169-7439(01)00155-1.

Wold, H. (1975) "Soft modelling by latent variables: the non-linear
iterative partial least squares (NIPALS) approach", in *Perspectives in
Probability and Statistics*, Academic Press, 117-142.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["pls_regression"]

_EPS = 1e-12


def pls_regression(X, Y, n_components=2):
    r"""NIPALS PLS1: scores, loadings and the regression coefficients."""
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    y = [float(v) for v in k.vec(Y)]
    n = len(Xm)
    if n == 0:
        raise ValueError("plsqs: no observations")
    if len(y) != n:
        raise ValueError("plsqs: %d rows but %d responses" % (n, len(y)))
    p = len(Xm[0])
    a = int(n_components)
    if a < 1:
        raise ValueError("plsqs: at least one component is required")
    a = min(a, p, n - 1) if n > 1 else min(a, p)

    xbar = [sum(Xm[i][j] for i in range(n)) / n for j in range(p)]
    ybar = sum(y) / n
    E = [[Xm[i][j] - xbar[j] for j in range(p)] for i in range(n)]
    f = [v - ybar for v in y]
    ss_x0 = sum(E[i][j] ** 2 for i in range(n) for j in range(p))
    ss_y0 = sum(v * v for v in f)

    W, T, P, q = [], [], [], []
    ex_x, ex_y = [], []
    for _ in range(a):
        w = [sum(E[i][j] * f[i] for i in range(n)) for j in range(p)]
        nw = math.sqrt(sum(v * v for v in w))
        if nw <= _EPS:
            break
        w = [v / nw for v in w]
        t = [sum(E[i][j] * w[j] for j in range(p)) for i in range(n)]
        tt = sum(v * v for v in t)
        if tt <= _EPS:
            break
        pl = [sum(E[i][j] * t[i] for i in range(n)) / tt for j in range(p)]
        qj = sum(t[i] * f[i] for i in range(n)) / tt
        ss_x = sum((t[i] * pl[j]) ** 2 for i in range(n) for j in range(p))
        ss_y = qj * qj * tt
        for i in range(n):
            for j in range(p):
                E[i][j] -= t[i] * pl[j]
            f[i] -= qj * t[i]
        W.append(w)
        T.append(t)
        P.append(pl)
        q.append(qj)
        ex_x.append(ss_x / ss_x0 if ss_x0 > _EPS else 0.0)
        ex_y.append(ss_y / ss_y0 if ss_y0 > _EPS else 0.0)

    a = len(W)
    if a == 0:
        raise ValueError("plsqs: the response has no covariance with X")

    # B = W (P'W)^-1 q  -- the coefficients on the ORIGINAL scale
    PW = [[sum(P[r][j] * W[c][j] for j in range(p)) for c in range(a)]
          for r in range(a)]
    for r in range(a):
        PW[r][r] += _EPS
    rhs = list(q)
    z = k.cholsolve([[sum(PW[u][r] * PW[u][c] for u in range(a))
                      for c in range(a)] for r in range(a)],
                    [sum(PW[u][r] * rhs[u] for u in range(a))
                     for r in range(a)])
    beta = [sum(W[c][j] * z[c] for c in range(a)) for j in range(p)]
    intercept = ybar - sum(beta[j] * xbar[j] for j in range(p))
    fitted = [intercept + sum(Xm[i][j] * beta[j] for j in range(p))
              for i in range(n)]
    resid = [y[i] - fitted[i] for i in range(n)]
    sse = sum(v * v for v in resid)
    r2 = 1.0 - sse / ss_y0 if ss_y0 > _EPS else 0.0

    return RichResult(payload={
        "estimate": beta, "coefficients": beta, "intercept": intercept,
        "fitted": fitted, "residuals": resid,
        "scores": [[T[c][i] for c in range(a)] for i in range(n)],
        "weights": W, "loadings": P, "y_loadings": q,
        "explained_x": ex_x, "explained_y": ex_y,
        "n_components": a, "r_squared": r2, "n": n, "p": p,
        "method": "PLS1 regression by NIPALS (Wold, Sjostrom & Eriksson 2001)",
        "note": "components maximise covariance with y, not variance of X -- "
                "that is what separates PLS from principal component "
                "regression",
    })


def cheatsheet():
    return ("plsqs: pls_regression(X, Y, n_components) -> NIPALS PLS1 "
            "(Wold, Sjostrom & Eriksson 2001, Chemom. Intell. Lab. Syst. "
            "58(2), 109-130)")
