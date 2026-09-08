# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""One outer iteration of iteratively reweighted least squares.

Holland and Welsch (1977), "Robust regression using iteratively
reweighted least-squares", Communications in Statistics -- Theory and
Methods 6(9):813-827, doi:10.1080/03610927708827533.  Their scheme
alternates between a weighted least-squares solve

    beta^(t+1) = (X' W^(t) X)^-1 X' W^(t) y

and a recomputation of the weights from the scaled residuals.  This
module performs exactly one such solve for a supplied weight vector,
and additionally reports the Huber weights that the next iteration
would use, so a caller can iterate.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["irls_solver"]


def irls_solver(y, X, weights):
    """Weighted least squares solve, one IRLS outer iteration.

    Parameters
    ----------
    y : array-like
        Response.
    X : array-like
        Covariate block; an intercept column is prepended.  May be None
        for an intercept-only fit.
    weights : array-like
        Non-negative case weights W^(t).  Pass ones for plain OLS.
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("irls_solver: y is empty")
    Z = core.design(X, n)
    if len(Z) != n:
        raise ValueError("irls_solver: X and y have different lengths")
    w = core.vec(weights) if weights is not None else [1.0] * n
    if len(w) != n:
        raise ValueError("irls_solver: weights and y have different lengths")
    for v in w:
        if v < 0:
            raise ValueError("irls_solver: weights must be non-negative")
    p = len(Z[0])
    if n <= p:
        raise ValueError("irls_solver: need more observations than parameters")
    ZtWZ = [[0.0] * p for _ in range(p)]
    ZtWy = [0.0] * p
    for i in range(n):
        for a in range(p):
            ZtWy[a] += Z[i][a] * w[i] * yv[i]
            for b in range(p):
                ZtWZ[a][b] += Z[i][a] * w[i] * Z[i][b]
    beta = core.cholsolve(ZtWZ, ZtWy)
    resid = [yv[i] - sum(Z[i][a] * beta[a] for a in range(p)) for i in range(n)]
    wss = sum(w[i] * resid[i] * resid[i] for i in range(n))
    sigma2 = wss / (n - p)
    inv = [core.cholsolve(ZtWZ, [1.0 if a == j else 0.0 for a in range(p)]) for j in range(p)]
    se = [math.sqrt(sigma2 * inv[j][j]) for j in range(p)]
    # weights the next Holland-Welsch iteration would use: Huber psi with
    # the usual 1.345 tuning constant on residuals scaled by the MAD.
    s = core.mad(resid)
    if s <= 0:
        s = 1.0
    nxt = []
    for r in resid:
        u = abs(r / s)
        nxt.append(1.0 if u <= 1.345 else 1.345 / u)
    est_idx = 1 if p > 1 else 0
    return RichResult(
        title="Iteratively reweighted least squares (one outer iteration)",
        summary_lines=[("n", n), ("parameters", p), ("estimate", beta[est_idx])],
        payload={
            "estimate": beta[est_idx],
            "coef": beta,
            "se": se[est_idx],
            "se_coef": se,
            "sigma2": sigma2,
            "wrss": wss,
            "next_weights": nxt,
            "scale": s,
            "p": p,
            "n": n,
            "method": "beta = (X' W X)^-1 X' W y, Holland & Welsch (1977)",
        },
    )


def cheatsheet():
    return "irlsfn: Iteratively reweighted least squares (one outer iteration)"


# compact alias per ledger/NAMING.md
irlssolver = irls_solver
