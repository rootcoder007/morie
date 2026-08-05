# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Genomic BLUP.

VanRaden (2008), "Efficient methods to compute genomic predictions",
Journal of Dairy Science 91(11):4414-4423,
doi:10.3168/jds.2007-0980.  The model is

    y = X beta + Z u + e,   u ~ N(0, G sigma_u^2),  e ~ N(0, I sigma_e^2),

and Henderson's mixed model equations give beta-hat and u-hat jointly:

    [ X'X      X'Z          ] [ beta ]   [ X'y ]
    [ Z'X   Z'Z + G^{-1} k  ] [  u   ] = [ Z'y ],   k = sigma_e^2/sigma_u^2.

With G = I the second block is ridge regression with penalty k, which
is the closed form the tests check; as k -> 0 the system tends to
ordinary least squares on the concatenated design.  G is ridged before
inversion because VanRaden's G is singular whenever markers outnumber
individuals.
"""

from __future__ import annotations

from . import _array_core as np
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gblup_estimator"]


def gblup_estimator(y, X, Z, G, var_u=1.0, var_e=1.0, ridge=1e-8):
    """Solve the mixed model equations for fixed and genomic effects."""
    yv = core.vec(y)
    Xm = core.mat(X)
    Zm = core.mat(Z)
    Gm = core.mat(G)
    n = len(yv)
    if n == 0:
        raise ValueError("gblup_estimator: y is empty")
    if len(Xm) != n or len(Zm) != n:
        raise ValueError("gblup_estimator: X and Z must have one row per observation")
    p = len(Xm[0])
    q = len(Zm[0])
    if len(Gm) != q or len(Gm[0]) != q:
        raise ValueError("gblup_estimator: G must be q x q")
    vu = float(var_u)
    ve = float(var_e)
    if vu <= 0 or ve <= 0:
        raise ValueError("gblup_estimator: variance components must be positive")
    k = ve / vu
    Gr = [[Gm[i][j] + (float(ridge) if i == j else 0.0) for j in range(q)] for i in range(q)]
    Ginv = [[float(v) for v in row] for row in np.linalg.inv(Gr)]
    m = p + q
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for a in range(p):
        for c in range(p):
            A[a][c] = sum(Xm[i][a] * Xm[i][c] for i in range(n))
        for c in range(q):
            A[a][p + c] = sum(Xm[i][a] * Zm[i][c] for i in range(n))
        b[a] = sum(Xm[i][a] * yv[i] for i in range(n))
    for a in range(q):
        for c in range(p):
            A[p + a][c] = sum(Zm[i][a] * Xm[i][c] for i in range(n))
        for c in range(q):
            A[p + a][p + c] = sum(Zm[i][a] * Zm[i][c] for i in range(n)) + Ginv[a][c] * k
        b[p + a] = sum(Zm[i][a] * yv[i] for i in range(n))
    sol = [float(v) for v in np.linalg.solve(A, b)]
    beta = sol[:p]
    u = sol[p:]
    fitted = [sum(Xm[i][a] * beta[a] for a in range(p)) + sum(Zm[i][c] * u[c] for c in range(q)) for i in range(n)]
    resid = [yv[i] - fitted[i] for i in range(n)]
    ss = 0.0
    for v in resid:
        ss += v * v
    return RichResult(
        title="Genomic BLUP",
        summary_lines=[("n", n), ("fixed", p), ("random", q)],
        payload={
            "estimate": u[0],
            "beta": beta,
            "u": u,
            "fitted": fitted,
            "residual_ss": ss,
            "lambda": k,
            "n": n,
            "method": "Henderson mixed model equations with G from VanRaden (2008)",
        },
    )


def cheatsheet():
    return "gblupr: genomic BLUP"


# compact alias per ledger/NAMING.md
gblupestimator = gblup_estimator
