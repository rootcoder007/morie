# morie.fn -- function file (rootcoder007/morie)
"""Robust regression through the minimum volume ellipsoid scatter.

Rousseeuw, P. J. (1985), "Multivariate estimation with high breakdown
point", in *Mathematical Statistics and Applications*, Vol. B,
Reidel, 283-297.  The MVE criterion is the one stated in the stub
docstring -- the smallest-volume ellipsoid covering h points -- and it
is applied here to the JOINT matrix Z = [X | y], exactly as module
mcdcv applies the MCD, so that a scatter estimator becomes a
regression estimator:

    beta = Sigma_XX^{-1} Sigma_Xy,     alpha = mu_y - beta' mu_X

with Sigma the MVE scatter of (X, y) and mu its centre.  The inflation
factor m2 multiplies Sigma_XX and Sigma_Xy alike, so it cancels out of
beta; the coefficients depend only on the SHAPE of the ellipsoid, not
its size.

That cancellation, together with the affine equivariance of the MVE,
gives this module its anchors, none of which runs through the search:
adding a constant to y must move alpha by exactly that constant and
leave beta untouched; multiplying y by a constant must multiply both
by exactly that constant; and multiplying a predictor column by a
constant must divide its coefficient by exactly that constant.

The search itself is anchored separately, in module mvedet, against
the closed-form univariate MVE -- the shortest interval containing h
points, which is the shortest-half construction of Rousseeuw (1984)
Theorem 2, p. 873.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _rousscore as R
from . import _s03core as k
from .mvedet import mve

from ._richresult import RichResult

__all__ = ["min_volume_ellipsoid"]


def min_volume_ellipsoid(y, X, h=None, n_starts=100000):
    """MVE scatter of [X | y] and the regression it implies.

    Parameters
    ----------
    y : array-like
        n responses.
    X : array-like
        n-by-q predictor matrix without an intercept column.
    h : int, optional
        Coverage; defaults to [(n + p + 1) / 2] with p = q + 1.
    n_starts : int
        Cap on the number of (p+1)-subsets enumerated.

    Returns
    -------
    estimate : the MVE objective, proportional to the squared volume
    coef, intercept, center, cov, m2, subset, covered, h, n, p
    """
    yy = k.vec(y)
    Xm = k.mat(X)
    n = len(yy)
    if n == 0:
        raise ValueError("min_volume_ellipsoid: y is empty")
    if k.nrow(Xm) != n:
        raise ValueError("min_volume_ellipsoid: X must have one row per response")
    q = k.ncol(Xm)
    if q == 0:
        raise ValueError("min_volume_ellipsoid: X has no columns")
    Z = [[Xm[i][j] for j in range(q)] + [yy[i]] for i in range(n)]
    r = mve(Z, h, n_starts)
    S = r["cov"]
    mu = r["center"]
    Sxx = [[S[a][b] for b in range(q)] for a in range(q)]
    Sxy = [S[a][q] for a in range(q)]
    beta = R.lusolve(Sxx, Sxy)
    if beta is None:
        raise ValueError("min_volume_ellipsoid: the predictor block of the MVE scatter is singular")
    alpha = mu[q]
    for a in range(q):
        alpha -= beta[a] * mu[a]
    return RichResult(
        title="MVE regression",
        summary_lines=[("n", n), ("q", q), ("h", r["h"]), ("objective", r["estimate"]), ("intercept", alpha)],
        payload={
            "estimate": r["estimate"],
            "coef": beta,
            "intercept": alpha,
            "center": mu,
            "cov": S,
            "cov_raw": r["cov_raw"],
            "m2": r["m2"],
            "subset": r["subset"],
            "covered": r["covered"],
            "h": r["h"],
            "n": n,
            "p": q + 1,
            "method": "Rousseeuw (1985) MVE of [X | y]; beta = Sigma_XX^-1 Sigma_Xy, alpha = mu_y - beta mu_X",
        },
    )


def cheatsheet():
    return "mvecv: regression from the MVE scatter of (X, y)"
