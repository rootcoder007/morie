# morie.fn -- function file (rootcoder007/morie)
"""Robust regression through the minimum covariance determinant scatter.

Rousseeuw, P. J. (1984), "Least median of squares regression",
*Journal of the American Statistical Association* 79(388), 871-880,
and Rousseeuw, P. J. (1985), "Multivariate estimation with high
breakdown point", Reidel, 283-297.  The MCD criterion itself is the
one stated in the stub docstring,

    argmin over subsets H of size h of det(cov(X[H])),

and it is applied here to the JOINT matrix Z = [X | y].  That is the
standard way a high-breakdown scatter estimator is turned into a
high-breakdown regression estimator: partition the MCD scatter of
(X, y) as

    Sigma = [[ Sigma_XX, Sigma_Xy ], [ Sigma_yX, Sigma_yy ]]

and read the coefficients off the population regression formula

    beta = Sigma_XX^{-1} Sigma_Xy,     alpha = mu_y - beta' mu_X.

Applying the MCD to X alone would ignore y entirely and could not
produce a regression at all, which is why y enters the criterion here.

The consistency factor cancels out of beta and alpha, since it
multiplies Sigma_XX and Sigma_Xy alike; it is still reported.

The construction has an exact consequence used as this module's
anchor.  With h = n the MCD subset is the whole sample, the scatter is
the ordinary sample covariance, and the formula above is then
identically the ordinary least squares fit with an intercept.  So at
h = n this function must reproduce OLS to the last digit, and the OLS
coefficients are obtained independently from the normal equations.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _rousscore as R
from . import _s03core as k
from .mcdv import consistency_factor

from ._richresult import RichResult

__all__ = ["min_covariance_determinant"]


def min_covariance_determinant(y, X, h=None, max_subsets=200000):
    """MCD scatter of [X | y] and the regression it implies.

    Parameters
    ----------
    y : array-like
        n responses.
    X : array-like
        n-by-q predictor matrix, WITHOUT an intercept column; the
        intercept is produced by the centring, as in the formula above.
    h : int, optional
        Subset size; defaults to [(n + p + 1) / 2] with p = q + 1.
    max_subsets : int
        Refuse rather than enumerate more than this many subsets.

    Returns
    -------
    estimate : the minimised determinant of the joint raw covariance
    coef     : the q slope coefficients
    intercept : alpha
    center   : the MCD location of (X, y)
    cov_raw, cov, factor, subset, h, n, p
    """
    yy = k.vec(y)
    Xm = k.mat(X)
    n = len(yy)
    if n == 0:
        raise ValueError("min_covariance_determinant: y is empty")
    if k.nrow(Xm) != n:
        raise ValueError("min_covariance_determinant: X must have one row per response")
    q = k.ncol(Xm)
    if q == 0:
        raise ValueError("min_covariance_determinant: X has no columns")
    Z = [[Xm[i][j] for j in range(q)] + [yy[i]] for i in range(n)]
    p = q + 1
    hh = R.mcd_h(n, p) if h is None else int(h)
    if hh <= p:
        raise ValueError("min_covariance_determinant: h must exceed p = q + 1")
    if hh > n:
        raise ValueError("min_covariance_determinant: h cannot exceed the number of observations")
    total = R.nchoosek(n, hh)
    if total > max_subsets:
        raise ValueError("min_covariance_determinant: %d subsets exceeds max_subsets" % total)
    best_idx = None
    best_det = None
    for idx in R.combos(n, hh):
        mu, S = R.meancov(Z, idx)
        d = R.covdet(S)
        if best_det is None or d < best_det:
            best_det = d
            best_idx = idx
    mu, S = R.meancov(Z, best_idx)
    Sxx = [[S[a][b] for b in range(q)] for a in range(q)]
    Sxy = [S[a][q] for a in range(q)]
    beta = R.lusolve(Sxx, Sxy)
    if beta is None:
        raise ValueError("min_covariance_determinant: the predictor scatter of the best subset is singular")
    alpha = mu[q]
    for a in range(q):
        alpha -= beta[a] * mu[a]
    c0 = consistency_factor(hh, n, p)
    Sc = [[S[a][b] * c0 for b in range(p)] for a in range(p)]
    return RichResult(
        title="MCD regression",
        summary_lines=[("n", n), ("q", q), ("h", hh), ("det", best_det), ("intercept", alpha)],
        payload={
            "estimate": best_det,
            "coef": beta,
            "intercept": alpha,
            "center": mu,
            "cov_raw": S,
            "cov": Sc,
            "factor": c0,
            "subset": [float(v) for v in best_idx],
            "h": hh,
            "n": n,
            "p": p,
            "method": "MCD of [X | y] by exhaustive enumeration; beta = Sigma_XX^-1 Sigma_Xy, alpha = mu_y - beta mu_X",
        },
    )


def cheatsheet():
    return "mcdcv: regression from the MCD scatter of (X, y)"
