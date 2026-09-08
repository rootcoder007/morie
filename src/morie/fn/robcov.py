# morie.fn -- function file (rootcoder007/morie)
"""Heteroskedasticity-consistent standard errors."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sandwich_robust_se"]


def sandwich_robust_se(X, y, kind="HC0"):
    """Standard errors that survive the wrong variance assumption.

    Ordinary least squares stays unbiased when the errors are
    heteroskedastic; only its standard errors are wrong.  The sandwich
    replaces the assumed ``sigma^2 (X'X)^-1`` with a middle built from
    the observed squared residuals, so nothing about the variance has to
    be modelled.  HC0 is biased down in small samples, which is what the
    HC1 to HC3 corrections exist to fix -- HC3 is the safest default
    when leverage is uneven.

    Formula: ``V = (X'X)^-1 X' Omega X (X'X)^-1`` with
    ``Omega = diag(w_i e_i^2)``: HC0 has ``w = 1``, HC1
    ``n / (n - p)``, HC2 ``1 / (1 - h_ii)``, HC3
    ``1 / (1 - h_ii)^2``.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design; supply your own intercept column.
    y : array-like, shape (n,)
        Response.
    kind : str, default "HC0"
        One of ``HC0``, ``HC1``, ``HC2``, ``HC3``.

    Returns
    -------
    RichResult
        ``estimate`` (the robust standard errors), ``coef``, ``V``,
        ``ols_se``, ``n``.

    References
    ----------
    White, H. (1980).  A heteroskedasticity-consistent covariance matrix
    estimator and a direct test for heteroskedasticity.  Econometrica
    48:817-838.  The HC1 to HC3 variants are MacKinnon, J. G. & White,
    H. (1985), Journal of Econometrics 29:305-325.
    """
    Xm = C.mat(X)
    yv = C.vec(y)
    n, p = C.shape(Xm)
    beta, fitted, resid, xtxinv = S.ols(Xm, yv)
    h = [C.dot(Xm[i], C.matvec(xtxinv, Xm[i])) for i in range(n)]
    w = []
    for i in range(n):
        if kind == "HC1":
            w.append(n / (n - p))
        elif kind == "HC2":
            w.append(1.0 / (1.0 - h[i]))
        elif kind == "HC3":
            w.append(1.0 / (1.0 - h[i]) ** 2)
        else:
            w.append(1.0)
    mid = [[sum(Xm[i][a] * w[i] * resid[i] ** 2 * Xm[i][b] for i in range(n))
            for b in range(p)] for a in range(p)]
    V = C.matmul(C.matmul(xtxinv, mid), xtxinv)
    s2 = sum(t * t for t in resid) / (n - p)
    return RichResult(payload={
        "estimate": [math.sqrt(V[j][j]) for j in range(p)], "coef": beta, "V": V,
        "ols_se": [math.sqrt(s2 * xtxinv[j][j]) for j in range(p)], "n": n,
        "method": "Sandwich heteroskedasticity-consistent standard errors"})


def cheatsheet():
    return "robcov: Heteroskedasticity-consistent standard errors."
