# morie.fn -- function file (rootcoder007/morie)
"""Ordinary least squares through the normal equations."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['olsnormeq', 'ols_normal_equations']


def olsnormeq(X, y, add_intercept=True):
    """Ordinary least squares through the normal equations.

    Formula: (X'X) beta = X'y;  beta = (X'X)^-1 X'y;  Var(beta) = sigma2 (X'X)^-1;  H = X(X'X)^-1 X'

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix, one record per row.
    y : array-like
        Response vector of length n.
    add_intercept : bool
        Prepend a column of ones to X.

    Returns
    -------
    RichResult
        ``beta``, ``fitted``, ``resid``, ``rss``, ``sigma2``, ``se``, ``leverage``, ``n``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 3, Sect. 3.2 pp. 72-73: setting the gradient of the residual sum of squares to zero gives the normal equations (X'X)beta = X'Y with unique solution beta = (X'X)^-1 X'y, variance sigma2 (X'X)^-1, and hat matrix H = X(X'X)^-1 X'.  sigma2 is the unbiased residual variance RSS/(n - p).  Read from the chapter PDF, not recalled.
    """
    Xm = C.mat(X)
    if add_intercept:
        Xm = C.cbind1(Xm)
    y = C.vec(y)
    n = len(Xm)
    if n != len(y):
        raise ValueError("X must have one row per entry of y")
    p = len(Xm[0])
    if n <= p:
        raise ValueError("need more records than columns for the residual variance")
    bhat, fitted, resid, xtxinv = C.lstsq(Xm, y)
    rss = sum(r * r for r in resid)
    s2 = rss / (n - p)
    return RichResult(payload={
        "beta": bhat, "fitted": fitted, "resid": resid,
        "rss": rss, "sigma2": s2,
        "se": [math.sqrt(s2 * xtxinv[j][j]) for j in range(p)],
        "leverage": C.hatdiag(Xm, xtxinv), "n": n, "p": p,
        "method": "OLS via the normal equations, MVSML Sect. 3.2"})


ols_normal_equations = olsnormeq


def cheatsheet():
    return 'olsnm: Ordinary least squares through the normal equations.'
