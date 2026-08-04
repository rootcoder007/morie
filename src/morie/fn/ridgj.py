# morie.fn -- function file (rootcoder007/morie)
"""Penalized residual sum of squares of ridge regression."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['ridgeobj', 'ridge_objective', 'ridgeobjective']


def ridgeobj(X, y, beta, lam, add_intercept=True):
    """Penalized residual sum of squares of ridge regression.

    Formula: PRSS(beta, lambda) = sum_i (y_i - b0 - sum_j x_ij b_j)^2 + lambda sum_j b_j^2

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix, one record per row.
    y : array-like
        Response vector of length n.
    beta : array-like
        Coefficient vector.
    lam : float
        Regularization parameter lambda; must be non-negative.
    add_intercept : bool
        Treat the first entry of beta as an unpenalized intercept and prepend a column of ones to X.

    Returns
    -------
    RichResult
        ``prss``, ``rss``, ``penalty``, ``lambda``, ``n``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 3, Sect. 3.6.1 p. 81: PRSS_lambda(beta) = RSS(beta) + lambda beta'D beta with D = diag(0, 1, ..., 1), so the intercept is not penalized.  Read from the chapter PDF, not recalled.
    """
    Xm = C.mat(X)
    if add_intercept:
        Xm = C.cbind1(Xm)
    y = C.vec(y)
    b = C.vec(beta)
    lam = float(lam)
    n, p = len(Xm), len(Xm[0])
    if n != len(y):
        raise ValueError("X must have one row per entry of y")
    if len(b) != p:
        raise ValueError("beta must have one entry per column of the design")
    if lam < 0.0:
        raise ValueError("lambda must be non-negative")
    rss = sum((y[i] - sum(Xm[i][j] * b[j] for j in range(p))) ** 2
              for i in range(n))
    start = 1 if add_intercept else 0
    pen = lam * sum(b[j] * b[j] for j in range(start, p))
    return RichResult(payload={
        "prss": rss + pen, "rss": rss, "penalty": pen, "lambda": lam,
        "n": n, "p": p,
        "method": "Ridge penalized RSS, MVSML Sect. 3.6.1"})


ridge_objective = ridgeobj
ridgeobjective = ridgeobj


def cheatsheet():
    return 'ridgj: Penalized residual sum of squares of ridge regression.'
