# morie.fn -- function file (rootcoder007/morie)
"""Penalized residual sum of squares of the lasso, with its soft-threshold step."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['lassoobj', 'lasso_objective', 'lassoobjective']


def lassoobj(X, y, beta, lam, add_intercept=True):
    """Penalized residual sum of squares of the lasso, with its soft-threshold step.

    Formula: PRSS(beta, lambda) = sum_i (y_i - b0 - sum_j x_ij b_j)^2 + lambda sum_j |b_j|;  S(b, lambda) = sign(b) max(|b| - lambda, 0)

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
        ``prss``, ``rss``, ``penalty``, ``soft``, ``lambda``, ``n``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 3, Sect. 3.6.2 pp. 93-94: the lasso penalizes the RSS by the sum of the absolute regression coefficients, and the coordinate-wise descent step of Friedman et al. (2007) uses the soft-threshold operator S(b, lambda) = b - lambda if b > 0 and lambda < |b|, b + lambda if b < 0 and lambda < |b|, and 0 if lambda >= |b| -- which is sign(b) max(|b| - lambda, 0).  ``soft`` reports that operator applied to the supplied coefficients.  Read from the chapter PDF, not recalled.
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
    pen = lam * sum(abs(b[j]) for j in range(start, p))
    soft = [0.0 if abs(v) <= lam else (v - lam if v > 0.0 else v + lam)
            for v in b]
    return RichResult(payload={
        "prss": rss + pen, "rss": rss, "penalty": pen, "soft": soft,
        "lambda": lam, "n": n, "p": p,
        "method": "Lasso penalized RSS, MVSML Sect. 3.6.2"})


lasso_objective = lassoobj
lassoobjective = lassoobj


def cheatsheet():
    return 'lassj: Penalized residual sum of squares of the lasso, with its soft-threshold step.'
