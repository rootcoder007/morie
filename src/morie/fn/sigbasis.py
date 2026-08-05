# morie.fn -- function file (rootcoder007/morie)
"""Sigmoid basis transformation of a design matrix.

Hastie, Tibshirani and Friedman (2009), *The Elements of Statistical
Learning*, 2nd ed., Springer, Section 2.6.1, book p. 30 (PDF p. 49):

    h_k(x) = 1 / (1 + exp(-x' beta_k))                              (2.31)

the "sigmoid transformation common to neural network models".  With the
transformed features in hand the parameters theta of the expansion
(2.30) are estimated by minimising

    RSS(theta) = sum_i (y_i - f_theta(x_i))^2                       (2.32)

which is done here when y is supplied.  The book's (2.31) has no bias
term inside the sigmoid beyond whatever constant column x carries, and
none is added.
"""

from __future__ import annotations

from . import _s03core as k

from ._richresult import RichResult

__all__ = ["sigbasis"]


def sigbasis(X, B, y=None):
    """Evaluate h_k(x) = 1/(1 + exp(-x'beta_k)) of equation (2.31).

    Parameters
    ----------
    X : array-like
        N-by-p design.
    B : array-like
        p-by-K matrix whose columns are the beta_k of (2.31).
    y : array-like, optional
        N-vector; when given, theta of (2.30) is fitted by least squares
        (2.32) on the transformed features.

    Returns
    -------
    RichResult with keys estimate, h, theta, fitted, rss, n, p, K, method.
    """
    Xm = k.mat(X)
    Bm = k.mat(B)
    n = k.nrow(Xm)
    if n == 0:
        raise ValueError("sigbasis: X is empty")
    p = k.ncol(Xm)
    if k.nrow(Bm) != p:
        raise ValueError("sigbasis: B must have one row per column of X")
    K = k.ncol(Bm)
    if K == 0:
        raise ValueError("sigbasis: B has no columns")
    H = [[k.sigmoid(sum(Xm[i][a] * Bm[a][c] for a in range(p))) for c in range(K)] for i in range(n)]
    theta = None
    fitted = None
    rss = float("nan")
    if y is not None:
        yv = k.vec(y)
        if len(yv) != n:
            raise ValueError("sigbasis: X and y must have the same number of rows")
        if n < K:
            raise ValueError("sigbasis: fewer observations than basis functions")
        theta = k.lstsq(H, yv, 0.0)
        fitted = [sum(H[i][c] * theta[c] for c in range(K)) for i in range(n)]
        rss = sum((yv[i] - fitted[i]) ** 2 for i in range(n))
    return RichResult(
        title="Sigmoid basis, ESL eq. (2.31)",
        summary_lines=[("n", n), ("K", K), ("rss", rss)],
        payload={
            "estimate": H[0][0],
            "h": H,
            "theta": theta,
            "fitted": fitted,
            "rss": rss,
            "n": n,
            "p": p,
            "K": K,
            "method": "Hastie-Tibshirani-Friedman (2009) ESL eqs. (2.30)-(2.32)",
        },
    )


def cheatsheet():
    return "sigbasis: h_k(x) = 1/(1+exp(-x'beta_k)), ESL eq. (2.31)"
