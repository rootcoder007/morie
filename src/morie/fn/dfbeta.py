# morie.fn -- function file (rootcoder007/morie)
"""DFBETAS, the scaled per-coefficient deletion influence."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['dfbetas']


def dfbetas(X, y, intercept=True):
    """DFBETAS, the scaled per-coefficient deletion influence.

    Deleting an observation and refitting n times is unnecessary: the rank-one update gives b - b_(i) = C x_i e_i / (1 - h_ii) exactly, and s_(i) follows from the residual sum of squares in closed form. Scaling by the delete-one root mean square rather than the full-fit one is what makes the statistic comparable across observations.


    Formula: DFBETAS_ij = (b_j - b_(i)j) / (s_(i) sqrt(C_jj)), C = (X'X)^-1

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    intercept : bool
        Prepend a column of ones.

    Returns
    -------
    RichResult
        ``dfbetas`` (n by p), ``cutoff``, ``leverage``, ``beta``, ``sigma_i``, ``n``, ``p``.

    References
    ----------
    Belsley, Kuh and Welsch (1980), Regression Diagnostics: Identifying
    Influential Data and Sources of Collinearity, Wiley.  The book is
    not held locally; the definitions and the cutoffs used here
    (2/sqrt(n) for DFBETAS, 2 sqrt(p/n) for DFFITS, scaling by the
    delete-one root mean square s_(i)) are as documented by the SAS
    and R reference implementations, which cite BKW for them.
    """
    X = C.mat(X)
    if intercept:
        X = C.cbind1(X)
    y = C.vec(y)
    n = len(X); p = len(X[0])
    beta, fit, res, xtxinv = C.lstsq(X, y)
    h = C.hatdiag(X, xtxinv)
    rss = sum(v * v for v in res)
    df = n - p
    out, si = [], []
    for i in range(n):
        d = 1.0 - h[i]
        if d <= 0:
            out.append([float("nan")] * p); si.append(float("nan")); continue
        s2i = (rss - res[i] * res[i] / d) / (df - 1) if df > 1 else float("nan")
        s = math.sqrt(s2i) if s2i == s2i and s2i > 0 else float("nan")
        si.append(s)
        cx = C.matvec(xtxinv, X[i])
        out.append([cx[j] * res[i] / d / (s * math.sqrt(xtxinv[j][j]))
                    for j in range(p)])
    return RichResult(payload={
        "dfbetas": out, "cutoff": 2.0 / math.sqrt(n), "leverage": h,
        "beta": beta, "sigma_i": si, "n": n, "p": p,
        "method": "DFBETAS (Belsley-Kuh-Welsch)"})



def cheatsheet():
    return "dfbeta: DFBETAS, the scaled per-coefficient deletion influence."
