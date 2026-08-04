# morie.fn -- function file (rootcoder007/morie)
"""Total variance of a compositional data set."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['comptotvar', 'aitchison_total_variance']


def comptotvar(X):
    """Total variance of a compositional data set.

    Two equivalent forms are returned because they are equivalent, and checking that they agree is the cheapest test that the log-ratio bookkeeping is right: the trace of the centred log-ratio covariance matrix, and the D-scaled sum of all pairwise log-ratio variances. Sample variances (denominator n-1) are used throughout.


    Formula: totvar(x) = trace(Gamma) = (1/D) sum_{i<j} var{log(x_i/x_j)}

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; strictly positive.

    Returns
    -------
    RichResult
        ``totvar``, ``totvar_trace``, ``clr_var``, ``n``, ``D``.

    References
    ----------
    Aitchison, A Concise Guide to Compositional Data Analysis,
    Chapter 2.  Verified against the text: the centre estimate is
    xi-hat = C(g_1, ..., g_D) with g_i the geometric mean of the ith
    component, and totvar(x) = trace(Gamma) = (1/D) sum_{i<j}
    var{log(x_i/x_j)}.
    """
    X = C.mat(X)
    n = len(X); D = len(X[0])
    for row in X:
        if any(v <= 0 for v in row):
            raise ValueError("compositions must be strictly positive")
    L = [[math.log(v) for v in row] for row in X]
    tot = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            tot += C.var([L[k][i] - L[k][j] for k in range(n)], 1)
    tot /= D
    clr = [[L[k][i] - sum(L[k]) / D for i in range(D)] for k in range(n)]
    cv = [C.var([clr[k][i] for k in range(n)], 1) for i in range(D)]
    return RichResult(payload={
        "totvar": tot, "totvar_trace": sum(cv), "clr_var": cv,
        "n": n, "D": D, "method": "Compositional total variance"})


aitchison_total_variance = comptotvar


def cheatsheet():
    return "aittvr: Total variance of a compositional data set."
