# morie.fn -- function file (rootcoder007/morie)
"""Log-ratio mean of a compositional data set."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["complrm", "compositional_lrmean"]


def complrm(X, total=1.0):
    """Mean of a compositional sample taken in log-ratio coordinates.

    Averaging in clr coordinates and mapping back is the only averaging
    that commutes with perturbation, and it lands exactly on the closed
    geometric mean -- the compositional centre.  Both are returned, so
    the identity clr^-1(mean clr) = C(g_1, ..., g_D) is visible rather
    than asserted.

    Formula: mean clr = (1/n) sum_k clr(x_k);  centre = clr^-1(mean clr)

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; strictly positive.
    total : float
        Constant the returned centre sums to.

    Returns
    -------
    RichResult
        ``clr_mean``, ``center``, ``sum_clr_mean``, ``n``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 4.  Consistent with the sibling module ``aitcen`` in this
    package, whose centre is the closed vector of column geometric
    means.
    """
    X = C.mat(X)
    n = len(X)
    D = len(X[0])
    for row in X:
        if any(v <= 0 for v in row):
            raise ValueError("compositions must be strictly positive")
    L = [[math.log(v) for v in row] for row in X]
    Z = [[L[k][j] - sum(L[k]) / D for j in range(D)] for k in range(n)]
    zm = [sum(Z[k][j] for k in range(n)) / n for j in range(D)]
    e = [math.exp(v) for v in zm]
    s = sum(e)
    k = float(total)
    return RichResult(payload={
        "clr_mean": zm, "center": [k * v / s for v in e],
        "sum_clr_mean": sum(zm), "n": n, "D": D,
        "method": "Log-ratio mean (clr average, closed back)"})


compositional_lrmean = complrm


def cheatsheet():
    return "aitlrm: centre = clr^-1(mean clr)"
