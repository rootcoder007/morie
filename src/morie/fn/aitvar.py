# morie.fn -- function file (rootcoder007/morie)
"""Variation matrix of a compositional data set."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["compvar", "aitchison_variation"]


def compvar(X):
    """Variation matrix: the pairwise log-ratio variances.

    The variation matrix is Aitchison's replacement for the covariance
    matrix of raw proportions, which is meaningless -- proportions sum
    to a constant, so their covariances are forced negative and change
    when a part is dropped.  Entry (i, j) is the variance of a single
    log-ratio, so it is unchanged by closure and by any subcomposition
    that keeps parts i and j.

    Sample variances (denominator n-1) throughout, matching the sibling
    modules ``aitcen`` and ``aittvr``.

    Formula: tau_ij = var( log(x_i / x_j) ),  totvar = (1/D) sum_{i<j} tau_ij

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; strictly positive.

    Returns
    -------
    RichResult
        ``variation`` (D x D, zero diagonal), ``totvar``, ``n``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 4.  The relation totvar = (1/D) sum_{i<j} tau_ij is the one
    already used by the sibling module ``aittvr`` in this package.
    """
    X = C.mat(X)
    n = len(X)
    D = len(X[0])
    for row in X:
        if any(v <= 0 for v in row):
            raise ValueError("compositions must be strictly positive")
    L = [[math.log(v) for v in row] for row in X]
    tau = [[0.0] * D for _ in range(D)]
    tot = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            v = C.var([L[k][i] - L[k][j] for k in range(n)], 1)
            tau[i][j] = v
            tau[j][i] = v
            tot += v
    return RichResult(payload={
        "variation": tau, "totvar": tot / D, "n": n, "D": D,
        "method": "Compositional variation matrix"})


aitchison_variation = compvar


def cheatsheet():
    return "aitvar: tau_ij = var(log(x_i/x_j))"
