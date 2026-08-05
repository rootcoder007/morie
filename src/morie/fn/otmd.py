# morie.fn -- function file (rootcoder007/morie)
"""Mahalanobis ground cost for optimal transport."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_mahalanobis_distance_ot"]


def _inverse(S):
    d = len(S)
    cols = []
    for j in range(d):
        e = [1.0 if i == j else 0.0 for i in range(d)]
        cols.append(core.cholsolve(S, e))
    return [[cols[j][i] for j in range(d)] for i in range(d)]


def ot_mahalanobis_distance_ot(X, Y, Sigma):
    """Ground cost that measures distance in units of the data's own spread.

    Euclidean cost silently declares the coordinates equally important and
    uncorrelated; on any real feature set that is false, and the transport
    plan inherits the lie.  Whitening by the covariance removes both the
    scale and the correlation, so the cost is invariant to any full-rank
    affine recoding of the data.

    Formula: ``C_ij = (x_i - y_j)' Sigma^{-1} (x_i - y_j)``.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Source points.
    Y : array-like, shape (m, d)
        Target points.
    Sigma : array-like, shape (d, d)
        Covariance, symmetric positive definite.

    Returns
    -------
    RichResult
        ``C`` (the cost matrix), ``cost`` (the exact transport cost under
        uniform marginals), ``n``, ``m``, ``d``.

    References
    ----------
    De Maesschalck, R., Jouan-Rimbaud, D. and Massart, D. L. (2000).  The
    Mahalanobis distance.  Chemometrics and Intelligent Laboratory Systems
    50(1):1-18.  doi:10.1016/S0169-7439(99)00047-7.
    """
    A = core.mat(X)
    B = core.mat(Y)
    S = core.mat(Sigma)
    d = len(A[0])
    if len(B[0]) != d or len(S) != d or len(S[0]) != d:
        raise ValueError("Sigma must be d by d and match both point clouds")
    Si = _inverse(S)
    C = []
    for xi in A:
        row = []
        for yj in B:
            dv = [xi[k] - yj[k] for k in range(d)]
            row.append(sum(dv[k] * Si[k][l] * dv[l]
                           for k in range(d) for l in range(d)))
        C.append(row)
    n, m = len(A), len(B)
    _, cost = ot.emd([1.0 / n] * n, [1.0 / m] * m, C)
    return RichResult(payload={
        "C": C, "cost": cost, "n": n, "m": m, "d": d,
        "method": "Mahalanobis ground cost"})


def cheatsheet():
    return "otmd: Mahalanobis ground cost matrix and its transport cost"
