"""Pairwise squared-Euclidean cost matrix."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_cost_pairwise"]


def ot_cost_pairwise(X, Y):
    """
    Squared-Euclidean ground cost matrix.

    Formula: C_ij = ||x_i - y_j||^2

    Verified against Peyre & Cuturi (2019), Remark 2.19 and Sec. 2.4
    (the p = 2 Wasserstein ground cost) -- source consulted.

    Parameters
    ----------
    X, Y : nested sequence
        Point sets, ``n x d`` and ``m x d``.

    Returns
    -------
    RichResult
        Keys: estimate (the matrix), nrow, ncol, total, method.

    References
    ----------
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport,
    Sec. 2.4.
    """
    A = _big2.mat(X)
    B = _big2.mat(Y)
    d = len(A[0])
    if len(B[0]) != d:
        raise ValueError("X and Y must have the same dimension")
    C = []
    total = 0.0
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            s = 0.0
            for k in range(d):
                t = A[i][k] - B[j][k]
                s += t * t
            row.append(s)
            total += s
        C.append(row)
    return RichResult(
        payload={
            "estimate": C,
            "nrow": len(A),
            "ncol": len(B),
            "total": total,
            "method": "Squared-Euclidean ground cost -- Peyre & Cuturi (2019) Sec. 2.4",
        }
    )


def cheatsheet():
    return "otc2c: Pairwise squared-Euclidean cost matrix"


# compact alias per ledger/NAMING.md
otcostpairwise = ot_cost_pairwise
