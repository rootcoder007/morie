"""Pairwise L_p cost matrix."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_cost_lp"]


def ot_cost_lp(X, Y, p=2.0):
    """
    l_p ground cost matrix.

    Formula: C_ij = ||x_i - y_j||_p

    Verified against Peyre & Cuturi (2019), Sec. 2.4 -- source
    consulted. ``p = inf`` gives the sup norm.

    Parameters
    ----------
    X, Y : nested sequence
        Point sets, ``n x d`` and ``m x d``.
    p : float, optional
        Norm order, at least 1 (default 2); ``float('inf')`` allowed.

    Returns
    -------
    RichResult
        Keys: estimate (the matrix), p, nrow, ncol, total, method.

    References
    ----------
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport,
    Sec. 2.4.
    """
    pv = float(p)
    if pv < 1.0:
        raise ValueError("p must be at least 1")
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
            if pv == float("inf"):
                s = max(abs(A[i][k] - B[j][k]) for k in range(d))
            else:
                s = 0.0
                for k in range(d):
                    s += abs(A[i][k] - B[j][k]) ** pv
                s = s ** (1.0 / pv)
            row.append(s)
            total += s
        C.append(row)
    return RichResult(
        payload={
            "estimate": C,
            "p": pv,
            "nrow": len(A),
            "ncol": len(B),
            "total": total,
            "method": "l_p ground cost -- Peyre & Cuturi (2019) Sec. 2.4",
        }
    )


def cheatsheet():
    return "otc2p: Pairwise L_p cost matrix"


# compact alias per ledger/NAMING.md
otcostlp = ot_cost_lp
