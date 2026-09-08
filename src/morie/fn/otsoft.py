"""Soft assignment matrix from entropic OT for matching."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_softassignment"]


def ot_softassignment(a, b, C, epsilon, max_iter=200):
    """
    Soft assignment matrix from an entropic transport plan.

    Formula: T from Sinkhorn, then each row normalised to sum 1

    Verified against Cuturi (2013) eq. (2) for the plan, and Peyre &
    Cuturi (2019) Remark 4.11 for reading a plan row as the conditional
    distribution of the destination given the source -- sources
    consulted. Row ``i`` of the result is ``T[i, :] / a_i``.

    Parameters
    ----------
    a, b : array-like
        Non-negative marginals; closed internally.
    C : nested sequence
        Cost matrix.
    epsilon : float
        Regularisation strength.
    max_iter : int, optional
        Fixed number of Sinkhorn scalings (default 200).

    Returns
    -------
    RichResult
        Keys: estimate (the row-normalised matrix), T, entropy_mean,
        hard, method. ``hard`` is the argmax column per row.

    References
    ----------
    Cuturi, M. (2013). Sinkhorn Distances. NIPS 26. Eq. (2).
    Peyre, G. & Cuturi, M. (2019), Remark 4.11.
    """
    T, u, v, av, bv = _big2.sinkhorn(a, b, C, epsilon, max_iter)
    nr, nc = len(T), len(T[0])
    P = []
    hard = []
    hsum = 0.0
    for i in range(nr):
        s = sum(T[i])
        row = [T[i][j] / s for j in range(nc)] if s > 0.0 else [0.0] * nc
        P.append(row)
        best = 0
        for j in range(1, nc):
            if row[j] > row[best]:
                best = j
        hard.append(best)
        hsum += _big2.entropy(row, None) if s > 0.0 else 0.0
    return RichResult(
        payload={
            "estimate": P,
            "T": T,
            "entropy_mean": hsum / nr,
            "hard": hard,
            "method": "Row-normalised entropic plan (soft assignment) -- Cuturi (2013) eq. (2)",
        }
    )


def cheatsheet():
    return "otsoft: Soft assignment matrix from entropic OT for matching"
