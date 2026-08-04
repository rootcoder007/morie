# morie.fn -- function file (rootcoder007/morie)
"""Cyclical monotonicity check for a coupling."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ot_cyclical_weight"]


def ot_cyclical_weight(X, Y, Cost, perm):
    """Test whether a matching can be improved by re-shuffling a cycle.

    Cyclical monotonicity is the combinatorial fingerprint of
    optimality: a coupling is optimal exactly when no finite cycle of
    reassignments lowers the cost.  Checking every permutation is
    factorial, so the practical test -- and the one implemented -- is
    over all transpositions, which is the two-cycle case and already
    catches most non-optimal matchings.

    Formula: for all pairs ``(i, j)``,
    ``c(x_s(i), y_i) + c(x_s(j), y_j) <= c(x_s(j), y_i) + c(x_s(i), y_j)``.

    Parameters
    ----------
    X, Y : array-like
        Point sets, kept for interface symmetry with the cost matrix.
    Cost : array-like, shape (n, n)
        Cost matrix, ``Cost[i][j] = c(x_i, y_j)``.
    perm : array-like, shape (n,)
        Zero-based assignment: ``y_i`` is matched to ``x_perm[i]``.

    Returns
    -------
    RichResult
        ``is_cm`` (1 when no transposition improves), ``slack`` (the
        most negative improvement found, zero when cyclically
        monotone), ``estimate`` (total cost), ``n``.

    References
    ----------
    Villani, C. (2003).  Topics in Optimal Transportation.  AMS GSM 58,
    theorem 2.12 (cyclical monotonicity).
    """
    M = C.mat(Cost)
    p = [int(round(v)) for v in C.vec(perm)]
    n = len(p)
    total = sum(M[p[i]][i] for i in range(n))
    worst = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = (M[p[j]][i] + M[p[i]][j]) - (M[p[i]][i] + M[p[j]][j])
            if d < worst:
                worst = d
    return RichResult(payload={
        "is_cm": 1.0 if worst >= 0.0 else 0.0, "slack": worst,
        "estimate": total, "n": n,
        "method": "Cyclical monotonicity over transpositions"})


def cheatsheet():
    return "otcw: Cyclical monotonicity check for a coupling."
