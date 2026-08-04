# morie.fn -- function file (rootcoder007/morie)
"""Chebyshev polynomial basis of the first kind."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['chebbasis', 'chebyshev_basis']


def chebbasis(x, K=5):
    """Chebyshev polynomial basis of the first kind.

    The three-term recurrence is used rather than the trigonometric definition because it is stable and stays valid outside [-1, 1], where arccos is not defined at all. On [-1, 1] the two agree, and the trigonometric form is returned as well so the caller can see that they do.


    Formula: T_0 = 1, T_1 = x, T_{n+1}(x) = 2 x T_n(x) - T_{n-1}(x); equivalently T_n(x) = cos(n arccos x)

    Parameters
    ----------
    x : array-like
        Points at which the basis is evaluated.
    K : int
        Highest degree; the basis has K+1 columns.

    Returns
    -------
    RichResult
        ``basis`` (n by K+1), ``degree``, ``trig`` (cos form where |x| <= 1), ``n``.

    References
    ----------
    Chebyshev (1853).  The original is not held locally; the recurrence
    T_{n+1} = 2 x T_n - T_{n-1} with T_0 = 1, T_1 = x and the identity
    T_n(cos t) = cos(n t) are the standard published definitions.
    """
    x = C.vec(x)
    K = int(K)
    if K < 0:
        raise ValueError("K must be non-negative")
    out = []
    for v in x:
        row = [1.0]
        if K >= 1:
            row.append(v)
        for n in range(1, K):
            row.append(2.0 * v * row[n] - row[n - 1])
        out.append(row)
    trig = [[math.cos(n * math.acos(v)) for n in range(K + 1)]
            if abs(v) <= 1.0 else [float("nan")] * (K + 1) for v in x]
    return RichResult(payload={
        "basis": out, "degree": K, "trig": trig, "n": len(x),
        "method": "Chebyshev polynomial basis (first kind)"})


chebyshev_basis = chebbasis


def cheatsheet():
    return "polyCh: Chebyshev polynomial basis of the first kind."
