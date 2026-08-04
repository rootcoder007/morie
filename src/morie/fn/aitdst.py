# morie.fn -- function file (rootcoder007/morie)
"""Aitchison distance between two compositions."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["compdist", "aitchison_distance"]


def compdist(x, y):
    """Aitchison distance between two compositions.

    The distance that Euclidean distance on raw proportions gets wrong:
    it is invariant to the closure constant and to perturbation, and it
    is subcompositionally coherent, so dropping a part can never make
    two compositions look further apart.  Euclidean distance on raw
    parts has none of those properties.

    Formula: d_a(x, y) = sqrt( sum_i (clr(x)_i - clr(y)_i)^2 )
                       = sqrt( (1/D) sum_{i<j} (log(x_i/x_j) - log(y_i/y_j))^2 )

    Parameters
    ----------
    x, y : array-like
        Strictly positive vectors of parts, the same length.

    Returns
    -------
    RichResult
        ``distance``, ``distance_pairwise``, ``D``.  The two agree;
        both are returned as a self-check on the clr bookkeeping.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 8.  Verified against the reference implementation in the
    CRAN package ``compositions`` 2.0-9, whose ``dist`` on an
    ``acomp`` is the Euclidean distance of the clr coordinates.
    """
    x = C.vec(x)
    y = C.vec(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of parts")
    if any(v <= 0 for v in x) or any(v <= 0 for v in y):
        raise ValueError("compositions must be strictly positive")
    D = len(x)
    Lx = [math.log(v) for v in x]
    Ly = [math.log(v) for v in y]
    mx = sum(Lx) / D
    my = sum(Ly) / D
    d2 = sum(((Lx[i] - mx) - (Ly[i] - my)) ** 2 for i in range(D))
    pw = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            pw += ((Lx[i] - Lx[j]) - (Ly[i] - Ly[j])) ** 2
    return RichResult(payload={
        "distance": math.sqrt(d2), "distance_pairwise": math.sqrt(pw / D),
        "D": D, "method": "Aitchison distance"})


aitchison_distance = compdist


def cheatsheet():
    return "aitdst: d_a(x,y) = ||clr(x) - clr(y)||"
