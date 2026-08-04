# morie.fn -- function file (rootcoder007/morie)
"""Aitchison norm of a composition."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["compnorm", "aitchison_norm"]


def compnorm(x):
    """Aitchison norm: the distance from a composition to the barycentre.

    It is the ordinary Euclidean norm of the clr coordinates, so it is
    zero exactly for the uniform composition C(1, ..., 1) and grows as
    the parts become more unequal.

    Formula: ||x||_a = sqrt( sum_i clr(x)_i^2 )
                     = sqrt( (1/D) sum_{i<j} log(x_i/x_j)^2 )

    Parameters
    ----------
    x : array-like
        Strictly positive vector of parts.

    Returns
    -------
    RichResult
        ``norm``, ``norm_pairwise``, ``D``.  The two agree; both are
        returned because their agreement is the cheapest check that the
        clr bookkeeping is right.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 4.  Verified against the reference implementation in the
    CRAN package ``compositions`` 2.0-9, whose ``norm`` on an
    ``acomp`` is sqrt(scalar(x, x)) with scalar the clr inner product.
    """
    x = C.vec(x)
    if any(v <= 0 for v in x):
        raise ValueError("compositions must be strictly positive")
    D = len(x)
    L = [math.log(v) for v in x]
    lg = sum(L) / D
    z = [v - lg for v in L]
    pw = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            pw += (L[i] - L[j]) ** 2
    return RichResult(payload={
        "norm": math.sqrt(sum(v * v for v in z)),
        "norm_pairwise": math.sqrt(pw / D), "D": D,
        "method": "Aitchison norm"})


aitchison_norm = compnorm


def cheatsheet():
    return "aitnrm: ||x||_a = sqrt(sum clr(x)^2)"
