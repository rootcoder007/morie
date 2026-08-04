# morie.fn -- function file (rootcoder007/morie)
"""Centred log-ratio (clr) transform of a composition."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["clr", "aitchison_clr"]


def clr(x):
    """Centred log-ratio transform of a single composition.

    The clr sends the simplex isometrically onto the hyperplane of
    vectors summing to zero, so ordinary Euclidean geometry -- inner
    products, distances, principal components -- becomes available.
    The price is the singularity: the image is a D-1 dimensional
    subspace of R^D, so the clr covariance matrix is always singular.
    ``sum_clr`` is returned precisely so a caller can see it is zero.

    Formula: clr(x)_i = log( x_i / g(x) ),  g(x) = (prod_j x_j)^(1/D)

    Parameters
    ----------
    x : array-like
        Strictly positive vector of parts.

    Returns
    -------
    RichResult
        ``clr``, ``geomean``, ``sum_clr``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 4.  Verified against the reference implementation in the
    CRAN package ``compositions`` 2.0-9, whose ``clr`` computes
    ``LOG - rowSums(LOG)/D`` on the logged parts.
    """
    x = C.vec(x)
    if any(v <= 0 for v in x):
        raise ValueError("compositions must be strictly positive")
    D = len(x)
    L = [math.log(v) for v in x]
    lg = sum(L) / D
    z = [v - lg for v in L]
    return RichResult(payload={
        "clr": z, "geomean": math.exp(lg), "sum_clr": sum(z), "D": D,
        "method": "Centred log-ratio transform"})


aitchison_clr = clr


def cheatsheet():
    return "aitclr: clr(x)_i = log(x_i / g(x))"
