# morie.fn -- function file (rootcoder007/morie)
"""Geometric mean of the parts of a composition."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["compgeo", "aitchison_geomean"]


def compgeo(x):
    """Geometric mean of the parts of a single composition.

    This is the divisor that makes the centred log-ratio transform
    centred; every other quantity in the log-ratio toolkit is built on
    it.  It is computed in logs, not as a product, because the product
    of a few hundred parts underflows long before the mean does.

    Formula: g(x) = (x_1 x_2 ... x_D)^(1/D) = exp( (1/D) sum_j log x_j )

    Parameters
    ----------
    x : array-like
        Strictly positive vector of parts.

    Returns
    -------
    RichResult
        ``geomean``, ``log_geomean``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 4, where g(x) is the normalising constant of the centred
    log-ratio transform.
    """
    x = C.vec(x)
    if any(v <= 0 for v in x):
        raise ValueError("compositions must be strictly positive")
    D = len(x)
    lg = sum(math.log(v) for v in x) / D
    return RichResult(payload={
        "geomean": math.exp(lg), "log_geomean": lg, "D": D,
        "method": "Geometric mean of a composition"})


aitchison_geomean = compgeo


def cheatsheet():
    return "aitgmu: g(x) = exp(mean(log x))"
