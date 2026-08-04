# morie.fn -- function file (rootcoder007/morie)
"""Geometric mean of a composition."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compgmean', 'aitchison_geomean']


def compgmean(x):
    """Geometric mean of a composition.

    Formula: g(x) = (x_1 x_2 ... x_D)^(1/D) = exp( (1/D) sum_i log x_i )

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.

    Returns
    -------
    RichResult
        ``geomean``, ``log_geomean``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  g(x) is the geometric mean that the centred log-ratio divides by; computed through the log-mean so it does not overflow for large D.
    """
    x = C.vec(x)
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    if any(v <= 0.0 for v in x):
        raise ValueError("compositions must be strictly positive")
    lg = sum(math.log(v) for v in x) / len(x)
    return RichResult(payload={
        "geomean": math.exp(lg), "log_geomean": lg, "D": len(x),
        "method": "Geometric mean of a composition"})


aitchison_geomean = compgmean


def cheatsheet():
    return 'aitgmu: Geometric mean of a composition.'
