# morie.fn -- function file (rootcoder007/morie)
"""Centred log-ratio transform of a composition."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['clr', 'aitchison_clr', 'aitchisonclr']


def clr(x):
    """Centred log-ratio transform of a composition.

    Formula: clr(x)_i = log( x_i / g(x) ),  g(x) the geometric mean; sum_i clr(x)_i = 0

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.

    Returns
    -------
    RichResult
        ``clr``, ``geomean``, ``D``, ``sum_check``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  clr(x) = (ln(x_1/g(x)), ..., ln(x_D/g(x))).  The transform maps S^D onto the hyperplane of vectors summing to zero, so ``sum_check`` is zero up to rounding and is reported as a check on the caller's data rather than as a result.
    """
    x = C.vec(x)
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    if any(v <= 0.0 for v in x):
        raise ValueError("compositions must be strictly positive")
    lg = sum(math.log(v) for v in x) / len(x)
    z = [math.log(v) - lg for v in x]
    return RichResult(payload={
        "clr": z, "geomean": math.exp(lg), "D": len(x), "sum_check": sum(z),
        "method": "Centred log-ratio transform"})


aitchison_clr = clr
aitchisonclr = clr


def cheatsheet():
    return 'aitclr: Centred log-ratio transform of a composition.'
