# morie.fn -- function file (rootcoder007/morie)
"""Powering, the scalar multiplication of the simplex."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['comppower', 'aitchison_powering']


def comppower(x, a, total=1.0):
    """Powering, the scalar multiplication of the simplex.

    Formula: a (.) x = C( x_1^a, x_2^a, ..., x_D^a )

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.
    a : float
        Real scalar.
    total : float
        Constant kappa the closure sums to.

    Returns
    -------
    RichResult
        ``composition``, ``alpha``, ``total``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Powering is the outer product of the simplex: raise every part to the power a and close.  It plays the role that multiplication by a scalar plays in real space, and satisfies d_a(a(.)x, a(.)x*) = |a| d_a(x, x*).
    """
    x = C.vec(x)
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    if any(v <= 0.0 for v in x):
        raise ValueError("compositions must be strictly positive")
    a = float(a)
    lg = [a * math.log(v) for v in x]
    m = max(lg)
    e = [math.exp(v - m) for v in lg]
    s = sum(e)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in e], "alpha": a, "total": k,
        "D": len(x), "method": "Powering on the simplex"})


aitchison_powering = comppower


def cheatsheet():
    return 'aitpow: Powering, the scalar multiplication of the simplex.'
