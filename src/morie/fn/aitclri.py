# morie.fn -- function file (rootcoder007/morie)
"""Inverse centred log-ratio transform."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['clrinv', 'aitchison_clr_inverse']


def clrinv(z, total=1.0):
    """Inverse centred log-ratio transform.

    Formula: clr^-1(z) = C( exp(z_1), ..., exp(z_D) )

    Parameters
    ----------
    z : array-like
        Centred log-ratio coordinates, length D.
    total : float
        Constant kappa the closure sums to.

    Returns
    -------
    RichResult
        ``composition``, ``total``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  The inverse of clr is the closure of the exponentiated coordinates.  It is well defined for any real z, not only for z summing to zero: adding a constant to every z_i leaves the closed result unchanged.
    """
    z = C.vec(z)
    if len(z) == 0:
        raise ValueError("z must be non-empty")
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in e], "total": k, "D": len(z),
        "method": "Inverse centred log-ratio transform"})


aitchison_clr_inverse = clrinv


def cheatsheet():
    return 'aitclri: Inverse centred log-ratio transform.'
