# morie.fn -- function file (rootcoder007/morie)
"""Inverse centred log-ratio transform."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["clrinv", "aitchison_clr_inverse"]


def clrinv(z, total=1.0):
    """Map clr coordinates back to a closed composition.

    The inverse does not require ``sum(z) == 0``: the closure absorbs
    any constant shift, so clr(clrinv(z)) is the centred version of z
    rather than z itself.  ``shift`` reports the constant that was
    absorbed, which is zero exactly when the input was a genuine clr
    vector.

    Formula: clr^-1(z) = C( exp(z_1), ..., exp(z_D) )

    Parameters
    ----------
    z : array-like
        clr coordinates.
    total : float
        Constant the returned composition sums to.

    Returns
    -------
    RichResult
        ``composition``, ``shift``, ``total``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 4.  Verified against the reference implementation in the
    CRAN package ``compositions`` 2.0-9, whose ``clrInv`` is
    ``acomp(exp(z))``, i.e. the closure of the exponentiated argument.
    """
    z = C.vec(z)
    D = len(z)
    shift = sum(z) / D
    e = [math.exp(v - shift) for v in z]
    s = sum(e)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in e], "shift": shift, "total": k,
        "D": D, "method": "Inverse centred log-ratio transform"})


aitchison_clr_inverse = clrinv


def cheatsheet():
    return "aitclri: clr^-1(z) = C(exp z)"
