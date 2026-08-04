# morie.fn -- function file (rootcoder007/morie)
"""Closure of a composition to a constant sum."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compclose', 'aitchison_closure']


def compclose(x, total=1.0):
    """Closure of a composition to a constant sum.

    Formula: C(x)_i = kappa * x_i / sum_j x_j

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.
    total : float
        Constant kappa the closure sums to.

    Returns
    -------
    RichResult
        ``composition``, ``total``, ``sum_raw``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  The closure operator C normalises any positive vector to the constant sum kappa that defines the simplex S^D.
    """
    x = C.vec(x)
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    if any(v <= 0.0 for v in x):
        raise ValueError("compositions must be strictly positive")
    s = sum(x)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in x], "total": k, "sum_raw": s,
        "D": len(x), "method": "Closure to a constant sum"})


aitchison_closure = compclose


def cheatsheet():
    return 'aitclos: Closure of a composition to a constant sum.'
