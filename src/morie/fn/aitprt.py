# morie.fn -- function file (rootcoder007/morie)
"""Perturbation, the group operation of the simplex."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compperturb', 'aitchison_perturbation']


def compperturb(x, y, total=1.0):
    """Perturbation, the group operation of the simplex.

    Formula: x (+) y = C( x_1 y_1, x_2 y_2, ..., x_D y_D )

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.
    y : array-like
        Second composition, same length as x, strictly positive.
    total : float
        Constant kappa the closure sums to.

    Returns
    -------
    RichResult
        ``composition``, ``total``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Perturbation is the inner sum of the simplex: componentwise multiplication followed by closure.  It plays the role that addition plays in real space, which is why translations of compositional data are perturbations.
    """
    x = C.vec(x); y = C.vec(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of parts")
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    if any(v <= 0.0 for v in x) or any(v <= 0.0 for v in y):
        raise ValueError("compositions must be strictly positive")
    p = [a * b for a, b in zip(x, y)]
    s = sum(p)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in p], "total": k, "D": len(x),
        "method": "Perturbation on the simplex"})


aitchison_perturbation = compperturb


def cheatsheet():
    return 'aitprt: Perturbation, the group operation of the simplex.'
