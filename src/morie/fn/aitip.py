# morie.fn -- function file (rootcoder007/morie)
"""Aitchison inner product of two compositions."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compinner', 'aitchison_inner_product']


def compinner(x, y):
    """Aitchison inner product of two compositions.

    Formula: <x, y>_a = (1/D) sum_{i<j} log(x_i/x_j) log(y_i/y_j)

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.
    y : array-like
        Second composition, same length as x, strictly positive.

    Returns
    -------
    RichResult
        ``inner``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Equation (10) of the retrieved paper.  Computed in the printed pairwise form rather than through clr, so the implementation matches the display it cites; the two are algebraically identical.
    """
    x = C.vec(x); y = C.vec(y)
    D = len(x)
    if D != len(y):
        raise ValueError("x and y must have the same number of parts")
    if D < 2:
        raise ValueError("an inner product on the simplex needs at least two parts")
    if any(v <= 0.0 for v in x) or any(v <= 0.0 for v in y):
        raise ValueError("compositions must be strictly positive")
    lx = [math.log(v) for v in x]
    ly = [math.log(v) for v in y]
    tot = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            tot += (lx[i] - lx[j]) * (ly[i] - ly[j])
    return RichResult(payload={
        "inner": tot / D, "D": D, "method": "Aitchison inner product"})


aitchison_inner_product = compinner


def cheatsheet():
    return 'aitip: Aitchison inner product of two compositions.'
