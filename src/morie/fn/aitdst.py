# morie.fn -- function file (rootcoder007/morie)
"""Aitchison distance between two compositions."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compdist', 'aitchison_distance']


def compdist(x, y):
    """Aitchison distance between two compositions.

    Formula: d_a(x, y)^2 = (1/D) sum_{i<j} ( log(x_i/x_j) - log(y_i/y_j) )^2

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.
    y : array-like
        Second composition, same length as x, strictly positive.

    Returns
    -------
    RichResult
        ``distance``, ``distance2``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  The squared distance printed immediately after equation (10).  It is relative: it is unchanged by perturbing both arguments by the same composition, and scales by |a| under powering.
    """
    x = C.vec(x); y = C.vec(y)
    D = len(x)
    if D != len(y):
        raise ValueError("x and y must have the same number of parts")
    if D < 2:
        raise ValueError("a distance on the simplex needs at least two parts")
    if any(v <= 0.0 for v in x) or any(v <= 0.0 for v in y):
        raise ValueError("compositions must be strictly positive")
    lx = [math.log(v) for v in x]
    ly = [math.log(v) for v in y]
    tot = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            tot += ((lx[i] - lx[j]) - (ly[i] - ly[j])) ** 2
    d2 = tot / D
    return RichResult(payload={
        "distance": math.sqrt(d2), "distance2": d2, "D": D,
        "method": "Aitchison distance"})


aitchison_distance = compdist


def cheatsheet():
    return 'aitdst: Aitchison distance between two compositions.'
