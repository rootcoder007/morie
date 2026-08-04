# morie.fn -- function file (rootcoder007/morie)
"""Aitchison norm of a composition."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compnorm', 'aitchison_norm', 'aitchisonnorm']


def compnorm(x):
    """Aitchison norm of a composition.

    Formula: ||x||_a = sqrt( (1/D) sum_{i<j} log(x_i/x_j)^2 )

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.

    Returns
    -------
    RichResult
        ``norm``, ``norm2``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  The norm associated with equation (10): ||x||_a^2 = <x, x>_a.
    """
    x = C.vec(x)
    D = len(x)
    if D < 2:
        raise ValueError("a norm on the simplex needs at least two parts")
    if any(v <= 0.0 for v in x):
        raise ValueError("compositions must be strictly positive")
    lx = [math.log(v) for v in x]
    tot = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            tot += (lx[i] - lx[j]) ** 2
    n2 = tot / D
    return RichResult(payload={
        "norm": math.sqrt(n2), "norm2": n2, "D": D,
        "method": "Aitchison norm"})


aitchison_norm = compnorm
aitchisonnorm = compnorm


def cheatsheet():
    return 'aitnrm: Aitchison norm of a composition.'
