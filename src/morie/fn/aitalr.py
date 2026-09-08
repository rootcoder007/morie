# morie.fn -- function file (rootcoder007/morie)
"""Additive log-ratio transform of a composition against a reference part."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['alr', 'aitchison_alr', 'aitchisonalr']


def alr(x, ref=None):
    """Additive log-ratio transform of a composition against a reference part.

    Formula: alr(x)_i = log( x_i / x_ref ),  i != ref

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.
    ref : int
        1-based index of the reference part; the default D uses the last part, as in Aitchison's own display.

    Returns
    -------
    RichResult
        ``alr``, ``ref``, ``parts``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  alr(x) = (ln(x_1/x_D), ..., ln(x_{D-1}/x_D)), the reference part being the last one.  ``ref`` generalises that to any part; ``parts`` reports which 1-based indices the returned coordinates belong to, in their original order.  The index is 1-based in BOTH language arms so the two agree.
    """
    x = C.vec(x)
    D = len(x)
    if D < 2:
        raise ValueError("an additive log-ratio needs at least two parts")
    if any(v <= 0.0 for v in x):
        raise ValueError("compositions must be strictly positive")
    k = D if ref is None else int(ref)
    if not 1 <= k <= D:
        raise ValueError("ref must be a 1-based part index")
    lr = math.log(x[k - 1])
    idx = [i for i in range(1, D + 1) if i != k]
    return RichResult(payload={
        "alr": [math.log(x[i - 1]) - lr for i in idx], "ref": k, "parts": idx,
        "D": D, "method": "Additive log-ratio transform"})


aitchison_alr = alr
aitchisonalr = alr


def cheatsheet():
    return 'aitalr: Additive log-ratio transform of a composition against a reference part.'
