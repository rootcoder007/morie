# morie.fn -- function file (rootcoder007/morie)
"""Inverse additive log-ratio transform."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['alrinv', 'aitchison_alr_inverse']


def alrinv(y, ref=None, total=1.0):
    """Inverse additive log-ratio transform.

    Formula: alr^-1(y) = C( exp(y_1), ..., exp(y_{D-1}), 1 ) with the 1 inserted at the reference position

    Parameters
    ----------
    y : array-like
        Additive log-ratio coordinates, length D - 1.
    ref : int
        1-based index the reference part is restored to; the default is the last position D.
    total : float
        Constant kappa the closure sums to.

    Returns
    -------
    RichResult
        ``composition``, ``ref``, ``total``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The definitions below were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sect. 4.1, which prints them with equation numbers and attributes them to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Inverting alr puts a 1 back in the reference slot and closes: the reference part is the one whose log-ratio against itself is zero.  ``ref`` must match the ``ref`` used in the forward transform for the round trip to return the original composition.
    """
    y = C.vec(y)
    D = len(y) + 1
    if len(y) == 0:
        raise ValueError("y must be non-empty")
    k = D if ref is None else int(ref)
    if not 1 <= k <= D:
        raise ValueError("ref must be a 1-based part index")
    full = [0.0] * D
    idx = [i for i in range(1, D + 1) if i != k]
    for pos, i in enumerate(idx):
        full[i - 1] = y[pos]
    m = max(full)
    e = [math.exp(v - m) for v in full]
    s = sum(e)
    t = float(total)
    return RichResult(payload={
        "composition": [t * v / s for v in e], "ref": k, "total": t, "D": D,
        "method": "Inverse additive log-ratio transform"})


aitchison_alr_inverse = alrinv


def cheatsheet():
    return 'aitalri: Inverse additive log-ratio transform.'
