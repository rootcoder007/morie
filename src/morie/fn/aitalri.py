# morie.fn -- function file (rootcoder007/morie)
"""Inverse additive log-ratio transform."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["alrinv", "aitchison_alr_inverse"]


def alrinv(y, ref=None, total=1.0):
    """Map alr coordinates back to a closed composition.

    The reference part is reinstated as a zero log-ratio before the
    closure, which is what makes the alr a genuine bijection: no
    information is lost and no centring constant has to be guessed.

    ``ref`` is a ONE-BASED index into the RECONSTRUCTED composition of
    length D = len(y) + 1, matching the R arm exactly.

    Formula: alr^-1(y) = C( exp(y_1), ..., 1 at position ref, ... )

    Parameters
    ----------
    y : array-like
        alr coordinates, length D-1.
    ref : int, optional
        One-based position the reference part is reinstated at
        (default: the last, D).
    total : float
        Constant the returned composition sums to.

    Returns
    -------
    RichResult
        ``composition``, ``ref``, ``total``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 4.  Verified against the reference implementation in the
    CRAN package ``compositions`` 2.0-9, whose ``alrInv`` is
    ``clo(exp(cbind(z, 0)))`` -- a zero column appended, then closed.
    """
    y = C.vec(y)
    D = len(y) + 1
    r = D if ref is None else int(ref)
    if not 1 <= r <= D:
        raise ValueError("ref must be a one-based part index in 1..D")
    full = list(y)
    full.insert(r - 1, 0.0)
    e = [math.exp(v) for v in full]
    s = sum(e)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in e], "ref": r, "total": k, "D": D,
        "method": "Inverse additive log-ratio transform"})


aitchison_alr_inverse = alrinv


def cheatsheet():
    return "aitalri: alr^-1(y) = C(exp y with a 1 at ref)"
