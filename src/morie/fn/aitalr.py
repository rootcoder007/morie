# morie.fn -- function file (rootcoder007/morie)
"""Additive log-ratio (alr) transform with a reference part."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["alr", "aitchison_alr"]


def alr(x, ref=None):
    """Additive log-ratio transform against a chosen reference part.

    Unlike the clr the alr is a bijection onto all of R^(D-1), so it
    inverts cleanly and its covariance matrix is non-singular -- which
    is why Aitchison builds the logistic-normal on it.  It is not an
    isometry, though: distances computed in alr coordinates depend on
    which part was chosen as reference, so use clr or ilr for anything
    geometric.

    ``ref`` is a ONE-BASED part index, matching the R arm exactly; the
    remaining parts keep their original order.

    Formula: alr(x)_i = log( x_i / x_ref ),  i != ref

    Parameters
    ----------
    x : array-like
        Strictly positive vector of parts.
    ref : int, optional
        One-based index of the reference part (default: the last, D).

    Returns
    -------
    RichResult
        ``alr``, ``ref``, ``kept``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 4.  Verified against the reference implementation in the
    CRAN package ``compositions`` 2.0-9, whose ``alr`` defaults its
    ``ivar`` to the last column and returns log(x_i) - log(x_ivar).
    """
    x = C.vec(x)
    if any(v <= 0 for v in x):
        raise ValueError("compositions must be strictly positive")
    D = len(x)
    r = D if ref is None else int(ref)
    if not 1 <= r <= D:
        raise ValueError("ref must be a one-based part index in 1..D")
    lr = math.log(x[r - 1])
    keep = [i for i in range(D) if i != r - 1]
    return RichResult(payload={
        "alr": [math.log(x[i]) - lr for i in keep],
        "ref": r, "kept": [i + 1 for i in keep], "D": D,
        "method": "Additive log-ratio transform"})


aitchison_alr = alr


def cheatsheet():
    return "aitalr: alr(x)_i = log(x_i / x_ref)"
