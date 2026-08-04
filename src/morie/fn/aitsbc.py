# morie.fn -- function file (rootcoder007/morie)
"""Subcomposition: closure of a selected subset of parts."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["subcomp", "aitchison_subcomposition"]


def subcomp(x, parts, total=1.0):
    """Form the subcomposition on a selected subset of parts.

    Subcompositional coherence -- that an analysis of a subset gives
    the same answer whether or not the other parts were ever measured
    -- is the property Aitchison uses to rule out correlations of raw
    proportions.  The subcomposition is the operation that property is
    stated about.

    ``parts`` is a list of ONE-BASED part indices, matching the R arm.

    Formula: sub(x; S) = C( x_i : i in S )

    Parameters
    ----------
    x : array-like
        Strictly positive vector of parts.
    parts : sequence of int
        One-based indices of the parts retained (at least two).
    total : float
        Constant the subcomposition sums to.

    Returns
    -------
    RichResult
        ``composition``, ``parts``, ``total``, ``D_sub``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 2, where the subcomposition is defined as the closure of
    the selected subvector.
    """
    x = C.vec(x)
    if any(v <= 0 for v in x):
        raise ValueError("compositions must be strictly positive")
    D = len(x)
    idx = [int(i) for i in parts]
    if len(idx) < 2:
        raise ValueError("a subcomposition needs at least two parts")
    if any(not 1 <= i <= D for i in idx):
        raise ValueError("parts must be one-based indices in 1..D")
    sub = [x[i - 1] for i in idx]
    s = sum(sub)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in sub], "parts": idx, "total": k,
        "D_sub": len(idx), "D": D, "method": "Subcomposition"})


aitchison_subcomposition = subcomp


def cheatsheet():
    return "aitsbc: sub(x; S) = C(x_S)"
