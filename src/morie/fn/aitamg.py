# morie.fn -- function file (rootcoder007/morie)
"""Amalgamation: sum a subset of parts into a single part."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["amalgam", "aitchison_amalgamation"]


def amalgam(x, parts, total=1.0):
    """Amalgamate a subset of parts into one, keeping the rest.

    Amalgamation is the other way to reduce the dimension of a
    composition, and unlike subcomposition it is NOT a log-ratio
    operation: it adds raw parts, so it does not commute with
    perturbation and it is not subcompositionally coherent.  It is
    included because real classification schemes amalgamate all the
    time, not because the geometry likes it.

    The amalgamated part is appended last; the retained parts keep
    their original order.  ``parts`` is ONE-BASED, matching the R arm.

    Formula: amalg(x; S) = C( (x_i : i not in S), sum_{j in S} x_j )

    Parameters
    ----------
    x : array-like
        Strictly positive vector of parts.
    parts : sequence of int
        One-based indices of the parts summed together.
    total : float
        Constant the result sums to.

    Returns
    -------
    RichResult
        ``composition``, ``amalgamated``, ``parts``, ``kept``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 2, which defines amalgamation and notes that it does not
    preserve the log-ratio structure.
    """
    x = C.vec(x)
    if any(v <= 0 for v in x):
        raise ValueError("compositions must be strictly positive")
    D = len(x)
    idx = [int(i) for i in parts]
    if len(idx) < 2:
        raise ValueError("an amalgamation needs at least two parts")
    if any(not 1 <= i <= D for i in idx):
        raise ValueError("parts must be one-based indices in 1..D")
    sel = set(idx)
    keep = [i for i in range(1, D + 1) if i not in sel]
    amal = sum(x[i - 1] for i in idx)
    raw = [x[i - 1] for i in keep] + [amal]
    s = sum(raw)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in raw], "amalgamated": amal,
        "parts": idx, "kept": keep, "D": len(raw),
        "method": "Amalgamation"})


aitchison_amalgamation = amalgam


def cheatsheet():
    return "aitamg: amalg(x; S) = C(x_notS, sum x_S)"
