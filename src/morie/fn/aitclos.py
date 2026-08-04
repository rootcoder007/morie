# morie.fn -- function file (rootcoder007/morie)
"""Closure of a vector onto the simplex."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["compclos", "aitchison_closure"]


def compclos(x, total=1.0):
    """Closure operator C(x) onto the simplex of constant sum ``total``.

    Compositional data carry only relative information, so the first
    thing any log-ratio method does is throw the scale away: two
    vectors differing by a positive factor are the same composition.
    The closure is that quotient made concrete.

    Formula: C(x) = kappa * (x_1, ..., x_D) / sum_j x_j

    Parameters
    ----------
    x : array-like
        Strictly positive vector of parts.
    total : float
        Constant kappa the closed vector sums to (1 or 100 in practice).

    Returns
    -------
    RichResult
        ``closed``, ``total``, ``sum_raw``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 2.  Definition verified against the reference
    implementation in the CRAN package ``compositions`` 2.0-9
    (van den Boogaart & Tolosana-Delgado), whose ``clo``/``acomp``
    divide by the row sum and rescale to the requested total.
    """
    x = C.vec(x)
    if any(v <= 0 for v in x):
        raise ValueError("compositions must be strictly positive")
    s = sum(x)
    k = float(total)
    return RichResult(payload={
        "closed": [k * v / s for v in x], "total": k, "sum_raw": s,
        "D": len(x), "method": "Closure C(x) onto the simplex"})


aitchison_closure = compclos


def cheatsheet():
    return "aitclos: C(x) = kappa x / sum(x)"
