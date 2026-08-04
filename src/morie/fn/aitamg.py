# morie.fn -- function file (rootcoder007/morie)
"""Amalgamation of a composition into grouped parts."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compamalg', 'aitchison_amalgamation']


def compamalg(x, groups, total=1.0):
    """Amalgamation of a composition into grouped parts.

    Formula: amalg(x; g)_k = sum_{i : g_i = k} x_i, then closed

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.
    groups : array-like of int
        Group label 1..k for each part of x, same length as x; every label from 1 to k must be used.
    total : float
        Constant kappa the closure sums to.

    Returns
    -------
    RichResult
        ``composition``, ``raw``, ``k``, ``total``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read; see EXTERNAL_SOURCES.md.  Amalgamation adds parts together, unlike subcomposition which drops them.  It is the operation that is NOT subcompositionally coherent: log-ratios among the amalgamated groups are not determined by the log-ratios among the original parts, which is why the two operations are kept as separate functions here.  Group labels are 1-based in BOTH language arms.  Implemented in the standard published form.  The log-ratio algebra it rests on was verified against Mateu-Figueras, Pawlowsky-Glahn and Egozcue, arXiv:0802.2643 Sect. 4.1 (fetched and archived), but this particular definition is not printed there and could not be checked against Aitchison's own text.
    """
    x = C.vec(x)
    g = [int(v) for v in groups]
    if len(g) != len(x):
        raise ValueError("groups must have one label per part of x")
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    if any(v <= 0.0 for v in x):
        raise ValueError("compositions must be strictly positive")
    k = max(g)
    if min(g) != 1 or sorted(set(g)) != list(range(1, k + 1)):
        raise ValueError("group labels must be 1..k with every label used")
    raw = [0.0] * k
    for lab, v in zip(g, x):
        raw[lab - 1] += v
    s = sum(raw)
    t = float(total)
    return RichResult(payload={
        "composition": [t * v / s for v in raw], "raw": raw, "k": k,
        "total": t, "D": len(x), "method": "Amalgamation"})


aitchison_amalgamation = compamalg


def cheatsheet():
    return 'aitamg: Amalgamation of a composition into grouped parts.'
