# morie.fn -- function file (rootcoder007/morie)
"""Tsallis q-entropy of a sample."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tsallis_entropy"]


def tsallis_entropy(y, q):
    """Tsallis entropy of the empirical distribution of a sample.

    Shannon entropy is additive: independent systems add.  Tsallis
    replaces the logarithm with a power, and the price of the extra
    parameter is that entropies of independent systems no longer just
    add -- they pick up a ``(1 - q) S_A S_B`` cross term.  That is the
    point rather than a defect; the parameter buys long-range
    correlated systems a workable entropy.  As ``q -> 1`` the
    expression tends to Shannon, and this returns that limit exactly at
    ``q = 1`` rather than dividing by zero.

    Formula: ``S_q = (1 / (q - 1)) (1 - sum_x p(x)^q)``.

    Parameters
    ----------
    y : array-like
        Sample; the empirical pmf over distinct values is used.
    q : float
        Entropic index.  ``q = 1`` gives Shannon entropy in nats.

    Returns
    -------
    RichResult
        ``estimate``, ``n_categories``, ``n``, ``q``.

    References
    ----------
    Tsallis, C. (1988).  Possible generalization of Boltzmann-Gibbs
    statistics.  Journal of Statistical Physics 52:479-487, equation
    (1).
    """
    v = C.vec(y)
    n = len(v)
    counts = {}
    for t in v:
        counts[t] = counts.get(t, 0) + 1
    p = [counts[k] / n for k in sorted(counts)]
    q = float(q)
    if q == 1.0:
        s = -sum(t * math.log(t) for t in p if t > 0)
    else:
        s = (1.0 - sum(t ** q for t in p)) / (q - 1.0)
    return RichResult(payload={
        "estimate": s, "n_categories": len(p), "n": n, "q": q,
        "method": "Tsallis q-entropy of the empirical pmf"})


tsallisentropy = tsallis_entropy


def cheatsheet():
    return "tsallen: Tsallis q-entropy of a sample."
