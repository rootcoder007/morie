# morie.fn -- function file (rootcoder007/morie)
"""Tsallis q-entropy of a supplied pmf."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tsallis_entropy"]


def tsallis_entropy(p, q):
    """Tsallis entropy of a probability vector that is already known.

    The sample-based sibling has to estimate the pmf first; this one
    takes it.  The vector is renormalised on entry, because a pmf that
    sums to 0.999 through rounding would otherwise silently shift the
    entropy, and a caller who passes counts rather than probabilities
    gets the answer they meant.

    Formula: ``S_q = (1 - sum_i p_i^q) / (q - 1)``.

    Parameters
    ----------
    p : array-like
        Non-negative weights; renormalised to sum to one.
    q : float
        Entropic index.  ``q = 1`` gives Shannon entropy in nats.

    Returns
    -------
    RichResult
        ``estimate``, ``k`` (support size), ``q``.

    References
    ----------
    Tsallis, C. (1988).  Possible generalization of Boltzmann-Gibbs
    statistics.  Journal of Statistical Physics 52:479-487, equation
    (1).
    """
    v = C.vec(p)
    s = sum(v)
    pp = [t / s for t in v]
    q = float(q)
    if q == 1.0:
        val = -sum(t * math.log(t) for t in pp if t > 0)
    else:
        val = (1.0 - sum(t ** q for t in pp)) / (q - 1.0)
    return RichResult(payload={
        "estimate": val, "k": len(pp), "q": q,
        "method": "Tsallis q-entropy of a supplied pmf"})


def cheatsheet():
    return "tsalls: Tsallis q-entropy of a supplied pmf."
