# morie.fn -- function file (rootcoder007/morie)
"""Randomized response as local differential privacy."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["randomized_response"]


def randomized_response(bit, epsilon=1.0):
    """The same mechanism, read as a differential-privacy guarantee.

    Warner 1965 mechanism turns out to be exactly the randomised
    response that gives local differential privacy, and the flip
    probability that achieves ``epsilon`` is
    ``1 / (1 + e^epsilon)``.  Reading it that way makes the privacy
    budget explicit: a small ``epsilon`` means a flip probability near
    one half, and therefore a variance that grows as the guarantee
    tightens.

    Determinism: the flips come from the shared Lehmer minstd stream, so
    a given seed reproduces the same released bits in both language
    arms.

    Formula: release ``b`` unchanged with probability
    ``e^eps / (1 + e^eps)``, flipped otherwise; the debiased rate is
    ``(mean(released) - q) / (1 - 2q)`` with ``q = 1 / (1 + e^eps)``.

    Parameters
    ----------
    bit : array-like
        True bits.
    epsilon : float, default 1.0
        Privacy budget.

    Returns
    -------
    RichResult
        ``estimate`` (debiased rate), ``released``, ``q`` (flip
        probability), ``raw_rate``, ``true_rate``, ``n``.

    References
    ----------
    Warner, S. L. (1965).  JASA 60:63-69.  The differential-privacy
    reading is Dwork, C. & Roth, A. (2014), The Algorithmic Foundations
    of Differential Privacy, Foundations and Trends in Theoretical
    Computer Science 9:211-407, section 3.2.
    """
    b = C.vec(bit)
    n = len(b)
    eps = float(epsilon)
    q = 1.0 / (1.0 + math.exp(eps))
    g = C.Lcg(1)
    rel = [(1.0 - b[i]) if g.unif() < q else b[i] for i in range(n)]
    raw = sum(rel) / n
    return RichResult(payload={
        "estimate": (raw - q) / (1.0 - 2.0 * q), "released": rel, "q": q,
        "raw_rate": raw, "true_rate": sum(b) / n, "n": n,
        "method": "Randomized response under local differential privacy"})


def cheatsheet():
    return "rrand: Randomized response as local differential privacy."
