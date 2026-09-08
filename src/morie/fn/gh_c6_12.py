# morie.fn -- function file (rootcoder007/morie)
"""Strong separation and the consistency it implies."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sepcons", "ghosal_sep_consist"]


def sepcons(delta, k, n):
    """Hellinger-affinity bound from strong delta-separation at stage k.

    Separation is the alternative route to consistency: instead of
    exhibiting a test, show the truth is separated from the alternative
    in affinity at SOME finite stage k, and the product structure does
    the rest.  The stage k enters as a divisor of n, so separating only
    at a large k gives a correspondingly slower exponential rate --
    which is the practical cost of not having a stage-1 separation.

    Formula: strongly delta-separated at stage k means
             rho_{1/2}(p_0^k, int p^k dmu(p)) < delta for every mu on V;
             then rho_{1/2}(p_0^n, int p^n dmu(p)) < e^{-(n/k) log_- delta}
             = delta^{n/k}

    Parameters
    ----------
    delta : float
        Separation level, 0 < delta < 1.
    k : int
        Stage at which separation holds, k >= 1.
    n : int
        Sample size.

    Returns
    -------
    RichResult
        ``bound`` (delta^{n/k}), ``rate`` (the exponent per
        observation), ``exponent``, ``delta``, ``k``, ``n``.

    References
    ----------
    Ghosal & van der Vaart (2017), Fundamentals of Nonparametric
    Bayesian Inference, Section 6.8.1: Definition 6.43 of strong
    delta-separation at stage k, Lemma 6.44 -- "If p_0 and V are
    strongly delta-separated at stage k, then rho_{1/2}(p_0^n,
    int p^n dmu(p)) < e^{-n/k log_- delta}" -- and Theorem 6.45, which
    concludes Pi(V | X_1, ..., X_n) -> 0 a.s. when p_0 is in the
    Kullback-Leibler support.  Read from the copy of the book held in
    the corpus.
    """
    d = float(delta)
    k = int(k)
    n = int(n)
    if not 0.0 < d < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1")
    if k < 1:
        raise ValueError("the stage k must be at least 1")
    if n < 1:
        raise ValueError("n must be at least 1")
    rate = -math.log(d) / k
    return RichResult(payload={
        "bound": math.exp(-rate * n), "rate": rate, "exponent": -rate * n,
        "delta": d, "k": float(k), "n": float(n),
        "method": "Strong separation bound, Ghosal Lemma 6.44"})


ghosal_sep_consist = sepcons


def cheatsheet():
    return "gh_c6_12: rho_{1/2} < delta^(n/k) from stage-k strong separation"
