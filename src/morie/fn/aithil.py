# morie.fn -- function file (rootcoder007/morie)
"""Hill numbers of order q."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["hillq", "compositional_hill"]


def hillq(x, q=1.0):
    """Effective number of parts of a composition at order q.

    Hill's family of diversity numbers indexes how strongly a diversity
    measure weights common versus rare parts.  For proportions p_i
    obtained by closing x to unit sum,

        qD = ( sum_i p_i^q )^(1 / (1 - q)),   q != 1

    and, in the limit q -> 1, the exponential of Shannon entropy,

        1D = exp( - sum_i p_i log p_i ).

    Every member is an effective number of equally common parts: for a
    perfectly even composition of D parts, qD = D at every q.  The
    familiar indices are special cases -- 0D is richness, 2D is the
    inverse Simpson concentration.

    Parameters
    ----------
    x : array-like
        Non-negative abundances or proportions; closed internally.
    q : float
        Order of the diversity number.

    Returns
    -------
    RichResult
        ``hill``, ``q``, ``prop``, ``richness``, ``shannon``,
        ``simpson``, ``D``.

    References
    ----------
    Hill, M. O. (1973), "Diversity and evenness: a unifying notation and
    its consequences", Ecology 54(2), 427-432, whose Equation (2) defines
    N_a = (sum_i p_i^a)^(1/(1-a)) and whose Sect. 2 gives the q -> 1
    limit as the exponential of Shannon's index.  Standard published
    form; the Ecology article is paywalled and was not read for this
    implementation.
    """
    x = C.vec(x)
    D = len(x)
    if D == 0:
        raise ValueError("x must be non-empty")
    if any(v < 0.0 for v in x):
        raise ValueError("abundances must be non-negative")
    tot = sum(x)
    if tot <= 0.0:
        raise ValueError("abundances must not all be zero")
    p = [v / tot for v in x]
    pos = [v for v in p if v > 0.0]
    sh = -sum(v * math.log(v) for v in pos)
    si = sum(v * v for v in pos)
    q = float(q)
    if abs(q - 1.0) < 1e-12:
        h = math.exp(sh)
    else:
        h = sum(v ** q for v in pos) ** (1.0 / (1.0 - q))
    return RichResult(payload={
        "hill": h, "q": q, "prop": p, "richness": len(pos),
        "shannon": sh, "simpson": si, "D": D,
        "method": "Hill number of order q (Hill 1973 eq. 2)"})


compositional_hill = hillq


def cheatsheet():
    return "aithil: Hill numbers of order q."
