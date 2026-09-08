"""Intersection of k independent events of common probability p: p^k.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (2.93)-(2.96).
"""

from ._richresult import RichResult

__all__ = ["piidint"]


def piidint(p, k=2):
    """P(all of k independent events, each of probability p) = p^k.

    The k = 2 and k = 3 cases are the pairwise and triple intersection
    terms of the book's inclusion-exclusion dice example.

    Parameters
    ----------
    p : float
        Common event probability, in [0, 1].
    k : int
        Number of events, >= 0.

    Returns
    -------
    RichResult
        Keys: p, k, p_intersection.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (2.93)-(2.96).
    """
    p_f = float(p)
    if not 0.0 <= p_f <= 1.0:
        raise ValueError("p must be in [0, 1]")
    k_i = int(k)
    if k_i < 0:
        raise ValueError("k must be >= 0")
    value = p_f ** k_i
    payload = {"p": p_f, "k": k_i, "p_intersection": value}
    lines = [("P(all k events)", value)]
    return RichResult(
        title="Intersection of k i.i.d. independent events: p^k.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "piidint: P(all of k independent events, each p) = p^k. Morin (2016) eqs (2.93)-(2.96)."
