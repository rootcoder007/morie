"""Binomial peak value PB(pn) at k = pn.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.96).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["binompeak"]


def binompeak(n, p):
    """Binomial peak value PB(pn) at k = pn.

    Parameters
    ----------
    n : int
        Trials, >= 0.
    p : float
        Success probability, in [0, 1].

    Returns
    -------
    RichResult
        Keys: k, PB.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.96).
    """
    k = int(round(float(p) * int(n)))
    value = _morin.binomial_pmf(k, n, p)
    payload = {"k": k, "PB": value}
    lines = [("PB(pn)", value)]
    return RichResult(
        title="Binomial peak value PB(pn) at k = pn.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binompeak: Binomial peak height at k = pn. Morin (2016) eq (4.96)."
