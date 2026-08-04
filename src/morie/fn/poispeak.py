"""Poisson peak value PP(pn) = (pn)^(pn) e^(-pn) / (pn)!.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.95).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["poispeak"]


def poispeak(n, p):
    """Poisson peak value PP(pn) = (pn)^(pn) e^(-pn) / (pn)!.

    Parameters
    ----------
    n : int
        Trials, >= 0.
    p : float
        Success probability, in [0, 1].

    Returns
    -------
    RichResult
        Keys: k, PP.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.95).
    """
    k = int(round(float(p) * int(n)))
    value = _morin.poisson_pmf(k, float(p) * int(n))
    payload = {"k": k, "PP": value}
    lines = [("PP(pn)", value)]
    return RichResult(
        title="Poisson peak value PP(pn) = (pn)^(pn) e^(-pn) / (pn)!.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poispeak: Poisson peak height at k = pn. Morin (2016) eq (4.95)."
