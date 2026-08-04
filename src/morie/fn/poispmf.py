"""Poisson distribution P(k) = a^k e^(-a) / k!.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.40).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["poispmf"]


def poispmf(k, a):
    """Poisson distribution P(k) = a^k e^(-a) / k!.

    Evaluated in log space so large k does not overflow.

    Parameters
    ----------
    k : int
        Event count, >= 0.
    a : float
        Expected count, >= 0.

    Returns
    -------
    RichResult
        Keys: k, a, probability.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.40).
    """
    value = _morin.poisson_pmf(k, a)
    payload = {"k": int(k), "a": float(a), "probability": value}
    lines = [("P(k)", value)]
    return RichResult(
        title="Poisson distribution P(k) = a^k e^(-a) / k!.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poispmf: Poisson pmf a^k e^-a / k!. Morin (2016) eq (4.40)."
