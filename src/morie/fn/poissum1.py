"""The Poisson pmf sums to 1 via the exponential series.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.11).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["poissum1"]


def poissum1(a, kmax=200):
    """The Poisson pmf sums to 1 via the exponential series.

    Parameters
    ----------
    a : float
        Expected count, >= 0.
    kmax : int
        Truncation point, >= 1.

    Returns
    -------
    RichResult
        Keys: total, error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.11).
    """
    a_f = float(a)
    if a_f < 0:
        raise ValueError("a must be >= 0")
    total = sum(_morin.poisson_pmf(k, a_f) for k in range(int(kmax)))
    payload = {"total": total, "error": abs(total - 1.0)}
    lines = [("sum P(k)", total)]
    return RichResult(
        title="The Poisson pmf sums to 1 via the exponential series.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poissum1: The Poisson pmf sums to 1. Morin (2016) eq (7.11)."
