"""Poisson-Stirling in centered variables k = a + x.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (5.17).
"""

import math

from . import _morin
from ._richresult import RichResult

__all__ = ["poisstirc"]


def poisstirc(x_dev, a):
    """Poisson-Stirling in centered variables k = a + x.

    Parameters
    ----------
    x_dev : float
        Deviation from the mean; k is round(a + x_dev) and must be >= 1.
    a : float
        Expected count, > 0.

    Returns
    -------
    RichResult
        Keys: k, approx, exact.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (5.17).
    """
    k = int(round(float(a) + float(x_dev)))
    if k < 1:
        raise ValueError("a + x must round to k >= 1")
    approx = _morin.poisson_stirling(k, a)
    exact = _morin.poisson_pmf(k, a)
    payload = {"k": k, "approx": approx, "exact": exact}
    lines = [("PP(a+x)", approx)]
    return RichResult(
        title="Poisson-Stirling in centered variables k = a + x.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poisstirc: Stirling Poisson pmf at k = round(a + x). Morin (2016) eq (5.17)."
