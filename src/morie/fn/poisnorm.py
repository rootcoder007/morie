"""Poisson normalization from the exponential series: e^a e^(-a) = 1.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.10).
"""

import math

from . import _morin

from ._richresult import RichResult

__all__ = ["poisnorm"]


def poisnorm(a, terms=40):
    """Poisson normalization from the exponential series: e^a e^(-a) = 1.

    Parameters
    ----------
    a : float
        Series argument.
    terms : int
        Number of Taylor terms, >= 0.

    Returns
    -------
    RichResult
        Keys: normalization, error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.10).
    """
    partials, closed = _morin.exp_taylor(a, terms)
    total = partials[-1] * math.exp(-float(a))
    payload = {"normalization": total, "error": abs(total - 1.0)}
    lines = [("sum P(k)", total)]
    return RichResult(
        title="Poisson normalization from the exponential series: e^a e^(-a) = 1.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poisnorm: Truncated e^a series times e^-a, which is 1. Morin (2016) eq (7.10)."
