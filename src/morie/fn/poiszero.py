"""Poisson P(0) = e^(-a): the typo-free-page worked example.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.99).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["poiszero"]


def poiszero(a=7.0):
    """Poisson P(0) = e^(-a): the typo-free-page worked example.

    The default a = 7 is the book's typos-per-page example, giving
    about a 0.1 percent chance of a clean page.

    Parameters
    ----------
    a : float
        Expected count, >= 0.

    Returns
    -------
    RichResult
        Keys: a, p_zero.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.99).
    """
    value = _morin.poisson_pmf(0, a)
    payload = {"a": float(a), "p_zero": value}
    lines = [("P(0)", value)]
    return RichResult(
        title="Poisson P(0) = e^(-a): the typo-free-page worked example.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poiszero: P(0) = e^(-a) for a Poisson count. Morin (2016) eq (4.99)."
