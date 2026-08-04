"""Binomial pmf for n rolls of a b-sided die.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.32).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["binomdie"]


def binomdie(k, n, b):
    """Binomial pmf for n rolls of a b-sided die.

    Success probability p = 1/b.

    Parameters
    ----------
    k : int
        Successes, >= 0.
    n : int
        Rolls, >= 0.
    b : float
        Number of faces, >= 1.

    Returns
    -------
    RichResult
        Keys: k, n, p, probability.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.32).
    """
    b_f = float(b)
    if b_f < 1:
        raise ValueError("b must be >= 1")
    value = _morin.binomial_pmf(k, n, 1.0 / b_f)
    payload = {"k": int(k), "n": int(n), "p": 1.0 / b_f, "probability": value}
    lines = [("P(k)", value)]
    return RichResult(
        title="Binomial pmf for n rolls of a b-sided die.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binomdie: Binomial pmf with p = 1/b for a b-sided die. Morin (2016) eq (4.32)."
