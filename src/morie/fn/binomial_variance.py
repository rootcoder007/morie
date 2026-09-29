"""Variance of the number of Heads in n biased flips: npq.

Implements eq (3.33) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["binomial_variance"]


def binomial_variance(n, p):
    """Variance of the number of Heads in n biased flips: npq.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.33).

    Examples
    --------
    >>> round(binomial_variance(10, 0.3)["variance"], 12)
    2.1
    """
    value = _morin.binomial_variance(n, p)
    payload = {"n": int(n), "p": float(p), "variance": value}
    lines = [("npq", value)]
    return RichResult(
        title="Variance of the number of Heads in n biased flips: npq.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e33: Variance of the number of Heads in n biased flips: npq. Morin (2016) eq (3.33)."
