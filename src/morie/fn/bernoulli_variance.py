"""Bernoulli variance p(1-p) = pq.

Implements eq (3.22) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["bernoulli_variance"]


def bernoulli_variance(p):
    """Bernoulli variance p(1-p) = pq.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.22).

    Examples
    --------
    >>> round(bernoulli_variance(0.3)["variance"], 12)
    0.21
    """
    value = _morin.bernoulli_variance(p)
    payload = {"p": float(p), "variance": value}
    lines = [("pq", value)]
    return RichResult(
        title="Bernoulli variance p(1-p) = pq.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e22: Bernoulli variance p(1-p) = pq. Morin (2016) eq (3.22)."
