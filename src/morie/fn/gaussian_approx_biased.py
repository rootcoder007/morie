"""Gaussian approximation for n biased flips, centered at pn.

Implements eq (5.15) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["gaussian_approx_biased"]


def gaussian_approx_biased(x, n, p):
    """Gaussian approximation for n biased flips, centered at pn.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.15).

    Examples
    --------
    >>> round(gaussian_approx_biased(4, 100, 0.3)["PG"], 12)
    0.05947780073
    """
    value = _morin.gaussian_approx_biased(x, n, p)
    payload = {"x": float(x), "n": int(n), "p": float(p), "PG": value}
    lines = [("PG(x)", value)]
    return RichResult(
        title="Gaussian approximation for n biased flips, centered at pn.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e15: Gaussian approximation for n biased flips, centered at pn. Morin (2016) eq (5.15)."
