"""P(0) = e^(-a) as the alternating exponential series.

Implements eq (4.53) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["poisson_zero_series"]


def poisson_zero_series(a, terms=60):
    """P(0) = e^(-a) as the alternating exponential series.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.53).

    Examples
    --------
    >>> round(poisson_zero_series(1.5, 20)["partial_sums"][-1], 12)
    0.223130160148
    """
    partials, closed = _morin.poisson_zero_series(a, terms)
    payload = {"partial_sums": partials, "e_minus_a": closed, "final_error": abs(partials[-1] - closed)}
    lines = [("series", partials[-1]), ("e^-a", closed)]
    return RichResult(
        title="P(0) = e^(-a) as the alternating exponential series.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e53: P(0) = e^(-a) as the alternating exponential series. Morin (2016) eq (4.53)."
