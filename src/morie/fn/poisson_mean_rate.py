"""Expected event count in time t equals lambda t (series-checked).

Implements eq (4.19) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["poisson_mean_rate"]


def poisson_mean_rate(lam, t):
    """Expected event count in time t equals lambda t (series-checked).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.19).

    Examples
    --------
    >>> round(poisson_mean_rate(2.5, 4.0)["expected_events"], 12)
    10.0
    """
    value = _morin.poisson_mean_rate(lam, t)
    payload = {"lambda": float(lam), "t": float(t), "expected_events": value}
    lines = [("lambda t", value)]
    return RichResult(
        title="Expected event count in time t equals lambda t (series-checked).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e19: Expected event count in time t equals lambda t (series-checked). Morin (2016) eq (4.19)."
