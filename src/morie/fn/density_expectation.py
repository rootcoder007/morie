"""Expectation of a continuous distribution: integral of x rho(x) dx.

Implements eq (4.55) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["density_expectation"]


def density_expectation(grid, density):
    """Expectation of a continuous distribution: integral of x rho(x) dx.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.55).

    Examples
    --------
    >>> round(density_expectation([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 0.25, 0.5, 0.75, 1.0])["expectation"], 12)
    1.375
    """
    value = _morin.density_expectation(grid, density)
    payload = {"expectation": value}
    lines = [("E(X)", value)]
    return RichResult(
        title="Expectation of a continuous distribution: integral of x rho(x) dx.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e55: Expectation of a continuous distribution: integral of x rho(x) dx. Morin (2016) eq (4.55)."
