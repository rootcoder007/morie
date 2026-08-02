"""Difference quotient of x^2: ((x+d)^2 - x^2)/d = 2x + d.

Implements eq (7.31) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_31"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_31(x, delta):
    """Difference quotient of x^2: ((x+d)^2 - x^2)/d = 2x + d.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.31).
    """
    quotient, derivative = _morin.power_derivative_quotient(x, 2, delta)
    explicit = 2.0 * float(x) + float(delta)
    if abs(quotient - explicit) > 1e-9 * max(1.0, abs(explicit)):
        raise AssertionError("quotient != 2x + delta")
    payload = {"quotient": quotient, "derivative_limit": derivative}
    lines = [("(f(x+d)-f(x))/d", quotient), ("2x", derivative)]
    return RichResult(
        title="Difference quotient of x^2: ((x+d)^2 - x^2)/d = 2x + d.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e31: Difference quotient of x^2: ((x+d)^2 - x^2)/d = 2x + d. Morin (2016) eq (7.31)."
