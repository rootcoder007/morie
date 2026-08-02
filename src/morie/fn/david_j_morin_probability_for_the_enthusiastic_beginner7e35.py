"""Binomial expansion of (x + d)^n: term-by-term with remainder.

Implements eq (7.35) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_35"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_35(x, n, delta):
    """Binomial expansion of (x + d)^n: term-by-term with remainder.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.35).
    """
    x_f, d_f = float(x), float(delta)
    n_i = int(n)
    terms = [math.comb(n_i, k) * x_f ** (n_i - k) * d_f ** k
             for k in range(n_i + 1)]
    total = float(sum(terms))
    exact = (x_f + d_f) ** n_i
    payload = {"terms": terms, "sum": total, "exact": exact,
               "abs_error": abs(total - exact)}
    lines = [("expansion sum", total), ("(x+d)^n", exact)]
    return RichResult(
        title="Binomial expansion of (x + d)^n: term-by-term with remainder.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e35: Binomial expansion of (x + d)^n: term-by-term with remainder. Morin (2016) eq (7.35)."
