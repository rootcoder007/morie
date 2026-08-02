"""Binomial theorem: (a+b)^n = sum_k C(n,k) a^(n-k) b^k.

Implements eq (1.21) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_21"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_21(a, b, n):
    """Binomial theorem: (a+b)^n = sum_k C(n,k) a^(n-k) b^k.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (1.21).
    """
    terms, total = _morin.binomial_expansion(a, b, n)
    direct = float((float(a) + float(b)) ** int(n))
    payload = {"terms": terms, "sum": total, "direct": direct,
               "max_abs_error": abs(total - direct)}
    lines = [("sum of terms", total), ("(a+b)^n", direct)]
    return RichResult(
        title="Binomial theorem: (a+b)^n = sum_k C(n,k) a^(n-k) b^k.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner1e21: Binomial theorem: (a+b)^n = sum_k C(n,k) a^(n-k) b^k. Morin (2016) eq (1.21)."
