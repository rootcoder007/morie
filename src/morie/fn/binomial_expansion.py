"""Binomial theorem: (a+b)^n = sum_k C(n,k) a^(n-k) b^k.

Implements eq (1.21) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["binomial_expansion"]


def binomial_expansion(a, b, n):
    """Binomial theorem: (a+b)^n = sum_k C(n,k) a^(n-k) b^k.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (1.21).

    Examples
    --------
    >>> round(binomial_expansion(2.0, 3.0, 4)["sum"], 12)
    625.0
    """
    terms, total = _morin.binomial_expansion(a, b, n)
    direct = float((float(a) + float(b)) ** int(n))
    payload = {"terms": terms, "sum": total, "direct": direct, "max_abs_error": abs(total - direct)}
    lines = [("sum of terms", total), ("(a+b)^n", direct)]
    return RichResult(
        title="Binomial theorem: (a+b)^n = sum_k C(n,k) a^(n-k) b^k.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner1e21: Binomial theorem: (a+b)^n = sum_k C(n,k) a^(n-k) b^k. Morin (2016) eq (1.21)."
