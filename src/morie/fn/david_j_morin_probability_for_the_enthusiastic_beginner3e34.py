"""Computational variance form Var(X) = E(X^2) - mu^2 for a pmf.

Implements eq (3.34) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_34"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_34(values, probs):
    """Computational variance form Var(X) = E(X^2) - mu^2 for a pmf.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.34).
    """
    values_a, probs_a = _morin._check_pmf(values, probs)
    mu = _morin.pmf_expectation(values_a, probs_a)
    e_x2 = float(np.sum(probs_a * values_a ** 2))
    value = e_x2 - mu ** 2
    payload = {"variance": value, "e_x2": e_x2, "mean": mu}
    lines = [("E(X^2) - mu^2", value)]
    return RichResult(
        title="Computational variance form Var(X) = E(X^2) - mu^2 for a pmf.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e34: Computational variance form Var(X) = E(X^2) - mu^2 for a pmf. Morin (2016) eq (3.34)."
