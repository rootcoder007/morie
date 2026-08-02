"""Recursive form: Var(X1+...+Xn) = Var(X1+...+X_{n-1}) + Var(Xn).

Implements eq (3.31) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_31"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_31(variances):
    """Recursive form: Var(X1+...+Xn) = Var(X1+...+X_{n-1}) + Var(Xn).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.31).
    """
    v = np.atleast_1d(np.asarray(variances, dtype=float))
    if v.size < 1 or np.any(v < 0):
        raise ValueError("variances must be non-empty and >= 0")
    running = 0.0
    steps = []
    for x in v:
        running = running + float(x)
        steps.append(running)
    direct = _morin.var_sum_independent(v)
    if abs(running - direct) > 1e-12 * max(1.0, direct):
        raise AssertionError("recursion disagrees with direct sum")
    payload = {"var_sum": running, "partial_sums": steps}
    lines = [("Var(sum)", running)]
    return RichResult(
        title="Recursive form: Var(X1+...+Xn) = Var(X1+...+X_{n-1}) + Var(Xn).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e31: Recursive form: Var(X1+...+Xn) = Var(X1+...+X_{n-1}) + Var(Xn). Morin (2016) eq (3.31)."
