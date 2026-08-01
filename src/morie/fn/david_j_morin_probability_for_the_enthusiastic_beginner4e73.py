"""Hypergeometric rearranged with falling factorials (limit setup).

Implements eq (4.73) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_73"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_73(k, n, p, N):
    """Hypergeometric rearranged with falling factorials (limit setup).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.73).
    """
    hyper, binom_p, err = _morin.hypergeometric_binomial_limit(k, n, p, N)
    payload = {"hypergeometric": hyper, "binomial": binom_p, "abs_error": err}
    lines = [("hypergeometric", hyper), ("binomial", binom_p)]
    return RichResult(
        title="Hypergeometric rearranged with falling factorials (limit setup).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e73: Hypergeometric rearranged with falling factorials (limit setup). Morin (2016) eq (4.73)."
