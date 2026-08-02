"""Prevalence sensitivity of the posterior (40% variant).

Implements eq (2.62) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_62"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_62(p_a=0.40, p_z_given_a=0.95, p_z_given_not_a=0.10):
    """Prevalence sensitivity of the posterior (40% variant).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.62).
    """
    value = _morin.bayes_explicit(p_a, p_z_given_a, p_z_given_not_a)
    payload = {"posterior": value}
    lines = [("P(A|Z)", value)]
    return RichResult(
        title="Prevalence sensitivity of the posterior (40% variant).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e62: Prevalence sensitivity of the posterior (40% variant). Morin (2016) eq (2.62)."
