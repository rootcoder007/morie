"""Bayes' theorem, simple form: P(A|Z) = P(Z|A) P(A) / P(Z).

Implements eq (2.51) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["bayes_simple"]


def bayes_simple(p_z_given_a, p_a, p_z):
    """Bayes' theorem, simple form: P(A|Z) = P(Z|A) P(A) / P(Z).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.51).
    """
    value = _morin.bayes_simple(p_z_given_a, p_a, p_z)
    payload = {"posterior": value}
    lines = [("P(A|Z)", value)]
    return RichResult(
        title="Bayes' theorem, simple form: P(A|Z) = P(Z|A) P(A) / P(Z).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e51: Bayes' theorem, simple form: P(A|Z) = P(Z|A) P(A) / P(Z). Morin (2016) eq (2.51)."
