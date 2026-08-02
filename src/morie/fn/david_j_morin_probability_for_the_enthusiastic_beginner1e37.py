"""Multinomial coefficient (N; n1, n2, ..., nk) = N!/(n1!...nk!).

Implements eq (1.37) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_37"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_37(ns):
    """Multinomial coefficient (N; n1, n2, ..., nk) = N!/(n1!...nk!).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (1.37).
    """
    value = _morin.multinomial_coefficient(ns)
    payload = {"ns": [int(x) for x in np.atleast_1d(ns)], "coefficient": value}
    lines = [("multinomial coefficient", value)]
    return RichResult(
        title="Multinomial coefficient (N; n1, n2, ..., nk) = N!/(n1!...nk!).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner1e37: Multinomial coefficient (N; n1, n2, ..., nk) = N!/(n1!...nk!). Morin (2016) eq (1.37)."
