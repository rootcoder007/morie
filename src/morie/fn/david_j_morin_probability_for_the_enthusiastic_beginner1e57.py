"""Stars and bars: N_U_n = C(n+N-1, N-1) unordered samples with repetition.

Implements eq (1.57) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_57"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_57(n, N):
    """Stars and bars: N_U_n = C(n+N-1, N-1) unordered samples with repetition.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (1.57).
    """
    value = _morin.stars_and_bars(n, N)
    payload = {"n": int(n), "N": int(N), "count": value}
    lines = [("n draws from N types", value)]
    return RichResult(
        title="Stars and bars: N_U_n = C(n+N-1, N-1) unordered samples with repetition.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner1e57: Stars and bars: N_U_n = C(n+N-1, N-1) unordered samples with repetition. Morin (2016) eq (1.57)."
