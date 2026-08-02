"""Standard deviation of a sum of n i.i.d. variables: sigma_sum = sqrt(n) sigma.

Implements eq (3.45; worked on p. 253) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_71"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_71(sigma, n):
    """Standard deviation of a sum of n i.i.d. variables: sigma_sum = sqrt(n) sigma.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.45; worked on p. 253).
    """
    value = _morin.sd_of_iid_sum(sigma, n)
    payload = {"sigma": float(sigma), "n": int(n), "sd_sum": value}
    lines = [("per-variable sigma", float(sigma)), ("sd of sum", value)]
    return RichResult(
        title="Standard deviation of a sum of n i.i.d. variables: sigma_sum = sqrt(n) sigma.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner1e71: Standard deviation of a sum of n i.i.d. variables: sigma_sum = sqrt(n) sigma. Morin (2016) eq (3.45; worked on p. 253)."
