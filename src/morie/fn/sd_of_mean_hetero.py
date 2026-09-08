"""sigma of the average of n variables with different sigmas: sqrt(sum sigma_i^2)/n.

Implements eq (3.55) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["sd_of_mean_hetero"]


def sd_of_mean_hetero(sigmas):
    """sigma of the average of n variables with different sigmas: sqrt(sum sigma_i^2)/n.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.55).
    """
    value = _morin.sd_of_mean_hetero(sigmas)
    payload = {"sd_avg": value, "n": int(np.atleast_1d(sigmas).size)}
    lines = [("sigma_avg", value)]
    return RichResult(
        title="sigma of the average of n variables with different sigmas: sqrt(sum sigma_i^2)/n.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e55: sigma of the average of n variables with different sigmas: sqrt(sum sigma_i^2)/n. Morin (2016) eq (3.55)."


# compact alias per ledger/NAMING.md
sdofmeanhetero = sd_of_mean_hetero
