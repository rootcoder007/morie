"""Standard deviation of the mean: sigma / sqrt(n).

Implements eq (3.53) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["sd_of_mean"]


def sd_of_mean(sigma, n):
    """Standard deviation of the mean: sigma / sqrt(n).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.53).
    """
    value = _morin.sd_of_mean(sigma, n)
    payload = {"sigma": float(sigma), "n": int(n), "sd_mean": value}
    lines = [("sigma/sqrt(n)", value)]
    return RichResult(
        title="Standard deviation of the mean: sigma / sqrt(n).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e53: Standard deviation of the mean: sigma / sqrt(n). Morin (2016) eq (3.53)."


# compact alias per ledger/NAMING.md
sdofmean = sd_of_mean
