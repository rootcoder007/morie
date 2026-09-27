# morie.fn -- function file (rootcoder007/morie)
"""Incidence rate ratio (IRR) effect size."""

import math

from . import _stats_core as stats
from ._containers import ESRes


def rate_ratio(
    events1: int,
    person_time1: float,
    events2: int,
    person_time2: float,
    confidence: float = 0.95,
) -> ESRes:
    """Incidence rate ratio.

    IRR = (e1/PT1) / (e2/PT2)

    Parameters
    ----------
    events1, person_time1 : int, float
    events2, person_time2 : int, float
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    if person_time1 <= 0 or person_time2 <= 0:
        raise ValueError("person-time must be positive")
    # 1/2 added to both event counts when either is zero (metafor::escalc "IRR"), as the R arm
    cc = 0.5 if min(events1, events2) == 0 else 0.0
    e1, e2 = events1 + cc, events2 + cc
    irr = (e1 / person_time1) / (e2 / person_time2)
    log_irr = math.log(irr)
    se = math.sqrt(1 / e1 + 1 / e2)
    z = stats.norm.ppf((1 + confidence) / 2)
    return ESRes(
        measure="Rate ratio",
        estimate=float(irr),
        ci_lower=float(math.exp(log_irr - z * se)),
        ci_upper=float(math.exp(log_irr + z * se)),
        se=float(se),
        n=events1 + events2,
    )


irr = rate_ratio


def cheatsheet() -> str:
    return "rate_ratio({}) -> Incidence rate ratio (IRR) effect size."


# compact alias per ledger/NAMING.md
rateratio = rate_ratio
