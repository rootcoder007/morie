"""Pre-trial custody credit days calculation (R v Summers, 1.5x)."""

from __future__ import annotations

import math

from morie.fn._containers import ESRes


def custody_days_credit(pretrial_days, *, credit_ratio: float = 1.5) -> ESRes:
    r"""Enhanced credit for pre-sentence custody: ``credited = ratio x days`` with ``1 <= ratio <= 1.5``.

    Criminal Code of Canada s. 719(3)-(3.1): a court may credit each day of
    pre-sentence custody at up to one and a half days; *R. v. Summers*,
    2014 SCC 26, held that the loss of early-release eligibility is by
    itself a circumstance justifying the enhanced 1.5 : 1 rate, so 1.5 is
    the usual ratio. Ratios outside ``[1, 1.5]`` are rejected. ``estimate``
    is the mean credited days; ``extra`` holds each person's credit and the
    totals.

    References
    ----------
    R. v. Summers, 2014 SCC 26, [2014] 1 S.C.R. 575.
    Criminal Code, R.S.C. 1985, c. C-46, s. 719(3), (3.1).

    Examples
    --------
    >>> custody_days_credit([10, 30, 45]).extra["credited"]
    [15.0, 45.0, 67.5]
    """
    if not 1.0 <= float(credit_ratio) <= 1.5:
        raise ValueError("credit_ratio must lie in [1, 1.5] (Criminal Code s. 719(3.1))")
    days = [float(v) for v in (pretrial_days.tolist() if hasattr(pretrial_days, "tolist") else pretrial_days)]
    if any(d < 0 for d in days):
        raise ValueError("pre-trial days must be non-negative")
    credited = [d * credit_ratio for d in days]
    return ESRes(
        measure="custody_days_credit",
        estimate=math.fsum(credited) / len(credited),
        n=len(days),
        extra={
            "credited": credited,
            "total_pretrial": math.fsum(days),
            "total_credited": math.fsum(credited),
            "credit_ratio": credit_ratio,
        },
    )


cstdy2 = custody_days_credit


def cheatsheet() -> str:
    return "custody_days_credit(days, credit_ratio=1.5) -> credited = ratio x days, ratio in [1, 1.5] (R v Summers)."
