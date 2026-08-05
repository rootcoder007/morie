"""Ordered sampling with replacement: N^n possible outcomes.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (1.4).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["ordrep"]


def ordrep(N, n):
    """Number of outcomes of n picks from N objects, with replacement, ordered.

    Eq (1.4): the count is N^n, because each of the n picks independently
    has N possible results.  Note it is N^n, not n^N.

    Parameters
    ----------
    N : int
        Number of distinct objects in the box.
    n : int
        Number of picks.

    Returns
    -------
    RichResult
        Keys: n_objects, n_picks, count, log_count.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq. (1.4).
    """
    N = _morin._check_nonneg_int(N, "N")
    n = _morin._check_nonneg_int(n, "n")
    count = N ** n
    import math
    log_count = n * math.log(N) if N > 0 and n > 0 else 0.0
    payload = {"n_objects": float(N), "n_picks": float(n),
               "count": float(count), "log_count": float(log_count)}
    return RichResult(
        title="Ordered sampling with replacement: N^n outcomes.",
        summary_lines=[("N", N), ("n", n), ("count", count)],
        payload=payload,
    )


def cheatsheet():
    return "ordrep: N^n outcomes for n ordered picks from N with replacement. Morin (2016) eq (1.4)."
