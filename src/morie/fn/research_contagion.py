# morie.fn -- function file (rootcoder007/morie)
"""Research P10: near-repeat contagion as a branching process.

Python twin of ``R/contagion.R`` (``morie_contagion_branching``); the
identities are machine-checked in ``research/lean/P10Contagion.lean``:

- ``Research.P10.generation_mean``: E Z_{k+1} = n E Z_k, so E Z_k = n^k
- ``Research.P10.cluster_size_of_lt_one``: 0 <= n < 1 gives sum_k n^k = 1/(1-n)
- ``Research.P10.stationary_rate``: background rate mu gives mean rate mu/(1-n)
- ``Research.P10.cluster_size_diverges_of_ge_one``: n >= 1, partial sums unbounded
- ``Research.P10.endogeneity_share``: the share of triggered events is n
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["contagion_branching"]

_THEOREMS = [
    "Research.P10.generation_mean",
    "Research.P10.cluster_size_of_lt_one",
    "Research.P10.stationary_rate",
    "Research.P10.cluster_size_diverges_of_ge_one",
    "Research.P10.endogeneity_share",
]


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def contagion_branching(n, mu=1.0, generations=10):
    """Branching-ratio arithmetic of a self-exciting (Hawkes) crime process.

    Every recorded event triggers on average ``n`` further events (the
    branching ratio). Generation ``k`` of a cluster has expected size
    ``n**k``; for ``n < 1`` the expected cluster size is ``1/(1-n)``, the
    stationary rate is ``mu/(1-n)`` and the share of triggered events is
    ``n``. For ``n >= 1`` the expected cluster size is infinite: a fitted
    branching ratio at or above one predicts unbounded crime.

    Parameters
    ----------
    n : float
        Branching ratio, non-negative.
    mu : float
        Background rate (events per unit time), positive.
    generations : int
        How many generation means to report.

    Returns
    -------
    RichResult
        ``n``, ``subcritical``, ``generation_means``,
        ``expected_cluster_size`` (``inf`` when ``n >= 1``),
        ``stationary_rate``, ``endogeneity_share`` (``nan`` when
        ``n >= 1``) and ``theorems``.

    Examples
    --------
    >>> b = contagion_branching(0.4, mu=2)
    >>> round(b.expected_cluster_size, 12), round(b.stationary_rate, 12)
    (1.666666666667, 3.333333333333)
    >>> [round(v, 6) for v in b.generation_means[:4]]
    [1.0, 0.4, 0.16, 0.064]
    >>> contagion_branching(1.1, mu=2).expected_cluster_size
    inf
    """
    if not _is_num(n) or n < 0:
        raise ValueError("n must be a single non-negative number")
    if not _is_num(mu) or mu <= 0:
        raise ValueError("mu must be a single positive number")
    n = float(n)
    sub = n < 1
    return RichResult(
        title="Branching ratio of a self-exciting process",
        payload={
            "n": n,
            "subcritical": sub,
            "generation_means": [n**k for k in range(int(generations))],
            "expected_cluster_size": 1 / (1 - n) if sub else math.inf,
            "stationary_rate": mu / (1 - n) if sub else math.inf,
            "endogeneity_share": n if sub else math.nan,
            "theorems": list(_THEOREMS),
        },
    )


def cheatsheet() -> str:
    return (
        "contagion_branching(n, mu=1, generations=10) -> generation means n^k, cluster size 1/(1-n), "
        "stationary rate mu/(1-n), endogeneity share n (Research P10)."
    )
