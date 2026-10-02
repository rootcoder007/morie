# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P10: near-repeat contagion as a branching process (``research/lean/P10Contagion.lean``).

* ``Research.P10.generation_mean``: E Z_k = n^k
* ``Research.P10.cluster_size_of_lt_one``: 0 <= n < 1 gives sum_k n^k = 1/(1-n)
* ``Research.P10.stationary_rate``: mean rate mu/(1-n)
* ``Research.P10.cluster_size_diverges_of_ge_one``: n >= 1 gives unbounded partial sums
* ``Research.P10.endogeneity_share``: share of triggered events = n

R parity: ``rmorie`` ``R/contagion.R`` (``morie_contagion_branching``).
"""

from __future__ import annotations

__all__ = ["contagion_branching"]


def contagion_branching(n, mu=1.0, generations=10) -> dict:
    """Branching-ratio arithmetic of a self-exciting (Hawkes) crime process.

    Examples
    --------
    >>> r = contagion_branching(n=0.4, mu=2)
    >>> (round(r["expected_cluster_size"], 12), round(r["stationary_rate"], 12), r["endogeneity_share"])
    (1.666666666667, 3.333333333333, 0.4)
    >>> contagion_branching(n=1.1, mu=2)["expected_cluster_size"]
    inf
    """
    n = float(n)
    mu = float(mu)
    generations = int(generations)
    if n != n or n < 0:
        raise ValueError("n must be a single non-negative number")
    if mu != mu or mu <= 0:
        raise ValueError("mu must be a single positive number")
    sub = n < 1
    return {
        "n": n,
        "subcritical": sub,
        "generation_means": [n**k for k in range(generations)],
        "expected_cluster_size": 1 / (1 - n) if sub else float("inf"),
        "stationary_rate": mu / (1 - n) if sub else float("inf"),
        "endogeneity_share": n if sub else float("nan"),
        "theorems": [
            "Research.P10.generation_mean",
            "Research.P10.cluster_size_of_lt_one",
            "Research.P10.stationary_rate",
            "Research.P10.cluster_size_diverges_of_ge_one",
            "Research.P10.endogeneity_share",
        ],
    }
