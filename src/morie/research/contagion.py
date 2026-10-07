# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P10: near-repeat contagion as a branching process (``research/lean/P10Contagion.lean``).

* ``Research.P10.generation_mean``: E Z_k = n^k
* ``Research.P10.cluster_size_of_lt_one``: 0 <= n < 1 gives sum_k n^k = 1/(1-n)
* ``Research.P10.stationary_rate``: mean rate mu/(1-n)
* ``Research.P10.cluster_size_diverges_of_ge_one``: n >= 1 gives unbounded partial sums
* ``Research.P10.endogeneity_share``: share of triggered events = n
* ``Research.P10.iter_tendsto`` / ``extinction_fixed`` / ``extinction_le_fixed`` / ``subcritical_extinction_one`` / ``supercritical_extinction_lt_one`` (``P10Extinction.lean``)

R parity: ``rmorie`` ``R/contagion.R`` (``morie_contagion_branching``).
"""

from __future__ import annotations

__all__ = ["contagion_branching", "contagion_extinction"]


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


def contagion_extinction(p, tol=1e-14, max_iter=100000) -> dict:
    """Extinction probability of a near-repeat chain: the smallest fixed point of the generating function.

    Iterates ``s_0 = 0``, ``s_{n+1} = f(s_n)`` (``Research.P10.iter_tendsto``,
    ``extinction_fixed``, ``extinction_le_fixed``); mean offspring below one gives
    certain extinction (``subcritical_extinction_one``), above one a survival
    probability strictly positive (``supercritical_extinction_lt_one``).

    Examples
    --------
    >>> r = contagion_extinction([0.3, 0.3, 0.4])
    >>> (round(r["extinction"], 9), r["regime"])
    (0.75, 'supercritical')
    >>> round(contagion_extinction([0.5, 0.3, 0.2])["extinction"], 9)
    1.0
    """
    pp = [float(v) for v in p]
    if not pp or any(v != v or v < 0 for v in pp) or abs(sum(pp) - 1) > 1e-10:
        raise ValueError("p must be non-negative probabilities summing to one")
    ks = list(range(len(pp)))

    def f(s):
        return sum(pk * s**k for pk, k in zip(pp, ks))

    m = sum(k * pk for pk, k in zip(pp, ks))
    s = 0.0
    iterates = []
    for _ in range(int(max_iter)):
        s_new = f(s)
        iterates.append(s_new)
        if abs(s_new - s) < tol:
            s = s_new
            break
        s = s_new
    regime = "subcritical" if m < 1 else ("supercritical" if m > 1 else "critical")
    return {
        "mean_offspring": m,
        "regime": regime,
        "extinction": s,
        "survival": 1 - s,
        "iterates": iterates,
        "fixed_point_check": f(s) - s,
        "theorems": [
            "Research.P10.iter_mono",
            "Research.P10.iter_tendsto",
            "Research.P10.extinction_fixed",
            "Research.P10.extinction_le_fixed",
            "Research.P10.subcritical_extinction_one",
            "Research.P10.supercritical_extinction_lt_one",
        ],
    }
