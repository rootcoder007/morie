"""Uncapacitated facility location (Balinski 1965; Kuehn and Hamburger 1963 heuristic).

Kuehn, A. A. and Hamburger, M. J. (1963). A heuristic program for locating warehouses.
Management Science 9, 643-666. Balinski, M. L. (1965). Integer programming: methods, uses,
computation. Management Science 12, 253-313.
"""

import math
from itertools import combinations

from ._facloc import distances
from ._richresult import RichResult

__all__ = ["facility_location"]


def facility_location(fixed_costs, demand=None, sites=None, dist=None, weights=None, max_enum=200000):
    r"""Open sites S minimising sum_{j in S} f_j + sum_i w_i min_{j in S} d_ij.

    Exact over all non-empty subsets when 2^m - 1 <= max_enum; otherwise the
    Kuehn-Hamburger greedy add (open the site with the largest saving while one saves)
    followed by drop and swap moves until none improves.

    Parameters
    ----------
    fixed_costs : sequence
        f_j per candidate site.
    demand, sites, dist, weights, max_enum
        As in :func:`p_median`.

    Returns
    -------
    RichResult
        Keys: sites, assignment, total, fixed, transport, method.

    References
    ----------
    Kuehn, A. A. and Hamburger, M. J. (1963). Management Science 9, 643-666.

    Examples
    --------
    >>> facility_location([1, 100, 1], [(0, 0), (1, 0), (10, 0)])["sites"]
    [0, 2]
    """
    D = distances(demand, sites, dist)
    n, m = len(D), len(D[0])
    f = [float(v) for v in fixed_costs]
    if len(f) != m:
        raise ValueError("one fixed cost per site")
    w = [1.0] * n if weights is None else [float(v) for v in weights]

    def total(S):
        t = 0.0
        for j in S:
            t += f[j]
        for i in range(n):
            t += w[i] * min(D[i][j] for j in S)
        return t

    if 2**m - 1 <= max_enum:
        best, bs = math.inf, None
        for k in range(1, m + 1):
            for S in combinations(range(m), k):
                v = total(S)
                if v < best - 1e-12:
                    best, bs = v, list(S)
        S, how = bs, "exact enumeration"
    else:
        S = [min(range(m), key=lambda j: (total((j,)), j))]
        best = total(S)
        while True:
            cand = [(total(tuple(sorted(S + [j]))), j) for j in range(m) if j not in S]
            if not cand or min(cand)[0] >= best - 1e-12:
                break
            best, j = min(cand)
            S = sorted(S + [j])
        improved = True
        while improved:
            improved = False
            moves = [tuple(s for s in S if s != o) for o in S if len(S) > 1]
            moves += [tuple(sorted([s for s in S if s != o] + [j])) for o in S for j in range(m) if j not in S]
            for T in moves:
                v = total(T)
                if v < best - 1e-12:
                    S, best, improved = list(T), v, True
                    break
        how = "greedy add + drop/swap (Kuehn and Hamburger 1963)"
    assign = [min(S, key=lambda j: (D[i][j], j)) for i in range(n)]
    fx = 0.0
    for j in S:
        fx += f[j]
    return RichResult(
        title="Uncapacitated facility location",
        summary_lines=[("total", best), ("method", how)],
        payload={"sites": S, "assignment": assign, "total": best, "fixed": fx, "transport": best - fx, "method": how},
    )


def cheatsheet():
    return "uflpop: uncapacitated facility location, exact or Kuehn-Hamburger heuristic"
