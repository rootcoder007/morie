"""Maximal covering location problem (Church and ReVelle 1974).

Church, R. and ReVelle, C. (1974). The maximal covering location problem. Papers of the
Regional Science Association 32, 101-118.
"""

from ._facloc import distances, search
from ._richresult import RichResult

__all__ = ["maximal_covering"]


def maximal_covering(p, radius, demand=None, sites=None, dist=None, weights=None, max_enum=200000):
    r"""Choose p sites maximising the weight of demand within ``radius`` of an open site.

    Exact by enumeration when C(m, p) <= max_enum, otherwise greedy plus substitution.

    Parameters
    ----------
    p : int
    radius : float
        Service distance S (covered when d_ij <= S).
    demand, sites, dist, weights, max_enum
        As in :func:`p_median`.

    Returns
    -------
    RichResult
        Keys: sites, covered (bool per demand point), covered_weight, coverage_share, method.

    References
    ----------
    Church, R. and ReVelle, C. (1974). Papers of the Regional Science Association 32, 101-118.

    Examples
    --------
    >>> maximal_covering(1, 1.5, [(0, 0), (1, 0), (2, 0), (9, 0)])["covered_weight"]
    3.0
    """
    D = distances(demand, sites, dist)
    n, m = len(D), len(D[0])
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    R = float(radius)

    def obj(S):
        s = 0.0
        for i in range(n):
            if any(D[i][j] <= R for j in S):
                s += w[i]
        return -s

    S, best, how = search(obj, m, int(p), max_enum)
    cov = [any(D[i][j] <= R for j in S) for i in range(n)]
    tot = 0.0
    for v in w:
        tot += v
    return RichResult(
        title="Maximal covering location",
        summary_lines=[("covered weight", -best), ("method", how)],
        payload={"sites": S, "covered": cov, "covered_weight": -best, "coverage_share": -best / tot, "method": how},
    )


def cheatsheet():
    return "mclpop: maximal covering location (Church and ReVelle 1974)"
