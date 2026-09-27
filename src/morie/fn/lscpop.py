"""Location set covering problem (Toregas, Swain, ReVelle and Bergman 1971).

Toregas, C., Swain, R., ReVelle, C. and Bergman, L. (1971). The location of emergency service
facilities. Operations Research 19, 1363-1373.
"""

from ._facloc import distances
from ._richresult import RichResult

__all__ = ["set_covering_location"]


def set_covering_location(radius, demand=None, sites=None, dist=None, costs=None):
    r"""Fewest (or cheapest) sites such that every demand point is within ``radius`` of one.

    Exact depth-first branch and bound: branch on the uncovered demand point with the
    fewest covering sites, trying those sites in order of cost then index; prune when
    the cost so far plus the cheapest site reaches the incumbent. Points no site can
    reach make the problem infeasible.

    Parameters
    ----------
    radius : float
    demand, sites : coordinates, optional
    dist : matrix, optional
    costs : sequence, optional
        Site costs (default 1: minimise the number of sites).

    Returns
    -------
    RichResult
        Keys: sites, cost, n_sites.

    References
    ----------
    Toregas, C., Swain, R., ReVelle, C. and Bergman, L. (1971). Operations Research 19, 1363-1373.

    Examples
    --------
    >>> set_covering_location(1.0, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])["sites"]
    [1, 4]
    """
    D = distances(demand, sites, dist)
    n, m = len(D), len(D[0])
    c = [1.0] * m if costs is None else [float(v) for v in costs]
    R = float(radius)
    cover = [[j for j in range(m) if D[i][j] <= R] for i in range(n)]
    if any(not cv for cv in cover):
        raise ValueError("some demand point is farther than radius from every site")
    cmin = min(c)
    best = [float("inf"), None]

    def rec(chosen, cost, covered):
        if cost + (cmin if len(covered) < n else 0.0) >= best[0] - 1e-12:
            return
        if len(covered) == n:
            best[0], best[1] = cost, sorted(chosen)
            return
        i = min((i for i in range(n) if i not in covered), key=lambda i: (len(cover[i]), i))
        for j in sorted(cover[i], key=lambda j: (c[j], j)):
            rec(chosen + [j], cost + c[j], covered | {k for k in range(n) if D[k][j] <= R})

    rec([], 0.0, frozenset())
    return RichResult(
        title="Location set covering",
        summary_lines=[("cost", best[0])],
        payload={"sites": best[1], "cost": best[0], "n_sites": len(best[1])},
    )


def cheatsheet():
    return "lscpop: location set covering, exact branch and bound"
