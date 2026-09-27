"""p-median location (Hakimi 1964; ReVelle and Swain 1970).

Hakimi, S. L. (1964). Optimum locations of switching centers and the absolute centers and
medians of a graph. Operations Research 12, 450-459. Teitz, M. B. and Bart, P. (1968).
Heuristic methods for estimating the generalized vertex median of a weighted graph.
Operations Research 16, 955-961.
"""

from ._facloc import distances, search
from ._richresult import RichResult

__all__ = ["p_median"]


def p_median(p, demand=None, sites=None, dist=None, weights=None, max_enum=200000):
    r"""Choose p sites minimising sum_i w_i min_{j in S} d_ij.

    Exact by enumeration when C(m, p) <= max_enum, otherwise greedy addition plus
    Teitz-Bart vertex substitution. Each demand point is assigned to its nearest open
    site (ties to the lower index).

    Parameters
    ----------
    p : int
    demand, sites : lists of coordinates, optional
        Euclidean distances; sites default to the demand points.
    dist : matrix, optional
        Demand-by-site distances (instead of coordinates).
    weights : sequence, optional
        Demand weights (default 1).
    max_enum : int

    Returns
    -------
    RichResult
        Keys: sites, assignment, objective, method.

    References
    ----------
    Hakimi, S. L. (1964). Operations Research 12, 450-459.
    Teitz, M. B. and Bart, P. (1968). Operations Research 16, 955-961.

    Examples
    --------
    >>> p_median(2, [(0, 0), (1, 0), (10, 0), (11, 0)])["objective"]
    2.0
    """
    D = distances(demand, sites, dist)
    n, m = len(D), len(D[0])
    w = [1.0] * n if weights is None else [float(v) for v in weights]

    def obj(S):
        s = 0.0
        for i in range(n):
            s += w[i] * min(D[i][j] for j in S)
        return s

    S, best, how = search(obj, m, int(p), max_enum)
    assign = [min(S, key=lambda j: (D[i][j], j)) for i in range(n)]
    return RichResult(
        title="p-median",
        summary_lines=[("objective", best), ("method", how)],
        payload={"sites": S, "assignment": assign, "objective": best, "method": how},
    )


def cheatsheet():
    return "pmedop: p-median, exact enumeration or Teitz-Bart substitution"
