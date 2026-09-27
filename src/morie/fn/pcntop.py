"""p-center location (Hakimi 1965; minimax facility location).

Hakimi, S. L. (1965). Optimum distribution of switching centers in a communication network
and some related graph theoretic problems. Operations Research 13, 462-475.
"""

from ._facloc import distances, search
from ._richresult import RichResult

__all__ = ["p_center"]


def p_center(p, demand=None, sites=None, dist=None, weights=None, max_enum=200000):
    r"""Choose p sites minimising max_i w_i min_{j in S} d_ij (vertex p-center).

    Exact by enumeration when C(m, p) <= max_enum, otherwise greedy addition plus vertex
    substitution on the minimax objective.

    Parameters
    ----------
    p : int
    demand, sites : lists of coordinates, optional
    dist : matrix, optional
    weights : sequence, optional
    max_enum : int

    Returns
    -------
    RichResult
        Keys: sites, assignment, radius, method.

    References
    ----------
    Hakimi, S. L. (1965). Operations Research 13, 462-475.

    Examples
    --------
    >>> p_center(1, [(0, 0), (1, 0), (4, 0)])["radius"]
    3.0
    """
    D = distances(demand, sites, dist)
    n, m = len(D), len(D[0])
    w = [1.0] * n if weights is None else [float(v) for v in weights]

    def obj(S):
        return max(w[i] * min(D[i][j] for j in S) for i in range(n))

    S, best, how = search(obj, m, int(p), max_enum)
    assign = [min(S, key=lambda j: (D[i][j], j)) for i in range(n)]
    return RichResult(
        title="p-center",
        summary_lines=[("radius", best), ("method", how)],
        payload={"sites": S, "assignment": assign, "radius": best, "method": how},
    )


def cheatsheet():
    return "pcntop: vertex p-center (minimax), exact enumeration or substitution"
