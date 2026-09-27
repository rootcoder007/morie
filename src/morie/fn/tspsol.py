"""Travelling salesman tours: exact Held-Karp dynamic programme, or nearest neighbour plus 2-opt.

Held, M. and Karp, R. M. (1962). A dynamic programming approach to sequencing problems.
Journal of SIAM 10, 196-210. Croes, G. A. (1958). A method for solving traveling-salesman
problems. Operations Research 6, 791-812 (2-opt).
"""

import math

from ._richresult import RichResult

__all__ = ["travelling_salesman"]


def _dist(points):
    return [[math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q))) for q in points] for p in points]


def _length(D, tour):
    s = 0.0
    for k in range(len(tour)):
        s += D[tour[k]][tour[(k + 1) % len(tour)]]
    return s


def travelling_salesman(points=None, dist=None, method="auto", max_exact=13):
    r"""Shortest closed tour through every node, starting and ending at node 0.

    ``method="exact"`` is the Held-Karp recursion
    C(S, j) = min_{i in S - j} C(S - j, i) + d_ij over subsets S containing j,
    O(2^n n^2) time; ``"heuristic"`` builds a nearest-neighbour tour from node 0
    (ties to the lower index) and applies first-improvement 2-opt reversals
    (Croes 1958) until none shortens the tour; ``"auto"`` is exact for
    n <= max_exact. Tours are returned starting at 0 with the second node the
    smaller of 0's two neighbours.

    Parameters
    ----------
    points : list of coordinate tuples, optional
        Euclidean distances are used.
    dist : square matrix, optional
        Symmetric distance matrix (instead of points).
    method : {"auto", "exact", "heuristic"}
    max_exact : int

    Returns
    -------
    RichResult
        Keys: tour, length, method.

    References
    ----------
    Held, M. and Karp, R. M. (1962). Journal of SIAM 10, 196-210.
    Croes, G. A. (1958). Operations Research 6, 791-812.

    Examples
    --------
    >>> travelling_salesman([(0, 0), (1, 0), (1, 1), (0, 1)])["length"]
    4.0
    """
    if (points is None) == (dist is None):
        raise ValueError("give exactly one of points and dist")
    D = [[float(v) for v in r] for r in dist] if dist is not None else _dist([[float(v) for v in p] for p in points])
    n = len(D)
    if n == 0 or any(len(r) != n for r in D):
        raise ValueError("need a non-empty square distance matrix")
    if method not in ("auto", "exact", "heuristic"):
        raise ValueError('method must be "auto", "exact" or "heuristic"')
    exact = method == "exact" or (method == "auto" and n <= max_exact)
    if n <= 3:
        tour = list(range(n))
    elif exact:
        full = 1 << (n - 1)  # subsets of nodes 1..n-1
        C = [[math.inf] * n for _ in range(full)]
        P = [[-1] * n for _ in range(full)]
        for j in range(1, n):
            C[1 << (j - 1)][j] = D[0][j]
        for S in range(1, full):
            for j in range(1, n):
                bj = 1 << (j - 1)
                if not S & bj or C[S][j] == math.inf:
                    continue
                for k in range(1, n):
                    bk = 1 << (k - 1)
                    if S & bk:
                        continue
                    c = C[S][j] + D[j][k]
                    if c < C[S | bk][k]:
                        C[S | bk][k] = c
                        P[S | bk][k] = j
        S = full - 1
        best, last = math.inf, -1
        for j in range(1, n):
            c = C[S][j] + D[j][0]
            if c < best:
                best, last = c, j
        tour = []
        while last != -1:
            tour.append(last)
            prev = P[S][last]
            S &= ~(1 << (last - 1))
            last = prev
        tour = [0] + tour[::-1]
    else:
        tour, seen = [0], {0}
        while len(tour) < n:
            cur = tour[-1]
            nxt = min((j for j in range(n) if j not in seen), key=lambda j: (D[cur][j], j))
            tour.append(nxt)
            seen.add(nxt)
        improved = True
        while improved:
            improved = False
            for a in range(n - 1):
                for b in range(a + 2, n if a > 0 else n - 1):
                    i, i1, j, j1 = tour[a], tour[a + 1], tour[b], tour[(b + 1) % n]
                    if D[i][j] + D[i1][j1] < D[i][i1] + D[j][j1] - 1e-12:
                        tour[a + 1 : b + 1] = tour[a + 1 : b + 1][::-1]
                        improved = True
    if n > 2 and tour[-1] < tour[1]:
        tour = [0] + tour[1:][::-1]
    return RichResult(
        title="Travelling salesman tour",
        summary_lines=[("length", _length(D, tour)), ("method", "exact" if exact else "heuristic")],
        payload={
            "tour": tour,
            "length": _length(D, tour),
            "method": "Held-Karp" if exact else "nearest neighbour + 2-opt",
        },
    )


def cheatsheet():
    return "tspsol: TSP tour, Held-Karp exact or nearest neighbour + 2-opt"
