# morie.fn -- function file (rootcoder007/morie)
"""Shortest-path distance between units in W."""

import heapq
import math


def swpath(W, i=0, j=2, weighted=False):
    r"""Shortest-path distance between units ``i`` and ``j`` in the neighbour graph of ``W``.

    Unweighted (default): the order of contiguity, the fewest links from
    ``i`` to ``j`` along non-zero ``w_ab`` treated as undirected (breadth-first
    search; ``spdep::nblag`` order). With ``weighted=True`` the non-zero
    entries are edge lengths and Dijkstra's algorithm gives the least total
    length (Dijkstra 1959). Returns ``inf`` when ``j`` is unreachable.

    References
    ----------
    Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
    *Numerische Mathematik* 1, 269-271.

    Examples
    --------
    >>> swpath([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], 0, 3)
    3
    """
    A = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    n = len(A)
    nb = [[b for b in range(n) if b != a and (A[a][b] != 0 or A[b][a] != 0)] for a in range(n)]
    if not weighted:
        dist = {i: 0}
        frontier = [i]
        while frontier:
            nxt = []
            for a in frontier:
                for b in nb[a]:
                    if b not in dist:
                        dist[b] = dist[a] + 1
                        nxt.append(b)
            frontier = nxt
        return dist.get(j, math.inf)
    best = [math.inf] * n
    best[i] = 0.0
    heap = [(0.0, i)]
    while heap:
        dcur, a = heapq.heappop(heap)
        if dcur > best[a]:
            continue
        for b in nb[a]:
            w = A[a][b] if A[a][b] != 0 else A[b][a]
            if dcur + w < best[b]:
                best[b] = dcur + w
                heapq.heappush(heap, (best[b], b))
    return best[j]


swpath_fn = swpath


def cheatsheet() -> str:
    return "swpath(W, i, j, weighted=False) -> contiguity order (BFS) or Dijkstra length between units."
