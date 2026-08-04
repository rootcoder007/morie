# morie.fn -- function file (rootcoder007/morie)
"""Average shortest path length of a graph."""

from __future__ import annotations

from . import _t4core as T

from ._richresult import RichResult

__all__ = ["avg_path_length"]


def avg_path_length(G, directed=False):
    """Mean geodesic distance, and its harmonic counterpart.

    Formula: ``L = (1/|R|) sum_{(i,j) in R} d(i,j)`` over the ordered
    pairs ``R`` with ``i != j`` that are actually connected, and

        ``L_harm^-1 = (1/(n(n-1))) sum_{i != j} 1/d(i,j)``

    with ``1/d = 0`` for unreachable pairs.

    The arithmetic mean diverges the moment the graph is disconnected,
    which is why the unreachable pairs are dropped from ``L`` and
    counted separately in ``reachable`` rather than being quietly
    imputed; the harmonic mean is the standard repair, is always finite,
    and is reported alongside so the two can be compared.  Distances
    come from breadth-first search from every vertex, so the graph is
    treated as unweighted -- a non-zero entry of ``G`` means an edge,
    not a length.

    Parameters
    ----------
    G : array-like
        ``n x n`` adjacency matrix; non-zero means an edge.
    directed : bool
        Follow edge direction.  If False the matrix is symmetrised.

    Returns
    -------
    RichResult
        ``estimate`` (L), ``harmonic``, ``diameter``, ``reachable``,
        ``pairs``, ``n``, ``method``.

    References
    ----------
    Newman (2010), Networks: An Introduction, Oxford University Press,
    sec. 7.6 (mean geodesic distance) and the harmonic-mean repair for
    disconnected graphs.  The book is not in the local corpus and is not
    fetchable, so this is the standard published form rather than a
    quoted equation; the breadth-first construction of ``d`` is not in
    dispute and the two conventions that are -- self-pairs and
    unreachable pairs -- are both stated explicitly above and reported
    separately in the payload.
    """
    A = T.mat(G)
    n = len(A)
    if n < 2 or any(len(r) != n for r in A):
        raise ValueError("G must be a square adjacency matrix with n >= 2")
    adj = []
    for i in range(n):
        nb = []
        for j in range(n):
            if i == j:
                continue
            if A[i][j] != 0.0 or (not directed and A[j][i] != 0.0):
                nb.append(j)
        adj.append(nb)
    total = 0.0
    harm = 0.0
    reach = 0
    diam = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for t in range(n):
            if t == s or dist[t] < 0:
                continue
            total += dist[t]
            harm += 1.0 / dist[t]
            reach += 1
            if dist[t] > diam:
                diam = dist[t]
    pairs = n * (n - 1)
    return RichResult(
        payload={
            "estimate": total / reach if reach else float("nan"),
            "harmonic": (pairs / harm) if harm > 0 else float("inf"),
            "diameter": int(diam),
            "reachable": int(reach),
            "pairs": int(pairs),
            "n": int(n),
            "method": "Mean geodesic distance",
        }
    )


def cheatsheet():
    return "avg_path_length(G): mean geodesic distance over connected pairs."


# compact alias per ledger/NAMING.md
avgpathlength = avg_path_length
