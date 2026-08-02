# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closeness centrality (Wasserman-Faust)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sgt_closeness_centrality"]


def sgt_closeness_centrality(A):
    """
    Closeness centrality of every vertex.

    Formula: C(v) = (n - 1) / sum_u d(v, u), with d the unweighted
    shortest-path distance computed by BFS from each vertex. The
    graph must be undirected and CONNECTED — on a disconnected graph
    some d(v,u) are infinite and Wasserman-Faust closeness is
    undefined (refused; use a harmonic variant deliberately instead).

    Parameters
    ----------
    A : array-like, shape (n, n)
        Binary symmetric adjacency matrix, connected, n >= 2.

    Returns
    -------
    result : dict
        Keys: estimate (max closeness), closeness (per vertex),
        argmax (0-based), n, method.

    References
    ----------
    Wasserman & Faust (1994), Ch 5.3.2.

    Examples
    --------
    Path a-b-c: centre has distances 1+1, ends 1+2.

    >>> path = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> out = sgt_closeness_centrality(path)
    >>> out["closeness"]
    [0.6666666666666666, 1.0, 0.6666666666666666]
    >>> out["estimate"]
    1.0
    >>> out["argmax"]
    1
    >>> sgt_closeness_centrality([[0, 0], [0, 0]])
    Traceback (most recent call last):
        ...
    ValueError: closeness needs a connected graph; vertex 0 cannot reach vertex 1.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"the adjacency matrix must be square; got shape {A.shape}.")
    n = A.shape[0]
    if n < 2:
        raise ValueError("closeness needs at least 2 vertices.")
    if not np.array_equal(A, A.T):
        raise ValueError("the adjacency matrix must be symmetric (undirected graph).")
    nbrs = [np.flatnonzero(A[i]) for i in range(n)]
    clos = []
    for v in range(n):
        dist = np.full(n, -1, dtype=int)
        dist[v] = 0
        queue = [v]
        while queue:
            nxt = []
            for u in queue:
                for w in nbrs[u]:
                    if dist[w] < 0:
                        dist[w] = dist[u] + 1
                        nxt.append(w)
            queue = nxt
        if np.any(dist < 0):
            far = int(np.flatnonzero(dist < 0)[0])
            raise ValueError(f"closeness needs a connected graph; vertex {v} cannot reach vertex {far}.")
        clos.append(float((n - 1) / np.sum(dist)))
    arg = int(np.argmax(clos))
    return RichResult(payload={
        "estimate": float(clos[arg]), "closeness": clos, "argmax": arg,
        "n": int(n),
        "method": "closeness (n-1)/sum BFS distances; connected required"})


def cheatsheet():
    return "sgtclo: C(v) = (n-1)/sum_u d(v,u), BFS; disconnected refused"
