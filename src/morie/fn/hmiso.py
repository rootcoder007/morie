# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Isomap: MDS on geodesic distances from kNN graph."""

import heapq

import numpy as np

from ._richresult import RichResult
from .hmmds import geron_mds, pairwise_distances

__all__ = ["geron_isomap"]

_METHOD = "Isomap (geodesic distances + classical MDS)"


def _dijkstra(adj, source, m):
    """Shortest paths from ``source`` over a sparse adjacency list."""
    dist = np.full(m, np.inf)
    dist[source] = 0.0
    seen = np.zeros(m, dtype=bool)
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if seen[u]:
            continue
        seen[u] = True
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def geron_isomap(X, n_components, n_neighbors=5):
    """
    Isomap: MDS on geodesic distances from kNN graph.

    Formula: D_geo via Dijkstra on kNN graph; MDS yields embedding

    The one idea: replace straight-line distance with distance *along
    the manifold*, approximated by shortest paths through a
    ``k``-nearest-neighbour graph.  On a rolled-up sheet two points on
    opposite faces are close in Euclidean terms and far along the sheet,
    and only the second is the honest distance.

    ``n_neighbors`` is the whole risk.  Too small and the graph is
    disconnected (infinite geodesics, and no embedding at all); too
    large and a "short-circuit" edge jumps between manifold folds and
    the geodesics collapse back to Euclidean.  A disconnected graph is
    raised on rather than patched with a large finite number, which
    would quietly change the answer.

    The graph is symmetrised (an edge is kept if either endpoint lists
    the other), Dijkstra is run from every node, and the resulting
    geodesic matrix is handed to
    :func:`morie.fn.hmmds.geron_mds` with ``precomputed=True``.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    n_components : int
        Embedding dimension.
    n_neighbors : int
        Neighbours per node, ``1 <= n_neighbors < m``.

    Returns
    -------
    result : RichResult
        Keys: embedding, geodesic_distances, eigenvalues, stress,
        n_neighbors, estimate, n, method.

    Examples
    --------
    Points evenly spaced on a line: with 2 neighbours the geodesic
    distance is the sum of the gaps, which equals the Euclidean
    distance here, so the embedding recovers the line:

    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> r = geron_isomap(X, n_components=1, n_neighbors=2)
    >>> emb = r["embedding"][:, 0]
    >>> [round(float(v), 9) for v in np.abs(emb - emb[0])]
    [0.0, 1.0, 2.0, 3.0]

    On a curve the geodesic exceeds the chord.  Three points on a right
    angle with 1-nearest-neighbour connectivity give a geodesic of 2
    between the ends where the chord is ``sqrt(2)``:

    >>> C = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    >>> g = geron_isomap(C, n_components=2, n_neighbors=1)
    >>> round(float(g["geodesic_distances"][0, 2]), 9)
    2.0
    >>> round(float(pairwise_distances(C)[0, 2]), 6)
    1.414214

    A graph too sparse to connect is refused:

    >>> geron_isomap([[0.0], [1.0], [100.0], [101.0]], n_components=1, n_neighbors=1)
    Traceback (most recent call last):
        ...
    ValueError: geron_isomap: the 1-nearest-neighbour graph is disconnected (2 components), so some geodesic distances are infinite; raise n_neighbors

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_isomap: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_isomap: X contains non-finite values")
    m = A.shape[0]
    k = int(n_neighbors)
    if not (1 <= k < m):
        raise ValueError(f"geron_isomap: n_neighbors must lie in 1..{m - 1}, got {n_neighbors!r}")

    D = pairwise_distances(A)
    adj = [[] for _ in range(m)]
    edge = np.zeros((m, m), dtype=bool)
    for i in range(m):
        nb = np.argsort(D[i], kind="mergesort")
        picked = [j for j in nb if j != i][:k]
        for j in picked:
            edge[i, j] = edge[j, i] = True
    for i in range(m):
        for j in np.flatnonzero(edge[i]):
            adj[i].append((int(j), float(D[i, j])))

    G = np.empty((m, m))
    for i in range(m):
        G[i] = _dijkstra(adj, i, m)
    if not np.all(np.isfinite(G)):
        # Count connected components for a message worth reading.
        seen = np.zeros(m, dtype=bool)
        comps = 0
        for i in range(m):
            if not seen[i]:
                comps += 1
                seen |= np.isfinite(G[i])
        raise ValueError(
            f"geron_isomap: the {k}-nearest-neighbour graph is disconnected ({comps} components), "
            f"so some geodesic distances are infinite; raise n_neighbors"
        )
    G = 0.5 * (G + G.T)
    np.fill_diagonal(G, 0.0)

    mds = geron_mds(G, n_components=n_components, precomputed=True)

    ratio = float(np.mean(G[np.triu_indices(m, 1)] / np.where(D[np.triu_indices(m, 1)] == 0, 1.0, D[np.triu_indices(m, 1)])))

    return RichResult(
        title="Isomap",
        summary_lines=[
            ("Points", int(m)),
            ("Neighbours", k),
            ("Mean geodesic / Euclidean", ratio),
            ("Stress", float(mds["stress"])),
        ],
        warnings=list(mds.warnings),
        interpretation=(
            "A geodesic/Euclidean ratio near 1 means the graph is short-circuiting across folds; "
            "far above 1 means the manifold really is curved."
        ),
        payload={
            "embedding": mds["embedding"],
            "geodesic_distances": G,
            "euclidean_distances": D,
            "eigenvalues": mds["eigenvalues"],
            "stress": float(mds["stress"]),
            "geodesic_ratio": ratio,
            "n_neighbors": k,
            "estimate": float(mds["stress"]),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmiso: Isomap -- kNN graph, Dijkstra geodesics, then classical MDS (delegates to hmmds)"
