# morie.fn -- k02 batch (rootcoder007/morie)
"""Louvain community detection, all levels.

Source consulted: Blondel, V.D., Guillaume, J.-L., Lambiotte, R. and
Lefebvre, E. (2008), Fast unfolding of communities in large networks,
*Journal of Statistical Mechanics* P10008.  The method alternates the local
moving phase (each node joins the neighbouring community with the largest
positive modularity gain) with an aggregation phase that contracts each
community to a single node carrying self-loops of the internal weight, and
stops when a whole pass produces no move.  Deterministic here: nodes are
visited in index order and ties go to the smallest community label.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02mod
from .sgtcoml import _phase1, _relabel

from ._richresult import RichResult

__all__ = ["louvain_communities"]


def louvain_communities(A, max_levels=20):
    """Multi-level Louvain partition.

    Parameters
    ----------
    A : array-like
        Symmetric adjacency (or weight) matrix.
    max_levels : int, default 20
        Cap on aggregation levels.

    Returns
    -------
    RichResult
        estimate (final modularity), communities, n_communities, levels,
        modularity_by_level, n, method.
    """
    a0 = np.atleast_2d(np.asarray(A, dtype=float))
    n0 = a0.shape[0]
    m2 = float(np.sum(a0))
    if m2 <= 0.0:
        return RichResult(
            payload={
                "estimate": 0.0,
                "communities": list(range(n0)),
                "n_communities": int(n0),
                "levels": 0,
                "modularity_by_level": [],
                "n": int(n0),
                "method": "Louvain community detection (Blondel et al. 2008)",
            }
        )
    labels = list(range(n0))
    a = a0
    hist = []
    levels = 0
    for _ in range(int(max_levels)):
        n = a.shape[0]
        comm, _sw = _phase1(a, n, m2)
        comm = _relabel(comm)
        if len(set(comm)) == n:
            break
        labels = [comm[labels[i]] for i in range(n0)]
        hist.append(float(k02mod(a0, labels)))
        levels += 1
        nc = len(set(comm))
        agg = np.zeros((nc, nc))
        for i in range(n):
            for j in range(n):
                agg[comm[i], comm[j]] += float(a[i, j])
        a = agg
    return RichResult(
        payload={
            "estimate": float(k02mod(a0, labels)),
            "communities": _relabel(labels),
            "n_communities": int(len(set(labels))),
            "levels": int(levels),
            "modularity_by_level": hist,
            "n": int(n0),
            "method": "Louvain community detection (Blondel, Guillaume, Lambiotte & Lefebvre 2008)",
        }
    )


# CANONICAL TEST
# >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
# ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
# >>> r = louvain_communities(A)
# >>> assert abs(r["estimate"] - 0.357142857142857) < 1e-12  # igraph cluster_louvain
# >>> assert r["communities"][:3] == [0, 0, 0] and r["communities"][3:] == [1, 1, 1]


def cheatsheet():
    return "louv(A): Louvain community detection, all levels."


louvaincommunities = louvain_communities
