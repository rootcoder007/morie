# morie.fn -- slice s03 (rootcoder007/morie)
"""Local clustering coefficient.

Source consulted: Watts, D. J. and Strogatz, S. H. (1998).  Collective
dynamics of 'small-world' networks.  *Nature* 393, 440-442, which
defines, for a vertex v with k_v neighbours,

    C_v = (number of edges among the neighbours of v) / ( k_v (k_v - 1) / 2 )

and the network coefficient C as the average of C_v over all vertices.
The *Nature* paper is paywalled; the definition is quoted in its
standard published form.

The global *transitivity* of Barrat and Weigt (2000) -- three times the
number of triangles over the number of connected triples -- is a
different quantity and is returned separately, because the two are
routinely confused and disagree on any graph with an uneven degree
distribution.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["clustering_coefficient"]


def clustering_coefficient(y, A=None, node=None):
    """C_v for one vertex, plus the average and the global transitivity.

    Parameters
    ----------
    y : 2-D array-like
        The adjacency matrix.  (First slot, for signature stability.)
    A : 2-D array-like, optional
        The adjacency matrix; wins over ``y``.
    node : int, optional
        The vertex whose C_v is returned as ``estimate``.

    Returns
    -------
    RichResult with payload:
        estimate     : C_v for ``node`` (or the average when node is None)
        local        : C_v for every vertex
        average      : mean of C_v over vertices with degree at least 2
        transitivity : 3 * triangles / connected triples
    """
    W = k.mat(A if A is not None else y)
    n = len(W)
    loc = []
    tri = 0.0
    trip = 0.0
    for v in range(n):
        nb = [u for u in range(n) if u != v and W[v][u] != 0.0]
        kv = len(nb)
        links = 0.0
        for a in range(kv):
            for b in range(a + 1, kv):
                if W[nb[a]][nb[b]] != 0.0:
                    links += 1.0
        tri += links
        trip += kv * (kv - 1.0) / 2.0
        loc.append(2.0 * links / (kv * (kv - 1.0)) if kv > 1 else 0.0)
    good = [loc[v] for v in range(n)
            if len([u for u in range(n) if u != v and W[v][u] != 0.0]) > 1]
    avg = k.mean(good) if good else float("nan")
    trans = tri / trip if trip > 0.0 else float("nan")
    est = loc[int(node)] if node is not None else avg
    return RichResult(
        title="Clustering coefficient",
        summary_lines=[("average C", avg), ("transitivity", trans)],
        payload={
            "estimate": est,
            "local": loc,
            "average": avg,
            "transitivity": trans,
            "n": n,
            "method": "Watts-Strogatz local clustering coefficient, with global transitivity",
        },
    )


def cheatsheet():
    return "clusca: Local clustering coefficient (transitivity)"
