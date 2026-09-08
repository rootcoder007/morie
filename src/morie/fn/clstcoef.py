# morie.fn -- function file (rootcoder007/morie)
"""Local and average clustering coefficient."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["clustering_coefficient"]


def clustering_coefficient(G):
    """Fraction of a node neighbours that are themselves connected.

    The measure exists because real networks are not random graphs: they
    have far more triangles than degree alone predicts.  A node of
    degree one or zero has no pair of neighbours to close, so it has no
    defined coefficient; those nodes are excluded from the average
    rather than counted as zero, which is the convention that keeps the
    average from drifting down with every leaf added.

    Formula: ``C_v = 2 e(N_v) / (k_v (k_v - 1))``, and the graph
    average is the mean of ``C_v`` over nodes with ``k_v >= 2``.

    Parameters
    ----------
    G : array-like, shape (n, n)
        Symmetric adjacency matrix; non-zero means an edge.

    Returns
    -------
    RichResult
        ``estimate`` (average clustering), ``local``, ``degree``,
        ``n_defined``, ``n``.

    References
    ----------
    Watts, D. J. & Strogatz, S. H. (1998).  Collective dynamics of
    small-world networks.  Nature 393:440-442.
    """
    A = C.mat(G)
    n = len(A)
    adj = [[1 if (A[i][j] != 0.0 or A[j][i] != 0.0) and i != j else 0
            for j in range(n)] for i in range(n)]
    deg = [sum(adj[i]) for i in range(n)]
    local, defined = [], []
    for v in range(n):
        nb = [u for u in range(n) if adj[v][u]]
        k = len(nb)
        if k < 2:
            local.append(float("nan"))
            continue
        e = sum(adj[nb[a]][nb[b]] for a in range(k) for b in range(a + 1, k))
        cv = 2.0 * e / (k * (k - 1.0))
        local.append(cv)
        defined.append(cv)
    avg = sum(defined) / len(defined) if defined else float("nan")
    return RichResult(payload={
        "estimate": avg, "local": local, "degree": deg,
        "n_defined": len(defined), "n": n,
        "method": "Watts-Strogatz clustering coefficient"})


def cheatsheet():
    return "clstcoef: Local and average clustering coefficient."
