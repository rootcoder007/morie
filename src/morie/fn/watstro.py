# morie.fn -- function file (rootcoder007/morie)
"""Watts-Strogatz small-world graph."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["watts_strogatz"]


def watts_strogatz(n, k, p, seed=1):
    """Rewire a ring lattice and watch the diameter collapse.

    The finding was that a graph does not have to look random to behave
    randomly: rewiring only a percent or two of a regular lattice leaves
    the clustering essentially intact while the average path length
    falls to near its random-graph value.  A handful of shortcuts is
    enough, which is why small-world structure turns up in networks that
    are otherwise strongly local.

    Determinism: edges are visited in a fixed order and the rewiring
    draws come from the shared Lehmer minstd stream, so a given seed
    gives the same graph in both language arms.

    Formula: start from a ring where each node joins its ``k`` nearest
    neighbours, then rewire each edge with probability ``p`` to a
    uniformly chosen node, refusing self-loops and duplicates.

    Parameters
    ----------
    n : int
        Nodes.
    k : int
        Even; each node starts joined to its ``k`` nearest neighbours.
    p : float
        Rewiring probability.
    seed : int, default 1
        Seed for the shared generator.

    Returns
    -------
    RichResult
        ``A`` (adjacency), ``estimate`` (mean degree), ``n_rewired``,
        ``n_edges``, ``n``.

    References
    ----------
    Watts, D. J. & Strogatz, S. H. (1998).  Collective dynamics of
    small-world networks.  Nature 393:440-442.
    """
    n = int(n)
    k = int(k)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(1, k // 2 + 1):
            t = (i + j) % n
            A[i][t] = A[t][i] = 1.0
    g = C.Lcg(seed)
    rew = 0
    for j in range(1, k // 2 + 1):
        for i in range(n):
            t = (i + j) % n
            if A[i][t] == 0.0:
                continue
            if g.unif() < p:
                cand = int(g.unif() * n)
                if cand >= n:
                    cand = n - 1
                if cand == i or A[i][cand] != 0.0:
                    continue
                A[i][t] = A[t][i] = 0.0
                A[i][cand] = A[cand][i] = 1.0
                rew += 1
    edges = sum(sum(row) for row in A) / 2.0
    return RichResult(payload={
        "A": A, "estimate": 2.0 * edges / n, "n_rewired": rew,
        "n_edges": edges, "n": n,
        "method": "Watts-Strogatz small-world graph"})


wattsstrogatz = watts_strogatz


def cheatsheet():
    return "watstro: Watts-Strogatz small-world graph."
