# morie.fn -- function file (rootcoder007/morie)
"""Louvain greedy modularity communities."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["louvain_communities"]


def modularity(A, z, resolution=1.0):
    """Q = (1/2m) sum_ij (A_ij - gamma k_i k_j / 2m) delta(z_i, z_j)."""
    n = len(A)
    k = [sum(A[i]) for i in range(n)]
    m2 = sum(k)
    if m2 <= 0.0:
        return 0.0
    q = 0.0
    for i in range(n):
        for j in range(n):
            if z[i] == z[j]:
                q += A[i][j] - resolution * k[i] * k[j] / m2
    return q / m2


def louvain_communities(G, resolution=1.0, max_pass=20):
    """
    Louvain community detection

    Formula: greedy modularity max via local moves

    Each node is moved to the neighbouring community that raises
    modularity most, repeatedly until no single move helps; the
    communities are then contracted into super-nodes and the sweep is
    repeated.  Moves are evaluated in a fixed node order with ties
    broken toward the lowest community index, so the partition is
    reproducible rather than merely good.

    Parameters
    ----------
    G : array-like
        n x n symmetric weighted adjacency matrix.
    resolution : float
        Resolution gamma; larger values give smaller communities.
    max_pass : int
        Cap on the number of aggregation passes.

    Returns
    -------
    result : dict
        Keys: estimate (modularity), z, counts, n_communities, Q, n.

    References
    ----------
    Blondel, Guillaume, Lambiotte & Lefebvre (2008), J. Stat. Mech.
    2008(10):P10008.
    """
    A0 = core.mat(G)
    n0 = len(A0)
    if n0 == 0:
        raise ValueError("empty input: G has no rows")
    if any(len(r) != n0 for r in A0):
        raise ValueError("G must be a square adjacency matrix")
    if not (resolution > 0.0):
        raise ValueError("resolution must be strictly positive")
    A = [[float(v) for v in r] for r in A0]
    member = list(range(n0))
    for _ in range(int(max_pass)):
        n = len(A)
        k = [sum(A[i]) for i in range(n)]
        m2 = sum(k)
        if m2 <= 0.0:
            break
        z = list(range(n))
        ktot = list(k)
        moved = True
        rounds = 0
        while moved and rounds < 50:
            moved = False
            rounds += 1
            for i in range(n):
                ci = z[i]
                ktot[ci] -= k[i]
                links = {}
                for j in range(n):
                    if j == i or A[i][j] == 0.0:
                        continue
                    links[z[j]] = links.get(z[j], 0.0) + A[i][j]
                best_c = ci
                best_g = links.get(ci, 0.0) - resolution * ktot[ci] * k[i] / m2
                for c in sorted(links):
                    g = links[c] - resolution * ktot[c] * k[i] / m2
                    if g > best_g + 1e-12:
                        best_g = g
                        best_c = c
                z[i] = best_c
                ktot[best_c] += k[i]
                if best_c != ci:
                    moved = True
        labs = sorted(set(z))
        remap = dict((c, j) for j, c in enumerate(labs))
        z = [remap[v] for v in z]
        K = len(labs)
        if K == n:
            member = [z[member[v]] for v in range(n0)]
            break
        member = [z[member[v]] for v in range(n0)]
        B = [[0.0] * K for _ in range(K)]
        for i in range(n):
            for j in range(n):
                B[z[i]][z[j]] += A[i][j]
        A = B
    labs = sorted(set(member))
    remap = dict((c, j) for j, c in enumerate(labs))
    member = [remap[v] for v in member]
    K = len(labs)
    counts = [sum(1 for v in member if v == c) for c in range(K)]
    Q = modularity(A0, member, resolution)
    return RichResult(payload={
        "estimate": Q,
        "z": member,
        "counts": counts,
        "n_communities": K,
        "Q": Q,
        "n": n0,
        "method": "Louvain greedy modularity communities",
    })


def cheatsheet():
    return "comlou: Louvain greedy modularity communities"


# compact alias per ledger/NAMING.md
louvaincommunities = louvain_communities
