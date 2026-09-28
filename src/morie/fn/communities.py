# morie.fn -- function file (rootcoder007/morie)
"""Agglomerative community detection on weighted graphs: Clauset-Newman-Moore fast greedy modularity and
Pons-Latapy walktrap, with Newman's modularity."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["graph_modularity", "fast_greedy_modularity", "walktrap_communities"]


def _adj(A):
    W = [[float(v) for v in row] for row in A]
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("adjacency must be square")
    for i in range(n):
        W[i][i] = 0.0
        for j in range(i):
            if W[i][j] != W[j][i] or W[i][j] < 0:
                raise ValueError("adjacency must be symmetric and non-negative")
    return W


def _labels_from_merges(n, merges, steps):
    comp = list(range(n))
    for a, b in merges[:steps]:
        comp = [a if c == b else c for c in comp]
    seen = {}
    return [seen.setdefault(c, len(seen) + 1) for c in comp]


def graph_modularity(A, membership) -> float:
    r"""Newman's modularity ``Q = sum_c (w_c / m - (s_c / (2m))^2)`` of a partition of a weighted graph.

    ``w_c`` is the total weight of edges inside community ``c``, ``s_c`` the
    summed strength of its vertices and ``m`` the total edge weight (loops ignored).

    References
    ----------
    Newman, M. E. J. (2004). Analysis of weighted networks. *Physical Review E*, 70, 056131.

    Examples
    --------
    >>> round(graph_modularity([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], [1, 1, 2, 2]), 12)
    0.5
    """
    W = _adj(A)
    n = len(W)
    m = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            m += W[i][j]
    q = 0.0
    for c in sorted(set(membership)):
        mem = [i for i in range(n) if membership[i] == c]
        win = 0.0
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                win += W[mem[a]][mem[b]]
        s = 0.0
        for i in mem:
            for j in range(n):
                s += W[i][j]
        q += win / m - (s / (2 * m)) ** 2
    return q


def fast_greedy_modularity(A) -> RichResult:
    r"""Fast greedy modularity optimisation (Clauset, Newman and Moore 2004), as ``igraph::cluster_fast_greedy``.

    Starting from singletons, the pair of adjacent communities with the
    largest ``Delta Q = 2 (e_ij - a_i a_j)`` merges (``e_ij`` half the fraction of
    edge weight between them, ``a_i`` the strength fraction); the partition
    with the highest modularity along the merge sequence is returned.

    References
    ----------
    Clauset, A., Newman, M. E. J. and Moore, C. (2004). Finding community
    structure in very large networks. *Physical Review E*, 70, 066111.

    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> fast_greedy_modularity(A).membership
    [1, 1, 1, 2, 2, 2]
    """
    W = _adj(A)
    n = len(W)
    m2 = 0.0
    for i in range(n):
        for j in range(n):
            m2 += W[i][j]
    e = {i: {j: W[i][j] / m2 for j in range(n) if W[i][j] > 0} for i in range(n)}
    a = [ssum(W[i]) / m2 for i in range(n)]
    q = 0.0
    for i in range(n):
        q -= a[i] * a[i]
    merges, qs = [], [q]
    alive = set(range(n))
    while True:
        best, bdq = None, -math.inf
        for i in sorted(alive):
            for j in sorted(e[i]):
                if j > i:
                    dq = 2 * (e[i][j] - a[i] * a[j])
                    if dq > bdq:
                        best, bdq = (i, j), dq
        if best is None:
            break
        i, j = best
        for k, v in e[j].items():
            if k == i:
                continue
            e[i][k] = e[i].get(k, 0.0) + v
            e[k][i] = e[k].get(i, 0.0) + v
            del e[k][j]
        del e[i][j]
        e[j] = {}
        a[i] += a[j]
        a[j] = 0.0
        alive.discard(j)
        q += bdq
        merges.append((i, j))
        qs.append(q)
    steps = max(range(len(qs)), key=lambda s: (qs[s], -s))
    return RichResult(
        payload={
            "membership": _labels_from_merges(n, merges, steps),
            "modularity": qs,
            "merges": [[i + 1, j + 1] for i, j in merges],
            "max_modularity": qs[steps],
        }
    )


def walktrap_communities(A, steps: int = 4) -> RichResult:
    r"""Walktrap community detection (Pons and Latapy 2006), as ``igraph::cluster_walktrap``.

    Every vertex gets a loop weighted by its mean edge weight; ``P^t_C`` is
    the distribution after ``steps`` random-walk steps started uniformly in
    community ``C``. Adjacent communities with the smallest
    ``Delta sigma = (1/n) |C1||C2| / (|C1| + |C2|) sum_k (P^t_C1k - P^t_C2k)^2 / d(k)``
    merge, and the partition of largest modularity is returned.

    References
    ----------
    Pons, P. and Latapy, M. (2006). Computing communities in large networks
    using random walks. *Journal of Graph Algorithms and Applications*,
    10(2), 191-218.

    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> walktrap_communities(A).membership
    [1, 1, 1, 2, 2, 2]
    """
    W = _adj(A)
    n = len(W)
    L = [list(row) for row in W]
    d = []
    for i in range(n):
        deg = sum(1 for j in range(n) if W[i][j] > 0)
        s = ssum(W[i])
        L[i][i] = 1.0 if deg == 0 else s / deg
        d.append(s + L[i][i])
    Pt = []
    for v in range(n):
        p = [0.0] * n
        p[v] = 1.0
        for _ in range(steps):
            q = [0.0] * n
            for i in range(n):
                if p[i] != 0.0:
                    pi = p[i] / d[i]
                    for j in range(n):
                        if L[i][j] > 0:
                            q[j] += pi * L[i][j]
            p = q
        Pt.append(p)
    comm = {i: {"P": Pt[i], "size": 1, "members": [i]} for i in range(n)}
    nbr = {i: {j for j in range(n) if W[i][j] > 0 and j != i} for i in range(n)}
    m = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            m += W[i][j]
    win = {i: 0.0 for i in range(n)}
    stot = {i: ssum(W[i]) for i in range(n)}
    q0 = 0.0
    for i in range(n):
        q0 -= (stot[i] / (2 * m)) ** 2
    qs, merges = [q0], []

    def dsig(a, b):
        ca, cb = comm[a], comm[b]
        s = 0.0
        for k in range(n):
            t = ca["P"][k] - cb["P"][k]
            s += t * t / d[k]
        return s * ca["size"] * cb["size"] / (ca["size"] + cb["size"]) / n

    nxt = n
    while True:
        best, bd = None, math.inf
        for a in sorted(comm):
            for b in sorted(nbr[a]):
                if b > a:
                    v = dsig(a, b)
                    if v < bd:
                        best, bd = (a, b), v
        if best is None:
            break
        a, b = best
        ca, cb = comm.pop(a), comm.pop(b)
        sa, sb = ca["size"], cb["size"]
        P = [(sa * ca["P"][k] + sb * cb["P"][k]) / (sa + sb) for k in range(n)]
        cross = 0.0
        for i in ca["members"]:
            for j in cb["members"]:
                cross += W[i][j]
        comm[nxt] = {"P": P, "size": sa + sb, "members": ca["members"] + cb["members"]}
        win[nxt] = win.pop(a) + win.pop(b) + cross
        stot[nxt] = stot.pop(a) + stot.pop(b)
        nbr[nxt] = (nbr.pop(a) | nbr.pop(b)) - {a, b}
        for c in nbr[nxt]:
            nbr[c] = (nbr[c] - {a, b}) | {nxt}
        q = 0.0
        for c in sorted(comm):
            q += win[c] / m - (stot[c] / (2 * m)) ** 2
        qs.append(q)
        merges.append((a, b, nxt))
        nxt += 1
    best_s = max(range(len(qs)), key=lambda s: (qs[s], -s))
    lab = list(range(n))
    for a, b, c in merges[:best_s]:
        lab = [c if v in (a, b) else v for v in lab]
    seen = {}
    return RichResult(
        payload={
            "membership": [seen.setdefault(v, len(seen) + 1) for v in lab],
            "modularity": qs,
            "merges": [[a + 1, b + 1] for a, b, _ in merges],
            "max_modularity": qs[best_s],
        }
    )


def cheatsheet() -> str:
    return "graph_modularity / fast_greedy_modularity / walktrap_communities -> community detection."
