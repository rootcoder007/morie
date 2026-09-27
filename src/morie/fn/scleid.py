# morie.fn -- slice s03 (rootcoder007/morie)
"""Leiden community detection (Traag, Waltman and van Eck 2019, Algorithm A.2).

Source: Traag, V. A., Waltman, L. and van Eck, N. J. (2019).  From Louvain
to Leiden: guaranteeing well-connected communities.  *Scientific Reports*
9, 5233 (arXiv:1810.08473), supplementary Algorithm A.2.  Each iteration
runs, until every community of the aggregate graph is a single node:

  FastMoveNodes      queue-based local moving, visiting nodes in random order;
                     a node moves to the neighbouring (or empty) community of
                     largest quality gain when that gain beats staying, and its
                     neighbours outside the new community are re-queued;
  RefinePartition    inside each community S, starting from singletons, every
                     well-connected node v (E(v, S - v) >= r ||v|| (||S|| - ||v||))
                     still alone joins a well-connected sub-community C with
                     probability proportional to exp(dH / theta), dH >= 0;
  AggregateGraph     refined sub-communities become nodes, and the aggregate
                     partition is the unrefined one.

Both quality functions share the form H = sum_c [ e_c - r W_c^2 / 2 ]: the
constant Potts model has node weight 1 and r = gamma (paper eq. 2);
modularity has node weight = strength and r = gamma / (2m).  Moving v
into community D gains k_{v,D} - r w_v W_D.

RANDOMNESS.  The visiting orders and the refinement draws come from the morie
Philox generator: every FastMoveNodes call and every refined community takes
the next stream number (seed fixed), consumed in the same order in both
language arms, so runs agree exactly across arms.
"""

from __future__ import annotations

import math

from . import _s03core as k
from ._richresult import RichResult
from ._rng import random_uniform


def _ssum(it):
    # plain left-to-right summation: sum() of floats is compensated from Python 3.12 on, which
    # would make results depend on the Python version and differ from the R arm
    s = 0.0
    for v in it:
        s += v
    return s


__all__ = ["leiden_clustering"]


class _Draws:
    def __init__(self, seed):
        self.seed = int(seed)
        self.stream = 0

    def take(self, n):
        self.stream += 1
        return [float(u) for u in random_uniform(n, seed=self.seed, stream=self.stream)] if n else []


def _order(us):
    return sorted(range(len(us)), key=lambda i: (us[i], i))


def _fast_move(adj, w, r, comm, draws):
    n = len(adj)
    tot = {}
    for v in range(n):
        tot[comm[v]] = tot.get(comm[v], 0.0) + w[v]
    free = [c for c in range(n) if c not in tot]
    queue = _order(draws.take(n))
    inq = [True] * n
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        inq[v] = False
        cur = comm[v]
        tot[cur] -= w[v]
        if tot[cur] <= 0.0 and cur not in free:
            free.append(cur)
        links = {}
        for u, a in adj[v].items():
            if u != v:
                links[comm[u]] = links.get(comm[u], 0.0) + a
        best, bgain = cur, links.get(cur, 0.0) - r * w[v] * tot[cur]
        for c in sorted(links):
            g = links[c] - r * w[v] * tot[c]
            if g > bgain + 1e-12:
                best, bgain = c, g
        if bgain < -1e-12:  # an empty community gains 0 (cur is not empty here, so free is not)
            best = min(free)
        if best in free:
            free.remove(best)
        comm[v] = best
        tot[best] = tot.get(best, 0.0) + w[v]
        if best != cur:
            for u in adj[v]:
                if u != v and comm[u] != best and not inq[u]:
                    queue.append(u)
                    inq[u] = True
    return comm


def _refine(adj, w, r, comm, theta, draws):
    n = len(adj)
    ref = list(range(n))
    groups = {}
    for v in range(n):
        groups.setdefault(comm[v], []).append(v)
    for c in sorted(groups):
        S = groups[c]
        inS = set(S)
        WS = _ssum(w[v] for v in S)
        ext = {v: _ssum(a for u, a in adj[v].items() if u in inS and u != v) for v in S}
        cw = {v: w[v] for v in S}
        cext = dict(ext)  # E(C, S - C) per refined sub-community, keyed by its id
        single = {v: True for v in S}
        us = draws.take(2 * len(S))
        for t, i in enumerate(_order(us[: len(S)])):
            v = S[i]
            if not single[v] or ext[v] < r * w[v] * (WS - w[v]) - 1e-12:
                continue
            links = {}
            for u, a in adj[v].items():
                if u in inS and u != v:
                    links[ref[u]] = links.get(ref[u], 0.0) + a
            cand, gains = [ref[v]], [0.0]  # staying alone gains 0
            for C in sorted(links):
                if cext[C] < r * cw[C] * (WS - cw[C]) - 1e-12:
                    continue
                g = links[C] - r * w[v] * cw[C]
                if g >= 0.0:
                    cand.append(C)
                    gains.append(g)
            gm = max(gains)
            ps = [math.exp((g - gm) / theta) for g in gains]
            s = _ssum(ps)
            x, acc, pick = us[len(S) + t] * s, 0.0, cand[-1]
            for C, p in zip(cand, ps):
                acc += p
                if x < acc:
                    pick = C
                    break
            old = ref[v]
            if pick == old:
                continue
            cw[old] -= w[v]
            ref[v] = pick
            single[v] = False
            for u in S:
                if ref[u] == pick and u != v:
                    single[u] = False
            # E(pick, S - pick) gains v's external links and loses the links between v and pick
            cext[pick] = cext[pick] + ext[v] - 2 * links[pick]
            cw[pick] += w[v]
    return ref


def _canon(lab):
    ids, out = {}, []
    for c in lab:
        out.append(ids.setdefault(c, len(ids)))
    return out


def _quality(W, lab, gamma, quality):
    n = len(W)
    e = _ssum(W[i][j] for i in range(n) for j in range(n) if lab[i] == lab[j]) / 2
    if quality == "cpm":
        size = {}
        for c in lab:
            size[c] = size.get(c, 0) + 1
        return e - gamma * _ssum(s * (s - 1) / 2 for s in size.values())
    m2 = _ssum(_ssum(row) for row in W)
    K = {}
    for i in range(n):
        K[lab[i]] = K.get(lab[i], 0.0) + _ssum(W[i])
    return 2 * e / m2 - gamma * _ssum(v * v for v in K.values()) / (m2 * m2)


def _connected(W, lab):
    n = len(W)
    for c in set(lab):
        mem = [v for v in range(n) if lab[v] == c]
        seen, stack = {mem[0]}, [mem[0]]
        while stack:
            u = stack.pop()
            for v in mem:
                if v not in seen and W[u][v] > 0:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != len(mem):
            return False
    return True


def leiden_clustering(graph, resolution=1.0, quality="modularity", max_iter=20, theta=0.01, seed=0):
    """Leiden community detection (Traag et al. 2019, Algorithm A.2).

    Parameters
    ----------
    graph : square matrix
        Symmetric non-negative weighted adjacency matrix.
    resolution : float
        gamma.
    quality : {"modularity", "cpm"}
        Modularity (node weight = strength, r = gamma / 2m) or the constant
        Potts model (node weight 1, r = gamma).
    max_iter : int
        Leiden iterations; stops early once an iteration changes nothing.
    theta : float
        Randomness of the refinement merge (igraph's beta).
    seed : int
        Philox seed.

    Returns
    -------
    RichResult
        labels, estimate (quality), quality, n_communities, connected, passes, n, method.

    References
    ----------
    Traag, V. A., Waltman, L. and van Eck, N. J. (2019). Scientific Reports 9, 5233.

    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
    ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> leiden_clustering(A)["labels"]
    [0, 0, 0, 1, 1, 1]
    """
    if quality not in ("modularity", "cpm"):
        raise ValueError('quality must be "modularity" or "cpm"')
    W = [[float(v) for v in row] for row in k.mat(graph)]
    n = len(W)
    if n == 0 or any(len(row) != n for row in W):
        raise ValueError("graph must be a non-empty square matrix")
    if any(W[i][j] < 0 or abs(W[i][j] - W[j][i]) > 1e-12 for i in range(n) for j in range(n)):
        raise ValueError("graph must be symmetric with non-negative weights")
    g = float(resolution)
    m2 = _ssum(_ssum(row) for row in W)
    if m2 <= 0:
        lab = list(range(n))
    else:
        draws = _Draws(seed)
        lab = list(range(n))
        passes = 0
        for _ in range(int(max_iter)):
            passes += 1
            adj = [{j: W[i][j] for j in range(n) if W[i][j] != 0.0} for i in range(n)]
            w = [_ssum(W[i]) for i in range(n)] if quality == "modularity" else [1.0] * n
            r = g / m2 if quality == "modularity" else g
            member = [[v] for v in range(n)]
            comm = list(lab)
            while True:
                comm = _fast_move(adj, w, r, comm, draws)
                if len(set(comm)) == len(adj):
                    break
                ref = _refine(adj, w, r, comm, float(theta), draws)
                rid = {}
                for c in ref:
                    rid.setdefault(c, len(rid))
                na = len(rid)
                nadj = [dict() for _ in range(na)]
                nw = [0.0] * na
                nmem = [[] for _ in range(na)]
                ncomm = [0] * na
                for v in range(len(adj)):
                    a = rid[ref[v]]
                    nw[a] += w[v]
                    nmem[a].extend(member[v])
                    ncomm[a] = comm[v]
                    for u, x in adj[v].items():
                        b = rid[ref[u]]
                        nadj[a][b] = nadj[a].get(b, 0.0) + x
                adj, w, member, comm = nadj, nw, nmem, ncomm
            new = [0] * n
            for c, mem in zip(comm, member):
                for v in mem:
                    new[v] = c
            new = _canon(new)
            if new == lab:
                break
            lab = new
    lab = _canon(lab)
    q = _quality(W, lab, g, quality) if m2 > 0 else 0.0
    nc = len(set(lab))
    return RichResult(
        title="Leiden community detection",
        summary_lines=[("communities", nc), ("quality", q)],
        payload={
            "labels": lab,
            "estimate": q,
            "quality": q,
            "n_communities": nc,
            "connected": _connected(W, lab),
            "passes": passes if m2 > 0 else 0,
            "n": n,
            "method": "Leiden (Traag, Waltman and van Eck 2019, Algorithm A.2), Philox visiting order and refinement",
        },
    )


def cheatsheet():
    return "scleid: Leiden community detection (fast local moving, refinement, aggregation)"
