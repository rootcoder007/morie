# morie.fn -- function file (rootcoder007/morie)
"""Leiden refinement step (Traag, Waltman & van Eck)."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sgt_leiden_step"]


def sgt_leiden_step(A, labels, gamma=1.0):
    """One Leiden refinement pass over an existing partition.

    Louvain can leave a community internally disconnected: a node that
    once glued two halves together may later move away, and nothing in
    the algorithm ever notices.  Leiden inserts a refinement phase
    between local moving and aggregation.  Each community of the input
    partition is broken back into singletons and rebuilt from the
    inside, and a node only joins a sub-community that is itself
    well-connected to the rest of its original community.  That is the
    condition which buys the guarantee, so it is the part implemented
    here.

    Determinism: nodes are visited in index order and communities in
    order of first appearance.  There are no random restarts and no
    randomised merge, so the same partition always comes back.

    Formula: quality is the Constant Potts Model
    ``H = sum_c [e_c - gamma * C(n_c, 2)]``; a subset ``C`` of the
    original community ``S`` is well connected when
    ``E(C, S \\ C) >= gamma * |C| * (|S| - |C|)``.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Non-negative weighted adjacency matrix; symmetrised on entry.
    labels : array-like, shape (n,)
        Current community index of each node.
    gamma : float, default 1.0
        CPM resolution.

    Returns
    -------
    RichResult
        ``labels_new`` (renumbered by first appearance), ``Q_new`` (CPM
        quality of the refined partition), ``n_communities``, ``n``.

    References
    ----------
    Traag, V. A., Waltman, L. & van Eck, N. J. (2019).  From Louvain to
    Leiden: guaranteeing well-connected communities.  Scientific
    Reports 9:5233.  Open access; fetched, and the CPM quality function
    and the well-connectedness condition above are the paper own.
    """
    A = C.mat(A)
    n = len(A)
    W = [[0.5 * (A[i][j] + A[j][i]) if i != j else 0.0 for j in range(n)] for i in range(n)]
    lab = [int(round(v)) for v in C.vec(labels)]
    seen = []
    for v in lab:
        if v not in seen:
            seen.append(v)
    comm = list(range(n))
    for s in seen:
        S = [i for i in range(n) if lab[i] == s]
        Sset = set(S)
        members = {i: [i] for i in S}
        for v in S:
            if len(members.get(comm[v], [])) != 1:
                continue
            ev = sum(W[v][u] for u in S if u != v)
            if ev < gamma * (len(S) - 1):
                continue
            best, bestd = None, 0.0
            for c in S:
                Cm = members.get(c)
                if not Cm or c == comm[v]:
                    continue
                Cset = set(Cm)
                out = sum(W[a][b] for a in Cm for b in S if b not in Cset)
                if out < gamma * len(Cm) * (len(S) - len(Cm)):
                    continue
                d = sum(W[v][u] for u in Cm) - gamma * len(Cm)
                if d > bestd:
                    bestd, best = d, c
            if best is not None:
                members[comm[v]] = []
                members[best].append(v)
                comm[v] = best
        del Sset
    order, newlab = [], [0] * n
    for i in range(n):
        if comm[i] not in order:
            order.append(comm[i])
        newlab[i] = order.index(comm[i])
    k = len(order)
    q = 0.0
    for c in range(k):
        mem = [i for i in range(n) if newlab[i] == c]
        e = sum(W[a][b] for ai, a in enumerate(mem) for b in mem[ai + 1:])
        q += e - gamma * len(mem) * (len(mem) - 1) / 2.0
    return RichResult(payload={
        "labels_new": newlab, "Q_new": q, "n_communities": k, "n": n,
        "method": "Leiden refinement phase, CPM quality"})


sgtleidenstep = sgt_leiden_step


def cheatsheet():
    return "sgtleid: Leiden refinement step (Traag, Waltman & van Eck)."
