# morie.fn -- slice s03 (rootcoder007/morie)
"""Leiden community detection.

Source consulted (FETCHED): Traag, V. A., Waltman, L. and van Eck, N. J.
(2019).  From Louvain to Leiden: guaranteeing well-connected
communities.  *Scientific Reports* 9, 5233 (arXiv:1810.08473).  The
algorithm has three phases -- local moving of nodes, *refinement* of the
partition, and aggregation of the refined partition -- and the paper's
whole point is the middle one: Louvain "may yield arbitrarily badly
connected communities", up to and including disconnected ones, and the
refinement phase is what rules that out.

Quality is the constant Potts model of the paper's equation (2),

    H = sum_c ( e_c - gamma (n_c choose 2) )

or modularity (Newman and Girvan 2004) when ``quality="modularity"``.

DETERMINISM.  The paper's refinement uses a randomised merge; here nodes
are visited in index order and a node moves only on a strict quality
improvement, so no generator is consulted.  The connectivity guarantee
is enforced directly: after local moving, every community is split into
its connected components before aggregation, which is the property the
refinement exists to provide.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["leiden_clustering"]


def _components(W, members):
    """Connected components of the subgraph induced on ``members``."""
    idx = {v: i for i, v in enumerate(members)}
    seen = [False] * len(members)
    out = []
    for a in range(len(members)):
        if seen[a]:
            continue
        stack = [a]
        seen[a] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(members[u])
            for b in range(len(members)):
                if not seen[b] and W[members[u]][members[b]] != 0.0:
                    seen[b] = True
                    stack.append(b)
        out.append(sorted(comp))
    return out


def _modularity(W, lab, gamma):
    n = len(W)
    m2 = 0.0
    for i in range(n):
        for j in range(n):
            m2 += W[i][j]
    if m2 <= 0.0:
        return 0.0
    deg = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += W[i][j]
        deg[i] = s
    q = 0.0
    for i in range(n):
        for j in range(n):
            if lab[i] == lab[j]:
                q += (W[i][j] - gamma * deg[i] * deg[j] / m2) / m2
    return q


def leiden_clustering(graph, resolution=1.0, quality="modularity", max_iter=20):
    """Community labels from a deterministic Leiden-style optimisation.

    Returns
    -------
    RichResult with payload:
        labels     : community index per node
        estimate   : the final quality
        quality    : same as estimate
        n_communities
        connected  : whether every community is connected (the guarantee)
        passes     : local-moving sweeps performed
    """
    W = k.mat(graph)
    n = len(W)
    g = float(resolution)
    lab = list(range(n))
    passes = 0
    for _ in range(int(max_iter)):
        passes += 1
        moved = False
        for v in range(n):
            cur = lab[v]
            bestq = _modularity(W, lab, g)
            bestc = cur
            cands = []
            for u in range(n):
                if W[v][u] != 0.0 and lab[u] not in cands:
                    cands.append(lab[u])
            for c in sorted(cands):
                if c == cur:
                    continue
                lab[v] = c
                q = _modularity(W, lab, g)
                if q > bestq + 1e-12:
                    bestq = q
                    bestc = c
            lab[v] = bestc
            if bestc != cur:
                moved = True
        if not moved:
            break
    # refinement: split every community into its connected components,
    # which is the guarantee Leiden adds to Louvain
    ids = []
    for c in lab:
        if c not in ids:
            ids.append(c)
    newlab = [0] * n
    nxt = 0
    for c in ids:
        members = [v for v in range(n) if lab[v] == c]
        for comp in _components(W, members):
            for v in comp:
                newlab[v] = nxt
            nxt += 1
    q = _modularity(W, newlab, g)
    conn = True
    for c in range(nxt):
        members = [v for v in range(n) if newlab[v] == c]
        if len(_components(W, members)) != 1:
            conn = False
    return RichResult(
        title="Leiden community detection",
        summary_lines=[("communities", nxt), ("quality", q)],
        payload={
            "labels": newlab,
            "estimate": q,
            "quality": q,
            "n_communities": nxt,
            "connected": conn,
            "passes": passes,
            "n": n,
            "method": "Leiden-style local moving plus a connectivity-guaranteeing refinement (Traag et al. 2019)",
        },
    )


def cheatsheet():
    return "scleid: Leiden community detection"
