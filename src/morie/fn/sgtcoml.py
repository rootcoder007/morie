# morie.fn -- k02 batch (rootcoder007/morie)
"""One local-moving pass of the Louvain method.

Source consulted: Blondel, V.D., Guillaume, J.-L., Lambiotte, R. and
Lefebvre, E. (2008), Fast unfolding of communities in large networks,
*Journal of Statistical Mechanics* P10008, section 2.  The first phase places
every node in its own community and then repeatedly moves each node to the
neighbouring community giving the largest positive modularity gain

    dQ = k_i_in / m - Sigma_tot k_i / (2 m^2)

where k_i_in is the total weight from i into the target community, Sigma_tot
the total incident weight of that community, and 2m the total edge weight.
Nodes are visited in index order and ties broken by the smallest community
label, so the pass is deterministic.  This function stops at the end of phase
one; ``louv`` runs the full multi-level algorithm.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02mod

from ._richresult import RichResult

__all__ = ["sgt_louvain_step"]


def _phase1(a, n, m2):
    comm = list(range(n))
    kdeg = [float(t) for t in np.sum(a, axis=1)]
    tot = list(kdeg)
    moved = True
    sweeps = 0
    while moved and sweeps < 100:
        moved = False
        sweeps += 1
        for i in range(n):
            ci = comm[i]
            tot[ci] -= kdeg[i]
            links = {}
            for j in range(n):
                if j != i and a[i, j] != 0.0:
                    links[comm[j]] = links.get(comm[j], 0.0) + float(a[i, j])
            links.setdefault(ci, 0.0)
            best = ci
            bestgain = links[ci] / m2 - tot[ci] * kdeg[i] / (m2 * m2)
            for c in sorted(links):
                g = links[c] / m2 - tot[c] * kdeg[i] / (m2 * m2)
                if g > bestgain + 1e-12:
                    bestgain = g
                    best = c
            tot[best] += kdeg[i]
            if best != ci:
                comm[i] = best
                moved = True
    return comm, sweeps


def _relabel(comm):
    seen = {}
    out = []
    for c in comm:
        if c not in seen:
            seen[c] = len(seen)
        out.append(seen[c])
    return out


def sgt_louvain_step(A):
    """A single Louvain local-moving phase.

    Parameters
    ----------
    A : array-like
        Symmetric adjacency (or weight) matrix.

    Returns
    -------
    RichResult
        estimate (modularity after the pass), communities, n_communities,
        modularity_before, sweeps, n, method.
    """
    a = np.atleast_2d(np.asarray(A, dtype=float))
    n = a.shape[0]
    m2 = float(np.sum(a))
    before = k02mod(a, list(range(n)))
    comm, sweeps = _phase1(a, n, m2) if m2 > 0.0 else (list(range(n)), 0)
    comm = _relabel(comm)
    return RichResult(
        payload={
            "estimate": float(k02mod(a, comm)),
            "communities": comm,
            "n_communities": int(len(set(comm))),
            "modularity_before": float(before),
            "sweeps": int(sweeps),
            "n": int(n),
            "method": "Louvain local-moving phase (Blondel, Guillaume, Lambiotte & Lefebvre 2008, sec. 2)",
        }
    )


# CANONICAL TEST
# >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
# ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
# >>> r = sgt_louvain_step(A)
# >>> assert r["n_communities"] == 2
# >>> assert abs(r["estimate"] - 0.357142857142857) < 1e-12   # igraph modularity
# >>> assert r["estimate"] > r["modularity_before"]


def cheatsheet():
    return "sgtcoml(A): one Louvain local-moving phase."


sgtlouvainstep = sgt_louvain_step
