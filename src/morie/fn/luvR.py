"""Louvain community detection with a resolution parameter."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["luvR", "louvain"]


def _phase1_gamma(a, n, m2, gamma):
    """Local-moving phase with resolution gamma; deterministic.

    Nodes are visited in index order; the candidate communities are
    scanned in ascending label order and a move needs a strict gain
    (> 1e-12), so ties stay put and no randomness is involved.
    """
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
            bestgain = links[ci] / m2 - gamma * tot[ci] * kdeg[i] / (m2 * m2)
            for c in sorted(links):
                g = links[c] / m2 - gamma * tot[c] * kdeg[i] / (m2 * m2)
                if g > bestgain + 1e-12:
                    bestgain = g
                    best = c
            tot[best] += kdeg[i]
            if best != ci:
                comm[i] = best
                moved = True
    return comm


def _relabel(comm):
    seen = {}
    out = []
    for c in comm:
        if c not in seen:
            seen[c] = len(seen)
        out.append(seen[c])
    return out


def _modularity_gamma(a, n, m2, lab, gamma):
    deg = [float(t) for t in np.sum(a, axis=1)]
    q = 0.0
    for i in range(n):
        for j in range(n):
            if lab[i] == lab[j]:
                q += (float(a[i, j]) - gamma * deg[i] * deg[j] / m2) / m2
    return q


def luvR(A, resolution=1.0, max_levels=20):
    """
    Multi-level Louvain with the Reichardt-Bornholdt resolution gamma.

    Maximizes the generalized modularity

        Q_gamma = (1/2m) sum_ij [ A_ij - gamma k_i k_j / 2m ] delta(c_i, c_j)

    by alternating (i) a local-moving phase in which each node joins the
    neighbouring community with the largest strictly positive gain, and
    (ii) an aggregation phase contracting each community into one node
    that carries the internal weight as a self-loop; the process stops
    when a level no longer improves Q_gamma. gamma = 1 recovers plain
    Louvain (Blondel et al. Sec. 2; the gain formula is their
    "gain in modularity" expression with the gamma-scaled degree term).

    Deterministic: nodes visited in index order, ties keep the current
    community (cf. morie.fn.louv, whose gamma = 1 special case this
    reproduces).

    Sources
    -------
    Blondel, V. D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E.
    (2008). Fast unfolding of communities in large networks. *J. Stat.
    Mech.*, P10008, Sec. 2, arXiv:0803.0476
    (fetched-wave3/blondel-2008-fast-unfolding-louvain.pdf).
    Reichardt, J. & Bornholdt, S. (2006). Statistical mechanics of
    community detection. *Physical Review E*, 74, 016110 (the gamma
    resolution parameter in the null-model term).

    Parameters
    ----------
    A : array-like, (n, n)
        Symmetric adjacency (or weight) matrix.
    resolution : float
        gamma; larger values favour more, smaller communities.
    max_levels : int
        Cap on aggregation levels.

    Returns
    -------
    RichResult
        Keys: communities (labels), estimate (final Q_gamma),
        n_communities, levels, modularity_by_level.
    """
    a = np.atleast_2d(np.asarray(A, dtype=float))
    n = a.shape[0]
    if a.shape[1] != n:
        raise ValueError("A must be square")
    gamma = float(resolution)
    m2 = float(np.sum(a))
    if m2 <= 0:
        raise ValueError("A must have positive total weight")
    mapping = list(range(n))
    cur = a
    qs = []
    labels = list(range(n))
    for level in range(int(max_levels)):
        nn = cur.shape[0]
        comm = _relabel(_phase1_gamma(cur, nn, m2, gamma))
        labels = [comm[mapping[i]] for i in range(n)]
        q = _modularity_gamma(a, n, m2, labels, gamma)
        if qs and q <= qs[-1] + 1e-12:
            break
        qs.append(q)
        k = max(comm) + 1
        if k == nn:
            break
        agg = np.zeros((k, k))
        for i in range(nn):
            for j in range(nn):
                agg[comm[i], comm[j]] += float(cur[i, j])
        cur = agg
        mapping = [comm[mapping[i]] for i in range(n)]
    return RichResult(payload={
        "communities": labels, "estimate": float(qs[-1]) if qs else 0.0,
        "n_communities": int(max(labels) + 1), "levels": len(qs),
        "modularity_by_level": [float(v) for v in qs], "n": int(n),
        "resolution": gamma,
        "method": "Louvain with resolution gamma (Blondel 2008 / Reichardt-Bornholdt 2006)",
    })


# long descriptive alias (stub-era name)
louvain = luvR


def cheatsheet():
    return "luvR: multi-level Louvain maximizing Q_gamma, deterministic sweep"
