# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Density-based clustering in the HDBSCAN style: mutual
reachability + single linkage + minimum cluster size (Campello et al.
2013; Alammar Ch 5)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_hdbscan_cluster"]


def alammar_hdbscan_cluster(X, min_cluster_size=3, min_samples=None):
    """Core distance = distance to the min_samples-th neighbour;
    mutual reachability = max(core_i, core_j, d_ij); single-linkage
    tree on that metric; flat clusters cut where components of size
    >= min_cluster_size separate; everything else is noise (-1).

    This is the geometric core of HDBSCAN (the stability-based flat
    cut is simplified to the cluster-count-maximising cut), stated as such rather
    than passed off as the full algorithm.

    References: Alammar and Grootendorst, Ch 5; Campello, Moulavi and
    Sander (2013).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    mcs = int(min_cluster_size)
    ms = int(min_samples) if min_samples is not None else mcs
    if mcs < 2:
        raise ValueError("min_cluster_size must be at least 2.")
    if not 1 <= ms < n:
        raise ValueError(
            f"min_samples must lie in [1, {n - 1}]; got {ms}.")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    core = np.sort(D, axis=1)[:, ms]     # ms-th neighbour, self at 0
    MR = np.maximum(np.maximum(core[:, None], core[None, :]), D)
    np.fill_diagonal(MR, 0.0)

    # single-linkage MST (Prim). The visited mask is load-bearing: a
    # plain np.minimum would overwrite the inf of nodes already in the
    # tree with their MR row and re-admit them, and every edge would
    # connect the same two points.
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    edges = []
    dist = MR[0].copy()
    src = np.zeros(n, dtype=int)
    dist[0] = np.inf
    for _ in range(n - 1):
        j = int(np.argmin(dist))
        edges.append((float(dist[j]), int(src[j]), j))
        visited[j] = True
        upd = (MR[j] < dist) & ~visited
        src[upd] = j
        dist[upd] = MR[j][upd]
        dist[visited] = np.inf
    edges.sort()
    # flat cut: try every distinct edge weight as a strict threshold
    # and keep the one yielding the MOST clusters of size >= mcs
    # (ties to the smaller threshold, which keeps clusters tight). A
    # single largest-gap cut fails on blobs + one far outlier: the
    # outlier gap dwarfs the blob gap and the blobs merge.
    def components(threshold):
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for w, a, b in edges:
            if w < threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        comp = {}
        for i in range(n):
            comp.setdefault(find(i), []).append(i)
        return sorted(comp.values(), key=lambda m: min(m))

    candidates = sorted({e[0] for e in edges}) + [np.inf]
    best = None
    threshold = np.inf
    for cand in candidates:
        comps = components(cand)
        score = sum(1 for m in comps if len(m) >= mcs)
        if best is None or score > best:
            best = score
            threshold = cand
    labels = [-1] * n
    lab = 0
    for members in components(threshold):
        if len(members) >= mcs:
            for i in members:
                labels[i] = lab
            lab += 1
    return RichResult(payload={
        "labels": labels, "n_clusters": lab,
        "n_noise": labels.count(-1),
        "core_distances": [float(v) for v in core],
        "cut_threshold": float(threshold),
        "estimate": float(lab), "n": n,
        "method": "Mutual-reachability single linkage with min cluster "
                  "size (Campello et al. 2013, simplified flat cut)"})


def cheatsheet():
    return "alhds: core distance, mutual reachability, MST, gap cut, noise = -1"
