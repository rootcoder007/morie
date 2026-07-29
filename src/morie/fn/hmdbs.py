# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Density-based spatial clustering (DBSCAN)."""

import numpy as np

from ._richresult import RichResult
from .grdbs import geron_dbscan_core_point

__all__ = ["geron_dbscan"]


def geron_dbscan(X, eps, min_samples, metric="euclidean"):
    """
    Density-based spatial clustering (DBSCAN).

    Formula: points with >=min_samples within eps are core; connect via
    eps-reachability

    Core/border/noise classification and the neighbourhood lists are
    DELEGATED to :func:`morie.fn.grdbs.geron_dbscan_core_point`. This
    module completes the algorithm: core points within ``eps`` of one
    another are merged into clusters by breadth-first expansion, and each
    border point joins the first cluster that reaches it.

    Two properties fall out and are reported rather than assumed:
    clusters need not be convex or equally sized (the reason DBSCAN beats
    k-means on rings), and noise is a genuine label -- ``-1`` -- not a
    tiny cluster. Border points are assignment-order dependent, which is
    DBSCAN's one real ambiguity; ``n_border`` says how many points are
    exposed to it.

    Parameters
    ----------
    X : array-like, shape (m, d)
    eps : float
        Neighbourhood radius, positive.
    min_samples : int
        Core-point density threshold, >= 1.
    metric : {"euclidean", "manhattan", "chebyshev"}, default "euclidean"

    Returns
    -------
    result : RichResult
        Keys: labels, n_clusters, n_noise, n_core, n_border, core_indices,
        cluster_sizes, is_core, neighbor_counts, estimate, n, method.

    Examples
    --------
    Two tight pairs and one outlier, at eps = 1 and min_samples = 2:

    >>> X = [[0.0], [0.5], [10.0], [10.5], [50.0]]
    >>> r = geron_dbscan(X, eps=1.0, min_samples=2)
    >>> r["labels"]
    [0, 0, 1, 1, -1]
    >>> r["n_clusters"], r["n_noise"]
    (2, 1)
    >>> r["cluster_sizes"]
    [2, 2]

    Raising the density requirement above the local density leaves nothing
    but noise:

    >>> geron_dbscan(X, eps=1.0, min_samples=3)["labels"]
    [-1, -1, -1, -1, -1]

    A chain of points is one cluster however non-convex it is -- each link
    only needs a neighbour within eps:

    >>> chain = [[0.0], [0.9], [1.8], [2.7]]
    >>> geron_dbscan(chain, eps=1.0, min_samples=2)["n_clusters"]
    1

    References
    ----------
    Géron Ch 8
    """
    base = geron_dbscan_core_point(X, eps=eps, min_samples=min_samples, metric=metric)
    is_core = np.asarray(base["is_core"], dtype=bool)
    neighbors = [list(v) for v in base["neighbors"]]
    m = is_core.size

    labels = np.full(m, -1, dtype=int)
    cid = 0
    for i in range(m):
        if not is_core[i] or labels[i] != -1:
            continue
        labels[i] = cid
        queue = [i]
        while queue:
            p = queue.pop(0)
            for q in neighbors[p]:
                if labels[q] == -1:
                    labels[q] = cid
                    if is_core[q]:
                        queue.append(q)
                elif is_core[q] and labels[q] != cid and is_core[p]:
                    # Two core points in one another's neighbourhood must share
                    # a cluster; BFS from a single seed makes this unreachable.
                    raise ValueError("geron_dbscan: internal inconsistency merging density-connected cores")
        cid += 1

    sizes = [int(np.sum(labels == c)) for c in range(cid)]
    n_border = int(np.sum((labels >= 0) & ~is_core))

    return RichResult(
        title="DBSCAN",
        summary_lines=[("Clusters", cid), ("Noise points", int(np.sum(labels == -1))), ("eps", float(eps))],
        interpretation="Clusters are density-connected components, so they need not be convex or equally sized; -1 means noise.",
        payload={
            "labels": labels.tolist(),
            "n_clusters": int(cid),
            "n_noise": int(np.sum(labels == -1)),
            "n_core": int(base["n_core"]),
            "n_border": n_border,
            "core_indices": np.flatnonzero(is_core).tolist(),
            "cluster_sizes": sizes,
            "is_core": is_core.tolist(),
            "neighbor_counts": list(base["neighbor_counts"]),
            "eps": float(eps),
            "min_samples": int(min_samples),
            "metric": metric,
            "estimate": float(cid),
            "n": int(m),
            "method": "DBSCAN by BFS over density-connected cores; core detection delegated to grdbs",
        },
    )


def cheatsheet():
    return "hmdbs: Density-based spatial clustering (DBSCAN)"
