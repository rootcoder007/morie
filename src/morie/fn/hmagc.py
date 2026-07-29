# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Agglomerative hierarchical clustering."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_agglomerative"]

_LINKAGES = ("single", "complete", "average", "centroid")


def geron_agglomerative(X, n_clusters=2, linkage="single"):
    """
    Agglomerative hierarchical clustering.

    Formula: iteratively merge closest clusters by linkage

    Parameters
    ----------
    X : array-like, shape (n, d)
        Observations.
    n_clusters : int
        Number of clusters to stop at; 1 <= n_clusters <= n.
    linkage : {"single", "complete", "average", "centroid"}
        Inter-cluster distance rule applied to Euclidean point distances.

    Returns
    -------
    result : RichResult
        Keys: labels, merges, heights, n_clusters, estimate, n, method.

    Examples
    --------
    >>> r = geron_agglomerative([[0.0], [1.0], [10.0], [11.0]], 2)
    >>> [int(v) for v in r["labels"]]
    [0, 0, 1, 1]
    >>> [round(float(h), 6) for h in r["heights"]]
    [1.0, 1.0]
    >>> r["merges"]
    [(0, 1), (2, 3)]

    Complete linkage merges the same pairs here but records the larger height
    for the final join:

    >>> rc = geron_agglomerative([[0.0], [1.0], [10.0], [11.0]], 1, linkage="complete")
    >>> [round(float(h), 6) for h in rc["heights"]]
    [1.0, 1.0, 11.0]

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_agglomerative: X must be 2-D, got ndim={A.ndim}")
    n = A.shape[0]
    if n == 0:
        raise ValueError("geron_agglomerative: X has no rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_agglomerative: X must be finite")
    k = int(n_clusters)
    if k < 1 or k > n:
        raise ValueError(f"geron_agglomerative: n_clusters must lie in [1, {n}], got {k}")
    if linkage not in _LINKAGES:
        raise ValueError(f"geron_agglomerative: linkage must be one of {_LINKAGES}, got {linkage!r}")

    diff = A[:, None, :] - A[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))

    members = {i: [i] for i in range(n)}
    active = list(range(n))
    merges = []
    heights = []

    def cluster_dist(a, b):
        sub = D[np.ix_(members[a], members[b])]
        if linkage == "single":
            return float(sub.min())
        if linkage == "complete":
            return float(sub.max())
        if linkage == "average":
            return float(sub.mean())
        ca = A[members[a]].mean(axis=0)
        cb = A[members[b]].mean(axis=0)
        return float(np.linalg.norm(ca - cb))

    while len(active) > k:
        best = None
        best_d = np.inf
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                a, b = active[ii], active[jj]
                dab = cluster_dist(a, b)
                if dab < best_d:
                    best_d = dab
                    best = (a, b)
        a, b = best
        merges.append((int(min(members[a])), int(min(members[b]))))
        heights.append(best_d)
        members[a] = sorted(members[a] + members[b])
        del members[b]
        active.remove(b)

    labels = np.empty(n, dtype=int)
    # Clusters are numbered by the index of their lowest-indexed member so the
    # labelling is deterministic regardless of merge order.
    for new_id, cid in enumerate(sorted(active, key=lambda c: min(members[c]))):
        labels[members[cid]] = new_id

    return RichResult(
        title="Agglomerative hierarchical clustering",
        summary_lines=[("Clusters", len(active)), ("Linkage", linkage), ("Last merge height", heights[-1] if heights else 0.0)],
        payload={
            "labels": labels,
            "merges": merges,
            "heights": heights,
            "n_clusters": int(len(active)),
            "linkage": linkage,
            "distances": D,
            "estimate": float(heights[-1]) if heights else 0.0,
            "n": int(n),
            "method": f"Agglomerative clustering with {linkage} linkage",
        },
    )


def cheatsheet():
    return "hmagc: Agglomerative hierarchical clustering"
