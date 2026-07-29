# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BIRCH: balanced iterative reducing and clustering using hierarchies."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_birch"]


class _CF:
    """Clustering feature (N, LS, SS) -- the only statistics BIRCH keeps."""

    __slots__ = ("n", "ls", "ss", "members")

    def __init__(self, x, index):
        self.n = 1
        self.ls = np.array(x, dtype=float)
        self.ss = float(np.dot(x, x))
        self.members = [index]

    @property
    def centroid(self):
        return self.ls / self.n

    def radius_if_added(self, x):
        n = self.n + 1
        ls = self.ls + x
        ss = self.ss + float(np.dot(x, x))
        # R^2 = SS/N - ||LS/N||^2, clipped because rounding can drive it
        # microscopically negative for identical points.
        val = ss / n - float(np.dot(ls, ls)) / (n * n)
        return float(np.sqrt(max(val, 0.0)))

    def add(self, x, index):
        self.n += 1
        self.ls = self.ls + x
        self.ss += float(np.dot(x, x))
        self.members.append(index)

    @property
    def radius(self):
        val = self.ss / self.n - float(np.dot(self.ls, self.ls)) / (self.n * self.n)
        return float(np.sqrt(max(val, 0.0)))


def geron_birch(X, n_clusters=3, threshold=0.5, branching_factor=50):
    """
    BIRCH: balanced iterative reducing and clustering using hierarchies.

    Formula: CF-tree of clustering features (N, LS, SS)

    Phase 1 streams the data into leaf clustering features, absorbing a point
    into the nearest CF when the resulting radius stays under `threshold` and
    opening a new CF otherwise; `branching_factor` caps the entries per leaf
    node. Phase 3 clusters the CF centroids (centroid-linkage agglomerative)
    down to `n_clusters` and propagates the labels back to the points.

    ponytail: a single level of leaf nodes, not a full height-balanced tree.
    That changes insertion cost from O(log) to O(#leaves) but not the
    resulting sub-clusters; swap in a proper tree if the leaf count grows.

    Parameters
    ----------
    X : array-like, shape (n, d)
    n_clusters : int or None
        Final cluster count; None returns the sub-clusters themselves.
    threshold : float
        Maximum sub-cluster radius (positive).
    branching_factor : int
        Maximum CF entries per leaf node (>= 2).

    Returns
    -------
    result : RichResult
        Keys: labels, subcluster_centers, subcluster_labels, n_subclusters,
        radii, estimate, n, method.

    Examples
    --------
    >>> r = geron_birch([[0.0], [0.1], [10.0], [10.1]], n_clusters=2, threshold=1.0, branching_factor=2)
    >>> r["n_subclusters"]
    2
    >>> [int(v) for v in r["labels"]]
    [0, 0, 1, 1]
    >>> [round(float(c), 4) for c in np.asarray(r["subcluster_centers"]).ravel()]
    [0.05, 10.05]

    A tight threshold refuses to absorb, so every point becomes its own CF:

    >>> geron_birch([[0.0], [0.1], [10.0]], n_clusters=None, threshold=0.001)["n_subclusters"]
    3

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_birch: X must be 2-D, got ndim={A.ndim}")
    n = A.shape[0]
    if n == 0:
        raise ValueError("geron_birch: X has no rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_birch: X must be finite")
    t = float(threshold)
    if not np.isfinite(t) or t <= 0:
        raise ValueError(f"geron_birch: threshold must be positive, got {t}")
    B = int(branching_factor)
    if B < 2:
        raise ValueError(f"geron_birch: branching_factor must be >= 2, got {B}")
    k = None if n_clusters is None else int(n_clusters)
    if k is not None and k < 1:
        raise ValueError("geron_birch: n_clusters must be >= 1 or None")

    leaves = [[]]
    for i in range(n):
        x = A[i]
        best_leaf = best_entry = None
        best_d = np.inf
        for li, leaf in enumerate(leaves):
            for ei, cf in enumerate(leaf):
                d = float(np.linalg.norm(cf.centroid - x))
                if d < best_d and cf.radius_if_added(x) <= t:
                    best_d, best_leaf, best_entry = d, li, ei
        if best_leaf is not None:
            leaves[best_leaf][best_entry].add(x, i)
        else:
            target = next((leaf for leaf in leaves if len(leaf) < B), None)
            if target is None:
                target = []
                leaves.append(target)
            target.append(_CF(x, i))

    cfs = [cf for leaf in leaves for cf in leaf]
    centers = np.vstack([cf.centroid for cf in cfs])
    radii = np.array([cf.radius for cf in cfs])
    m = len(cfs)

    if k is None or k >= m:
        sub_labels = np.arange(m)
    else:
        groups = {i: [i] for i in range(m)}
        active = list(range(m))
        while len(active) > k:
            best = None
            best_d = np.inf
            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    a, b = active[ii], active[jj]
                    ca = centers[groups[a]].mean(axis=0)
                    cb = centers[groups[b]].mean(axis=0)
                    d = float(np.linalg.norm(ca - cb))
                    if d < best_d:
                        best_d, best = d, (a, b)
            a, b = best
            groups[a] = groups[a] + groups[b]
            del groups[b]
            active.remove(b)
        sub_labels = np.empty(m, dtype=int)
        for new_id, gid in enumerate(sorted(active, key=lambda g: min(groups[g]))):
            sub_labels[groups[gid]] = new_id

    labels = np.empty(n, dtype=int)
    for ci, cf in enumerate(cfs):
        labels[cf.members] = sub_labels[ci]

    return RichResult(
        title="BIRCH",
        summary_lines=[("Sub-clusters", m), ("Clusters", int(sub_labels.max()) + 1), ("Threshold", t)],
        payload={
            "labels": labels,
            "subcluster_centers": centers,
            "subcluster_labels": sub_labels,
            "subcluster_sizes": np.array([cf.n for cf in cfs]),
            "n_subclusters": int(m),
            "radii": radii,
            "n_leaves": len(leaves),
            "estimate": float(m),
            "n": int(n),
            "method": "BIRCH: CF-tree leaf summarisation then centroid-linkage global clustering",
        },
    )


def cheatsheet():
    return "hmbrch: BIRCH: balanced iterative reducing and clustering using hierarchies"
