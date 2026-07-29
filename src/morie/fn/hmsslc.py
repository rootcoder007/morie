# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Semi-supervised learning via k-means representative labeling."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_semisupervised_cluster"]


def _lloyd(Z, k, seed=0, iters=100):
    """Lloyd's algorithm with k-means++ seeding on a deterministic LCG stream.

    Local by design: the standalone k-means lives in ``morie.fn.hmkmn``;
    these are the few lines the label-propagation recipe needs.
    """
    n = Z.shape[0]
    s = int(seed) % 2**32
    centers = [Z[0]]
    for _ in range(1, k):
        d2 = np.min(np.asarray([np.sum((Z - c) ** 2, axis=1) for c in centers]), axis=0)
        tot = float(np.sum(d2))
        s = (1664525 * s + 1013904223) % 2**32
        u = (s + 0.5) / 2**32 * (tot if tot > 0 else 1.0)
        idx = int(np.searchsorted(np.cumsum(d2), u)) if tot > 0 else len(centers) % n
        centers.append(Z[min(idx, n - 1)])
    C = np.vstack(centers)
    lab = np.zeros(n, dtype=int)
    for it in range(iters):
        d = np.asarray([np.sum((Z - c) ** 2, axis=1) for c in C])
        new = np.argmin(d, axis=0)
        if it > 0 and np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            if np.any(lab == j):
                C[j] = Z[lab == j].mean(axis=0)
    return lab, C


def geron_semisupervised_cluster(X, X_labeled, y_labeled, n_clusters=2, seed=0, y_true=None):
    """
    Semi-supervised learning via k-means representative labeling.

    Formula: label cluster representatives; propagate to members

    Géron's label-propagation recipe, executed: cluster the unlabeled
    pool, find each cluster's **representative instance** -- the real
    data point closest to the centroid, never the centroid itself, since
    a centroid is usually not a valid instance -- take that
    representative's label from the small labeled set (nearest labeled
    neighbour), and propagate it to every member of the cluster. The
    labeling effort is then `n_clusters` decisions instead of n.

    Parameters
    ----------
    X : array-like
        Unlabeled pool (n, d), n >= n_clusters.
    X_labeled : array-like
        Labeled instances (m, d), same width as X.
    y_labeled : array-like
        Their labels, length m.
    n_clusters : int, default 2
        Clusters / labeling budget (>= 1).
    seed : int, default 0
        LCG seed for k-means++ seeding.
    y_true : array-like, optional
        Gold labels for the pool, to score the propagation.

    Returns
    -------
    result : RichResult
        Keys: labels, cluster, representatives, representative_labels,
        accuracy, estimate, n, method.

    Examples
    --------
    Two tight groups, one labeled example near each: every member of a
    group inherits the right label from its representative.

    >>> X = [[0.0], [0.1], [0.2], [9.8], [9.9], [10.0]]
    >>> r = geron_semisupervised_cluster(X, [[0.05], [9.95]], [0, 1], n_clusters=2)
    >>> [int(v) for v in r["labels"]]
    [0, 0, 0, 1, 1, 1]
    >>> sorted(int(v) for v in r["representative_labels"])
    [0, 1]
    >>> float(geron_semisupervised_cluster(X, [[0.05], [9.95]], [0, 1], y_true=[0, 0, 0, 1, 1, 1])["accuracy"])
    1.0

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_semisupervised_cluster: X must be a non-empty (n, d) matrix")
    B = np.asarray(X_labeled, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.size == 0:
        raise ValueError("geron_semisupervised_cluster: X_labeled must be a non-empty (m, d) matrix")
    if B.shape[1] != A.shape[1]:
        raise ValueError(
            f"geron_semisupervised_cluster: X has {A.shape[1]} features but X_labeled has {B.shape[1]}"
        )
    yl = np.asarray(y_labeled).ravel()
    if yl.size != B.shape[0]:
        raise ValueError(
            f"geron_semisupervised_cluster: X_labeled has {B.shape[0]} rows but y_labeled has {yl.size} labels"
        )
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        raise ValueError("geron_semisupervised_cluster: X and X_labeled must be finite")
    k = int(n_clusters)
    if k < 1:
        raise ValueError(f"geron_semisupervised_cluster: n_clusters must be >= 1, got {k}")
    if k > A.shape[0]:
        raise ValueError(
            f"geron_semisupervised_cluster: asked for {k} clusters from {A.shape[0]} unlabeled points"
        )

    cluster, C = _lloyd(A, k, seed=seed)
    reps = np.empty(k, dtype=int)
    rep_lab = np.empty(k, dtype=yl.dtype)
    for j in range(k):
        members = np.flatnonzero(cluster == j)
        if members.size == 0:
            raise ValueError(
                f"geron_semisupervised_cluster: cluster {j} came out empty; reduce n_clusters or change the seed"
            )
        d = np.sum((A[members] - C[j]) ** 2, axis=1)
        reps[j] = int(members[int(np.argmin(d))])
        dl = np.sum((B - A[reps[j]]) ** 2, axis=1)
        rep_lab[j] = yl[int(np.argmin(dl))]

    labels = rep_lab[cluster]

    acc = None
    if y_true is not None:
        g = np.asarray(y_true).ravel()
        if g.size != A.shape[0]:
            raise ValueError(f"geron_semisupervised_cluster: {A.shape[0]} rows but {g.size} gold labels")
        acc = float(np.mean(labels == g))

    return RichResult(
        title="Semi-supervised label propagation",
        summary_lines=[
            ("Unlabeled pool", int(A.shape[0])),
            ("Labeled instances", int(B.shape[0])),
            ("Clusters / labeling budget", k),
            ("Propagated accuracy", acc if acc is not None else "n/a (no gold labels)"),
        ],
        interpretation=(
            "Labeling k representatives buys labels for the whole pool; the accuracy of the result is "
            "capped by how well the clusters line up with the classes."
        ),
        payload={
            "labels": labels,
            "cluster": cluster,
            "centers": C,
            "representatives": reps,
            "representative_labels": rep_lab,
            "accuracy": acc,
            "estimate": float(acc) if acc is not None else float(k / A.shape[0]),
            "n": int(A.shape[0]),
            "method": "k-means clustering, nearest-labeled representative labeling, propagation to members",
        },
    )


def cheatsheet():
    return "hmsslc: Semi-supervised learning via k-means representative labeling"
