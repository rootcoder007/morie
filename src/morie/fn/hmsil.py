# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Silhouette score for cluster quality."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_silhouette"]


def geron_silhouette(X, labels, metric="euclidean"):
    """
    Silhouette score for cluster quality.

    Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))

    ``a(i)`` is the mean distance from i to the other members of its own
    cluster (i itself excluded -- including it would shrink a(i) towards
    zero and inflate the score), and ``b(i)`` is the smallest mean
    distance to any *other* cluster. A singleton cluster has no
    within-cluster distances at all, so its silhouette is defined to be 0
    by convention rather than left undefined.

    Parameters
    ----------
    X : array-like
        Data (n, d), n >= 2.
    labels : array-like
        Cluster assignment per row; at least 2 distinct clusters and at
        most n - 1.
    metric : {"euclidean", "manhattan"}, default "euclidean"
        Distance used for a and b.

    Returns
    -------
    result : RichResult
        Keys: silhouette, samples, a, b, cluster_means, estimate, n, method.

    Examples
    --------
    Two tight, far-apart pairs: a = 0.1, b = (10 + 10.1)/2 = 10.05, so
    s = 9.95/10.05 for the outermost point.

    >>> r = geron_silhouette([[0.0], [0.1], [10.0], [10.1]], [0, 0, 1, 1])
    >>> round(float(r["samples"][0]), 6)
    0.99005
    >>> bool(r["silhouette"] > 0.98)
    True

    Interleaved clusters score near zero or below:

    >>> r2 = geron_silhouette([[0.0], [1.0], [0.1], [1.1]], [0, 0, 1, 1])
    >>> bool(r2["silhouette"] < 0.1)
    True

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.shape[0] < 2:
        raise ValueError("geron_silhouette: X must be a 2-D array with at least 2 rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_silhouette: X contains non-finite values")
    lab = np.asarray(labels).ravel()
    if lab.size != A.shape[0]:
        raise ValueError(f"geron_silhouette: X has {A.shape[0]} rows but labels has {lab.size} entries")
    uniq = np.unique(lab)
    if uniq.size < 2:
        raise ValueError("geron_silhouette: the silhouette is undefined with fewer than 2 clusters")
    if uniq.size >= A.shape[0]:
        raise ValueError(
            f"geron_silhouette: {uniq.size} clusters for {A.shape[0]} points leaves no within-cluster distances"
        )
    m = str(metric).lower()
    if m not in ("euclidean", "manhattan"):
        raise ValueError(f"geron_silhouette: metric must be 'euclidean' or 'manhattan', got {metric!r}")

    diff = A[:, None, :] - A[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2)) if m == "euclidean" else np.sum(np.abs(diff), axis=2)

    n = A.shape[0]
    a = np.zeros(n)
    b = np.zeros(n)
    s = np.zeros(n)
    for i in range(n):
        own = lab == lab[i]
        own_others = own.copy()
        own_others[i] = False
        if not own_others.any():
            a[i] = 0.0
            b[i] = np.min([D[i, lab == c].mean() for c in uniq if c != lab[i]])
            s[i] = 0.0  # singleton cluster: undefined cohesion, scored 0 by convention
            continue
        a[i] = float(D[i, own_others].mean())
        b[i] = float(np.min([D[i, lab == c].mean() for c in uniq if c != lab[i]]))
        denom = max(a[i], b[i])
        s[i] = float((b[i] - a[i]) / denom) if denom > 0 else 0.0

    means = {int(c) if np.issubdtype(uniq.dtype, np.integer) else c: float(s[lab == c].mean()) for c in uniq}

    return RichResult(
        title="Silhouette score",
        summary_lines=[
            ("Points", n),
            ("Clusters", int(uniq.size)),
            ("Mean silhouette", float(np.mean(s))),
            ("Worst point", float(np.min(s))),
        ],
        interpretation=(
            "s near +1 means the point sits well inside its cluster, near 0 means it is on a boundary, "
            "negative means it would be happier in the neighbouring cluster."
        ),
        payload={
            "silhouette": float(np.mean(s)),
            "samples": s,
            "a": a,
            "b": b,
            "cluster_means": means,
            "n_clusters": int(uniq.size),
            "estimate": float(np.mean(s)),
            "n": int(n),
            "method": f"Silhouette coefficient with {m} distances (self excluded from cohesion)",
        },
    )


def cheatsheet():
    return "hmsil: Silhouette score for cluster quality"
