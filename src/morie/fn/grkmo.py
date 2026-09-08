# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""k-means within-cluster sum of squares (inertia)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_kmeans_objective"]

_METHOD = "k-means inertia (within-cluster sum of squares)"


def geron_kmeans_objective(X, centroids, labels):
    r"""The objective k-means actually minimises.

    .. math::
        J = \sum_{i=1}^{m} \bigl\| x^{(i)} - \mu_{c(i)} \bigr\|^2

    Squared distances, unsquared would be k-medians.  The consequence
    is worth stating: because the sum of squared deviations is minimised
    by the *mean*, the assignment step and the mean-update step are both
    descending the same ``J``, which is why the algorithm converges.

    ``J`` falls monotonically as ``k`` grows and hits 0 at ``k = m``, so
    it can never be used to choose ``k`` on its own -- that is what the
    elbow plot and the silhouette score are for.

    Parameters
    ----------
    X : array-like, shape (m, n)
    centroids : array-like, shape (k, n)
    labels : array-like of int, shape (m,)
        Cluster index per instance, in ``0 .. k-1``.

    Returns
    -------
    RichResult
        Payload keys ``inertia``, ``per_cluster_inertia``,
        ``cluster_sizes``, ``distances``,
        ``centroids_are_means`` (True when each centroid equals its
        cluster's mean, i.e. the assignment is at a local optimum),
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 8, K-means section.

    Examples
    --------
    Two points, one centroid halfway between them: each is 1 away, so
    ``J = 2``:

    >>> r = geron_kmeans_objective([[0.0], [2.0]], [[1.0]], [0, 0])
    >>> r["inertia"]
    2.0
    >>> r["centroids_are_means"]
    True

    Put the centroid on top of each point instead and the objective
    vanishes:

    >>> geron_kmeans_objective([[0.0], [2.0]], [[0.0], [2.0]], [0, 1])["inertia"]
    0.0
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    C = np.atleast_2d(np.asarray(centroids, dtype=float))
    lab = np.asarray(labels).ravel().astype(int)
    if A.ndim != 2 or C.ndim != 2:
        raise ValueError(f"X and centroids must both be 2-D, got {A.shape} and {C.shape}.")
    if A.shape[1] != C.shape[1]:
        raise ValueError(f"X has {A.shape[1]} features but centroids have {C.shape[1]}.")
    m, k = A.shape[0], C.shape[0]
    if lab.size != m:
        raise ValueError(f"labels has {lab.size} entries but X has {m} rows.")
    if lab.min() < 0 or lab.max() >= k:
        raise ValueError(f"labels must lie in [0, {k - 1}], got range [{lab.min()}, {lab.max()}].")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(C)):
        raise ValueError("X and centroids must be finite.")

    d2 = np.sum((A - C[lab]) ** 2, axis=1)
    per = np.array([d2[lab == j].sum() for j in range(k)])
    sizes = np.array([int((lab == j).sum()) for j in range(k)])

    is_mean = True
    for j in range(k):
        if sizes[j] and not np.allclose(A[lab == j].mean(axis=0), C[j], atol=1e-10):
            is_mean = False
            break

    return RichResult(
        title="k-means objective",
        summary_lines=[("Inertia", float(d2.sum())), ("k", int(k)),
                       ("Sizes", sizes.tolist())],
        payload={
            "inertia": float(d2.sum()),
            "per_cluster_inertia": per.tolist(),
            "cluster_sizes": sizes.tolist(),
            "distances": np.sqrt(d2).tolist(),
            "centroids_are_means": bool(is_mean),
            "empty_clusters": np.flatnonzero(sizes == 0).tolist(),
            "estimate": float(d2.sum()),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grkmo: J = sum ||x - mu_c(x)||^2; monotone in k, so useless for choosing k"
