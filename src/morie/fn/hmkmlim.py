# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Limits of k-means: fails with non-spherical or varying-size clusters."""

from . import _array_core as np

from ._richresult import RichResult
from .hmkmn import geron_kmeans

__all__ = ["geron_kmeans_limits"]

_METHOD = "k-means assumption diagnostics"


def geron_kmeans_limits(X, n_clusters=2, seed=0):
    """
    Limits of k-means: fails with non-spherical or varying-size clusters.

    Formula: assumes isotropic clusters; fails for anisotropic

    A diagnostic, not a clusterer.  k-means assigns by *Euclidean*
    distance to the centre, which is the maximum-likelihood rule only
    when every cluster is an isotropic Gaussian of equal spread.  Two
    measurable consequences of violating that are computed here on the
    k-means solution (delegated to :func:`morie.fn.hmkmn.geron_kmeans`):

    ``anisotropy`` -- per cluster, the ratio of the largest to the
    smallest eigenvalue of its covariance.  1.0 is a perfect sphere;
    large values mean an elongated cluster whose far end is closer to a
    neighbouring centre than to its own.

    ``reassigned_fraction`` -- the fraction of points that change
    cluster when the Euclidean rule is replaced by a Mahalanobis rule
    using each cluster's own covariance.  Anything above zero is direct
    evidence that the isotropy assumption changed the answer.

    ``size_ratio`` -- largest cluster over smallest, since k-means also
    pulls boundaries toward the larger cluster.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    n_clusters : int
        Number of clusters to fit.
    seed : int
        Seed for k-means.

    Returns
    -------
    result : RichResult
        Keys: anisotropy, max_anisotropy, reassigned_fraction,
        size_ratio, labels, mahalanobis_labels, estimate, n, method.

    Examples
    --------
    Two compact, well-separated, equally-sized blobs: nothing is
    reassigned and the sizes match.

    >>> X = [[0.0, 0.0], [0.2, 0.1], [0.1, -0.1], [5.0, 0.0], [5.2, 0.1], [5.1, -0.1]]
    >>> r = geron_kmeans_limits(X, n_clusters=2, seed=0)
    >>> float(r["reassigned_fraction"]), float(r["size_ratio"])
    (0.0, 1.0)

    An elongated cluster registers as strongly anisotropic:

    >>> Y = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
    ...      [0.0, 9.0], [1.0, 9.0], [2.0, 9.0], [3.0, 9.0]]
    >>> a = geron_kmeans_limits(Y, n_clusters=2, seed=0)
    >>> bool(a["max_anisotropy"] > 100)
    True

    Unequal cluster sizes are reported rather than hidden:

    >>> Z = [[0.0], [0.1], [0.2], [0.3], [10.0]]
    >>> float(geron_kmeans_limits(Z, n_clusters=2, seed=0)["size_ratio"])
    4.0

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_kmeans_limits: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_kmeans_limits: X contains non-finite values")
    m, d = A.shape
    k = int(n_clusters)
    if not (2 <= k <= m):
        raise ValueError(
            f"geron_kmeans_limits: n_clusters must lie in 2..{m}; the diagnostic compares clusters, "
            f"so one cluster has nothing to compare (got {n_clusters!r})"
        )

    km = geron_kmeans(A, n_clusters=k, seed=int(seed))
    labels = km["labels"]
    centers = km["centers"]
    counts = np.bincount(labels, minlength=k)
    if np.any(counts == 0):
        raise ValueError("geron_kmeans_limits: k-means returned an empty cluster; lower n_clusters")

    aniso = np.empty(k)
    covs = []
    for j in range(k):
        pts = A[labels == j]
        if pts.shape[0] < 2:
            cov = np.eye(d)
            aniso[j] = 1.0
        else:
            cov = np.cov(pts, rowvar=False)
            cov = np.atleast_2d(cov)
            ev = np.linalg.eigvalsh(cov)
            ev = np.clip(ev, 0.0, None)
            lo = float(ev.min())
            hi = float(ev.max())
            aniso[j] = float("inf") if (lo == 0 and hi > 0) else (hi / lo if lo > 0 else 1.0)
        covs.append(cov)

    # Mahalanobis reassignment with a ridge so a degenerate cluster
    # covariance does not make the inverse blow up.
    ridge = 1e-9 * float(np.trace(np.cov(A, rowvar=False).reshape(d, d))) + 1e-12
    md = np.empty((m, k))
    for j in range(k):
        cov = covs[j] + ridge * np.eye(d)
        diff = A - centers[j]
        md[:, j] = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)
    maha_labels = np.argmin(md, axis=1)
    reassigned = float(np.mean(maha_labels != labels))

    finite_aniso = aniso[np.isfinite(aniso)]
    max_aniso = float(np.max(aniso)) if aniso.size else 1.0

    warns = []
    if reassigned > 0:
        warns.append(
            f"{reassigned:.0%} of points change cluster under a Mahalanobis rule: "
            f"the isotropy assumption is changing the answer."
        )
    if max_aniso > 10:
        warns.append(f"largest cluster anisotropy is {max_aniso:.3g}; the clusters are far from spherical.")

    return RichResult(
        title="k-means assumption check",
        summary_lines=[
            ("Clusters", k),
            ("Worst anisotropy", max_aniso),
            ("Reassigned under Mahalanobis", reassigned),
            ("Size ratio", float(counts.max() / counts.min())),
        ],
        warnings=warns,
        interpretation=(
            "Euclidean assignment is the right rule only for equal-size isotropic clusters; "
            "a Gaussian mixture handles the anisotropic case."
        ),
        payload={
            "anisotropy": aniso,
            "max_anisotropy": max_aniso,
            "mean_anisotropy": float(np.mean(finite_aniso)) if finite_aniso.size else float("inf"),
            "reassigned_fraction": reassigned,
            "size_ratio": float(counts.max() / counts.min()),
            "labels": labels,
            "mahalanobis_labels": maha_labels,
            "counts": counts,
            "estimate": reassigned,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkmlim: k-means assumption diagnostics -- anisotropy, size ratio, Mahalanobis reassignment"
