# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""k-means clustering via Lloyd's algorithm."""

from . import _array_core as np

from ._richresult import RichResult
from .hmkmpp import geron_kmeans_plus_plus

__all__ = ["geron_kmeans"]

_METHOD = "k-means (Lloyd's algorithm, k-means++ seeded)"


def geron_kmeans(X, n_clusters, seed=0, max_iter=300, tol=1e-10, n_init=10):
    """
    k-means clustering via Lloyd's algorithm.

    Formula: min sum_i ||x_i - mu_{c_i}||^2

    Lloyd's algorithm alternates assignment (each point to its nearest
    centre) and update (each centre to its cluster's mean).  Both steps
    can only lower the inertia and there are finitely many assignments,
    so it always terminates -- at a *local* minimum, which is why
    ``n_init`` restarts are run and the best inertia kept.

    Seeding is delegated to
    :func:`morie.fn.hmkmpp.geron_kmeans_plus_plus`.  An empty cluster is
    re-seeded to the point currently furthest from its centre rather
    than dropped, so ``n_clusters`` centres always come back.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    n_clusters : int
        Number of clusters, ``1 <= n_clusters <= m``.
    seed : int
        Base seed; restart ``i`` uses ``seed + i``.
    max_iter : int
        Iteration cap per restart.
    tol : float
        Stop when the centres move less than this in total.
    n_init : int
        Number of restarts.

    Returns
    -------
    result : RichResult
        Keys: labels, centers, inertia, n_iter, distances, counts,
        estimate, n, method.

    Examples
    --------
    Two obvious clusters are recovered exactly, and the inertia is the
    within-group sum of squares ``4 * 0.25 = 1``:

    >>> X = [[0.0], [1.0], [10.0], [11.0]]
    >>> r = geron_kmeans(X, n_clusters=2, seed=0)
    >>> sorted(float(c) for c in r["centers"].ravel())
    [0.5, 10.5]
    >>> round(r["inertia"], 10)
    1.0
    >>> int(r["labels"][0] == r["labels"][1]), int(r["labels"][0] == r["labels"][2])
    (1, 0)

    One cluster puts the centre at the global mean and the inertia at
    the total sum of squares:

    >>> one = geron_kmeans([[0.0], [2.0], [4.0]], n_clusters=1)
    >>> float(one["centers"][0, 0]), round(one["inertia"], 10)
    (2.0, 8.0)

    Every point its own cluster gives zero inertia:

    >>> round(geron_kmeans([[0.0], [1.0], [2.0]], n_clusters=3)["inertia"], 12)
    0.0

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_kmeans: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_kmeans: X contains non-finite values")
    m = A.shape[0]
    k = int(n_clusters)
    if not (1 <= k <= m):
        raise ValueError(f"geron_kmeans: n_clusters must lie in 1..{m}, got {n_clusters!r}")
    if int(max_iter) < 1:
        raise ValueError(f"geron_kmeans: max_iter must be at least 1, got {max_iter!r}")
    if float(tol) < 0:
        raise ValueError(f"geron_kmeans: tol must be non-negative, got {tol!r}")
    if int(n_init) < 1:
        raise ValueError(f"geron_kmeans: n_init must be at least 1, got {n_init!r}")

    best = None
    for run in range(int(n_init)):
        centers = geron_kmeans_plus_plus(A, k, seed=int(seed) + run)["centers"].copy()
        labels = np.zeros(m, dtype=int)
        it = 0
        for it in range(1, int(max_iter) + 1):
            d2 = np.sum((A[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            new = centers.copy()
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new[j] = A[mask].mean(axis=0)
                else:
                    # Re-seed an empty cluster at the worst-served point.
                    worst = int(np.argmax(np.min(d2, axis=1)))
                    new[j] = A[worst]
            shift = float(np.sum(np.abs(new - centers)))
            centers = new
            if shift <= float(tol):
                break
        d2 = np.sum((A[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        inertia = float(np.sum(np.min(d2, axis=1)))
        if best is None or inertia < best[0]:
            best = (inertia, centers, labels, it, np.sqrt(d2))

    inertia, centers, labels, n_iter, dist = best
    counts = np.bincount(labels, minlength=k)

    return RichResult(
        title="k-means",
        summary_lines=[
            ("Clusters", k),
            ("Inertia", inertia),
            ("Iterations (best run)", n_iter),
            ("Smallest cluster", int(counts.min())),
        ],
        interpretation=(
            "Inertia falls monotonically with k, so it cannot be used to choose k on its own -- "
            "use the elbow or a silhouette score."
        ),
        payload={
            "labels": labels,
            "centers": centers,
            "inertia": inertia,
            "n_iter": int(n_iter),
            "distances": dist,
            "counts": counts,
            "estimate": inertia,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkmn: k-means by Lloyd's algorithm with k-means++ restarts (delegates seeding to hmkmpp)"
