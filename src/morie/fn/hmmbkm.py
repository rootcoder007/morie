# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mini-batch k-means: update centers using small random batches."""

import numpy as np

from ._richresult import RichResult
from .hmkmpp import geron_kmeans_plus_plus

__all__ = ["geron_minibatch_kmeans"]

_METHOD = "Mini-batch k-means"


def geron_minibatch_kmeans(X, n_clusters, batch_size, seed=0, n_iter=100):
    """
    Mini-batch k-means: update centers using small random batches.

    Formula: center_update = (1-lr)*mu_c + lr*mean(batch_in_c)

    Sculley's variant.  Each centre keeps its own per-centre counter and
    moves with learning rate ``1/count``, which makes the update an
    exact running mean of every point ever assigned to that centre --
    that is the reason for the per-centre counter rather than a global
    step size, and it is what makes the centres converge rather than
    jitter forever.

    Seeding is delegated to
    :func:`morie.fn.hmkmpp.geron_kmeans_plus_plus`.  The result is an
    approximation to full k-means: the final inertia over the whole
    dataset is returned so the cost of the approximation is visible.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    n_clusters : int
        Number of clusters.
    batch_size : int
        Points sampled per iteration, ``1 <= batch_size <= m``.
    seed : int
        Seed for seeding and batch draws.
    n_iter : int
        Number of mini-batch iterations.

    Returns
    -------
    result : RichResult
        Keys: labels, centers, inertia, counts, n_iter, estimate, n, method.

    Examples
    --------
    Two well-separated groups are still recovered from batches of two:

    >>> X = [[0.0], [0.5], [10.0], [10.5]]
    >>> r = geron_minibatch_kmeans(X, n_clusters=2, batch_size=2, seed=0, n_iter=200)
    >>> lo, hi = sorted(float(c) for c in r["centers"].ravel())
    >>> bool(abs(lo - 0.25) < 0.05 and abs(hi - 10.25) < 0.05)
    True
    >>> int(r["labels"][0] == r["labels"][1]), int(r["labels"][1] == r["labels"][2])
    (1, 0)

    Its inertia is at least that of the exact algorithm on the same
    data -- mini-batch buys speed, not accuracy:

    >>> from morie.fn.hmkmn import geron_kmeans
    >>> exact = geron_kmeans(X, n_clusters=2, seed=0)["inertia"]
    >>> bool(r["inertia"] >= exact - 1e-9)
    True

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_minibatch_kmeans: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_minibatch_kmeans: X contains non-finite values")
    m = A.shape[0]
    k = int(n_clusters)
    if not (1 <= k <= m):
        raise ValueError(f"geron_minibatch_kmeans: n_clusters must lie in 1..{m}, got {n_clusters!r}")
    bs = int(batch_size)
    if not (1 <= bs <= m):
        raise ValueError(f"geron_minibatch_kmeans: batch_size must lie in 1..{m}, got {batch_size!r}")
    iters = int(n_iter)
    if iters < 1:
        raise ValueError(f"geron_minibatch_kmeans: n_iter must be at least 1, got {n_iter!r}")

    centers = geron_kmeans_plus_plus(A, k, seed=int(seed))["centers"].copy()
    counts = np.zeros(k, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    for _ in range(iters):
        batch = rng.choice(m, size=bs, replace=False)
        Xb = A[batch]
        d2 = np.sum((Xb[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assign = np.argmin(d2, axis=1)
        for i in range(bs):
            c = int(assign[i])
            counts[c] += 1
            lr = 1.0 / counts[c]
            centers[c] = (1.0 - lr) * centers[c] + lr * Xb[i]

    d2_all = np.sum((A[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2_all, axis=1)
    inertia = float(np.sum(np.min(d2_all, axis=1)))

    return RichResult(
        title="Mini-batch k-means",
        summary_lines=[("Clusters", k), ("Batch size", bs), ("Inertia (full data)", inertia)],
        interpretation=(
            "The per-centre 1/count learning rate makes each centre the running mean of everything "
            "assigned to it, so it converges rather than oscillating."
        ),
        payload={
            "labels": labels,
            "centers": centers,
            "inertia": inertia,
            "counts": counts,
            "n_iter": iters,
            "estimate": inertia,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmbkm: mini-batch k-means with per-centre 1/count learning rate (seeded by hmkmpp)"
