# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""k-means++ initialization for well-separated initial centers."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_kmeans_plus_plus"]

_METHOD = "k-means++ seeding (D^2 sampling)"


def geron_kmeans_plus_plus(X, n_clusters, seed=0):
    """
    k-means++ initialization for well-separated initial centers.

    Formula: P(x_i) proportional to min_j ||x_i - mu_j||^2

    The first centre is uniform; each later centre is drawn with
    probability proportional to its squared distance to the nearest
    centre already chosen.  That single change is what makes Lloyd's
    algorithm reliable: a point far from every existing centre is
    quadratically more likely to be picked, so the seeds spread out
    instead of clumping, and the expected final inertia is within
    ``O(log k)`` of optimal.

    Sampling is exact (cumulative-sum inversion of the D^2 weights), not
    a "pick the farthest point" shortcut -- the greedy farthest-point
    rule is a different algorithm and is far more sensitive to outliers.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data; a 1-D array is treated as one column.
    n_clusters : int
        Number of centres, ``1 <= n_clusters <= m``.
    seed : int
        Seed for the draws.

    Returns
    -------
    result : RichResult
        Keys: centers, indices, d2, min_pair_distance, estimate, n, method.

    Examples
    --------
    Two far-apart groups: whichever point is picked first, the second
    centre almost surely comes from the other group, so the seeds are at
    least 100 apart:

    >>> X = [[0.0], [1.0], [100.0], [101.0]]
    >>> r = geron_kmeans_plus_plus(X, n_clusters=2, seed=0)
    >>> bool(r["min_pair_distance"] > 90)
    True
    >>> r["centers"].shape
    (2, 1)

    Asking for as many centres as points returns every point exactly
    once:

    >>> a = geron_kmeans_plus_plus([[0.0], [1.0], [2.0]], n_clusters=3, seed=5)
    >>> sorted(int(v) for v in a["indices"])
    [0, 1, 2]

    Duplicate points get zero D^2 weight and are never chosen twice:

    >>> d = geron_kmeans_plus_plus([[0.0], [0.0], [7.0]], n_clusters=2, seed=1)
    >>> sorted(float(v) for v in d["centers"].ravel())
    [0.0, 7.0]

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_kmeans_plus_plus: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_kmeans_plus_plus: X contains non-finite values")
    m = A.shape[0]
    k = int(n_clusters)
    if not (1 <= k <= m):
        raise ValueError(f"geron_kmeans_plus_plus: n_clusters must lie in 1..{m}, got {n_clusters!r}")

    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(m))
    chosen = [first]
    d2 = np.sum((A - A[first]) ** 2, axis=1)

    while len(chosen) < k:
        total = float(np.sum(d2))
        if total <= 0:
            # Every remaining point coincides with a chosen centre; fall
            # back to any index not already used rather than repeating one.
            remaining = [i for i in range(m) if i not in chosen]
            if not remaining:
                raise ValueError(
                    f"geron_kmeans_plus_plus: cannot pick {k} distinct centres from {m} points"
                )
            nxt = int(remaining[0])
        else:
            u = float(rng.random()) * total
            nxt = int(np.searchsorted(np.cumsum(d2), u, side="right"))
            nxt = min(nxt, m - 1)
        chosen.append(nxt)
        d2 = np.minimum(d2, np.sum((A - A[nxt]) ** 2, axis=1))

    idx = np.asarray(chosen, dtype=int)
    centers = A[idx].copy()
    if k > 1:
        diffs = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=2))
        min_pair = float(np.min(dist[np.triu_indices(k, 1)]))
    else:
        min_pair = float("inf")

    return RichResult(
        title="k-means++ seeding",
        summary_lines=[("Centres", k), ("Closest pair of seeds", min_pair)],
        interpretation=(
            "D^2 sampling spreads the seeds; the residual D^2 is the inertia a single "
            "assignment pass would already achieve."
        ),
        payload={
            "centers": centers,
            "indices": idx,
            "d2": d2,
            "min_pair_distance": min_pair,
            "estimate": float(np.sum(d2)),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkmpp: k-means++ D^2 seeding -- P(x) proportional to its squared distance to the nearest centre"
