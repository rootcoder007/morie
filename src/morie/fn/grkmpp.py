# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""k-means++ seeding: sample centroids proportionally to D(x)^2."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_kmeans_pp_seeding"]

_METHOD = "k-means++ seeding"


def _lcg(seed):
    """Reference LCG: s = (1664525 s + 1013904223) mod 2**32, u = (s+0.5)/2**32."""
    s = int(seed) % 2**32
    while True:
        s = (1664525 * s + 1013904223) % 2**32
        yield (s + 0.5) / 2**32


def geron_kmeans_pp_seeding(X, k, seed=0):
    r"""Choose ``k`` starting centroids, spread out on purpose.

    .. math::
        P(x \mid C) \propto \min_{c \in C} \|x - c\|^2

    The squared distance is what makes this work: a point twice as far
    from the current centroids is *four* times as likely to be picked,
    so seeds land in distinct clusters instead of huddling in the
    densest region.  Plain uniform seeding is why naive k-means needs a
    dozen restarts.

    Draws come from the deterministic LCG above, so a given ``seed``
    reproduces the same seeding everywhere.

    Parameters
    ----------
    X : array-like, shape (m, n)
    k : int
        Number of centroids, ``1 <= k <= m``.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``centroids``, ``indices``, ``min_pairwise_distance``,
        ``sampling_probabilities`` (the D^2 distribution used at each
        step after the first), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 8, K-means++ section.  Score a seeding with
    :func:`morie.fn.grkmo.geron_kmeans_objective`.

    Examples
    --------
    Four points in two far-apart pairs.  Whatever the first pick,
    ``D^2`` sampling puts the second centroid in the other pair -- the
    two seeds are 100 apart, not 1:

    >>> X = [[0.0], [1.0], [100.0], [101.0]]
    >>> r = geron_kmeans_pp_seeding(X, k=2, seed=0)
    >>> len(r["indices"])
    2
    >>> r["min_pairwise_distance"] > 50
    True

    Duplicate points cannot both be chosen -- once one is a centroid the
    other has ``D^2 = 0``:

    >>> r2 = geron_kmeans_pp_seeding([[0.0], [0.0], [5.0]], k=2, seed=3)
    >>> sorted(round(c[0], 6) for c in r2["centroids"])
    [0.0, 5.0]
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty 2-D array, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X must be finite.")
    m = A.shape[0]
    k = int(k)
    if not (1 <= k <= m):
        raise ValueError(f"k must lie in [1, {m}] (one centroid per distinct instance at most), got {k}.")

    u = _lcg(seed)
    first = min(int(next(u) * m), m - 1)
    idx = [first]
    d2 = np.sum((A - A[first]) ** 2, axis=1)
    probs = []

    for _ in range(1, k):
        total = float(d2.sum())
        if total <= 0:
            remaining = [i for i in range(m) if i not in idx]
            if not remaining:
                raise ValueError(
                    f"k={k} centroids requested but X has only "
                    f"{len(idx)} distinct points."
                )
            pick = remaining[0]
            probs.append(None)
        else:
            p = d2 / total
            probs.append(p.tolist())
            target = next(u) * total
            pick = int(np.searchsorted(np.cumsum(d2), target))
            pick = min(pick, m - 1)
        idx.append(pick)
        d2 = np.minimum(d2, np.sum((A - A[pick]) ** 2, axis=1))

    C = A[idx]
    if k > 1:
        i, j = np.triu_indices(k, k=1)
        mind = float(np.min(np.linalg.norm(C[i] - C[j], axis=1)))
    else:
        mind = float("inf")

    return RichResult(
        title="k-means++ seeding",
        summary_lines=[("k", k), ("Indices", idx),
                       ("Min centroid separation", mind)],
        payload={
            "centroids": C.tolist(),
            "indices": [int(i) for i in idx],
            "min_pairwise_distance": mind,
            "sampling_probabilities": probs,
            "seed": int(seed),
            "estimate": C.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grkmpp: seed t+1 sampled with P(x) propto min_c ||x-c||^2, deterministic LCG"
