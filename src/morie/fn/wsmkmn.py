# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""K-means clustering (Lloyd's algorithm)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_kmeans"]


def wasserman_kmeans(X, k, max_iter=300):
    """
    K-means by Lloyd's algorithm with deterministic seeding.

    Formula: min sum_k sum_{x in C_k} |x - mu_k|^2. Initial centres
    are the k points chosen by the farthest-first traversal starting
    from the point nearest the grand mean — fully deterministic, no
    RNG. Iterates assign/update until labels stop changing. An empty
    cluster keeps its previous centre (documented degenerate case).

    Parameters
    ----------
    X : array-like, shape (n, d)
        Data, n >= k.
    k : int
        Clusters, >= 1.
    max_iter : int
        Lloyd iteration cap.

    Returns
    -------
    result : dict
        Keys: estimate (within-cluster sum of squares), centers
        (k x d row-major), labels (0-based), iterations, converged,
        n, d, k, method.

    References
    ----------
    Wasserman (2004), Ch 19 (clustering); Lloyd (1982).

    Examples
    --------
    Two obvious blobs:

    >>> X = [[0.0], [0.2], [-0.2], [10.0], [10.2], [9.8]]
    >>> out = wasserman_kmeans(X, 2)
    >>> sorted(round(c, 10) for c in out["centers"])
    [0.0, 10.0]
    >>> out["labels"][:3] == [out["labels"][0]] * 3
    True
    >>> round(out["estimate"], 10)
    0.16
    >>> wasserman_kmeans(X, 7)
    Traceback (most recent call last):
        ...
    ValueError: k-means needs n >= k; got n=6, k=7.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    if n < k:
        raise ValueError(f"k-means needs n >= k; got n={n}, k={k}.")
    grand = np.mean(X, axis=0)
    first = int(np.argmin(np.sum((X - grand) ** 2, axis=1)))
    centre_idx = [first]
    for _ in range(1, k):
        dmin = np.min(
            [np.sum((X - X[i]) ** 2, axis=1) for i in centre_idx], axis=0)
        centre_idx.append(int(np.argmax(dmin)))
    C = X[centre_idx].copy()
    labels = np.full(n, -1)
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        dist = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        new = np.argmin(dist, axis=1)
        if np.array_equal(new, labels):
            converged = True
            break
        labels = new
        for j in range(k):
            members = X[labels == j]
            if members.size:
                C[j] = np.mean(members, axis=0)
    wcss = float(np.sum((X - C[labels]) ** 2))
    return RichResult(payload={
        "estimate": wcss, "centers": [float(v) for v in C.ravel()],
        "labels": [int(v) for v in labels], "iterations": int(it),
        "converged": bool(converged), "n": int(n), "d": int(d), "k": k,
        "method": "Lloyd k-means, farthest-first deterministic seeding"})


def cheatsheet():
    return "wsmkmn: farthest-first seed (no RNG), Lloyd until labels fixed"
