# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spectral clustering: eigenvectors of graph Laplacian."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_spectral_clustering"]


def _lloyd(Z, k, seed=0, iters=100):
    """Lloyd's algorithm on the spectral embedding.

    Deliberately local: the standalone k-means lives in ``morie.fn.hmkmn``;
    this is the few lines spectral clustering needs to finish the job.
    """
    n = Z.shape[0]
    s = int(seed) % 2**32
    centers = [Z[0]]
    for _ in range(1, k):  # k-means++ seeding on a deterministic LCG stream
        d2 = np.min(np.asarray([np.sum((Z - c) ** 2, axis=1) for c in centers]), axis=0)
        tot = float(np.sum(d2))
        s = (1664525 * s + 1013904223) % 2**32
        u = (s + 0.5) / 2**32 * (tot if tot > 0 else 1.0)
        idx = int(np.searchsorted(np.cumsum(d2), u)) if tot > 0 else len(centers) % n
        centers.append(Z[min(idx, n - 1)])
    C = np.vstack(centers)
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.asarray([np.sum((Z - c) ** 2, axis=1) for c in C])
        new = np.argmin(d, axis=0)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            if np.any(lab == j):
                C[j] = Z[lab == j].mean(axis=0)
    return lab, C


def geron_spectral_clustering(X, n_clusters=2, affinity="rbf", gamma=1.0, n_neighbors=3, seed=0):
    """
    Spectral clustering: eigenvectors of graph Laplacian.

    Formula: embed via eigenvectors of L = D - W; k-means in embedding

    The affinity graph is built (RBF or symmetric k-nearest-neighbour),
    the unnormalised Laplacian ``L = D - W`` is formed, and the
    eigenvectors of its `n_clusters` smallest eigenvalues become the
    embedding that k-means then partitions. The eigenvalue spectrum is
    returned because it is diagnostic, not decoration: the multiplicity
    of the zero eigenvalue equals the number of connected components of
    the graph, so asking for more clusters than components is meaningful
    while asking for fewer is not.

    Parameters
    ----------
    X : array-like
        Data (n, d), n >= n_clusters.
    n_clusters : int, default 2
        Clusters (>= 2).
    affinity : {"rbf", "knn"}, default "rbf"
        Similarity graph.
    gamma : float, default 1.0
        RBF width parameter (> 0).
    n_neighbors : int, default 3
        Neighbours for the knn graph.
    seed : int, default 0
        LCG seed for k-means++ seeding.

    Returns
    -------
    result : RichResult
        Keys: labels, embedding, eigenvalues, affinity_matrix,
        n_components, estimate, n, method.

    Examples
    --------
    Two well-separated groups on the line: the RBF graph splits into two
    components, so the Laplacian has two (numerically) zero eigenvalues
    and the clustering recovers the groups.

    >>> X = [[0.0], [0.2], [10.0], [10.2]]
    >>> r = geron_spectral_clustering(X, 2)
    >>> int(r["n_components"])
    2
    >>> bool(r["labels"][0] == r["labels"][1] and r["labels"][2] == r["labels"][3])
    True
    >>> bool(r["labels"][0] != r["labels"][2])
    True
    >>> bool(abs(float(r["eigenvalues"][0])) < 1e-12)
    True

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.shape[0] < 2:
        raise ValueError("geron_spectral_clustering: X must be 2-D with at least 2 rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_spectral_clustering: X contains non-finite values")
    k = int(n_clusters)
    n = A.shape[0]
    if k < 2:
        raise ValueError(f"geron_spectral_clustering: n_clusters must be >= 2, got {k}")
    if k > n:
        raise ValueError(f"geron_spectral_clustering: asked for {k} clusters from {n} points")
    aff = str(affinity).lower()
    if aff not in ("rbf", "knn"):
        raise ValueError(f"geron_spectral_clustering: affinity must be 'rbf' or 'knn', got {affinity!r}")
    g = float(gamma)
    if not np.isfinite(g) or g <= 0:
        raise ValueError(f"geron_spectral_clustering: gamma must be positive and finite, got {g}")

    diff = A[:, None, :] - A[None, :, :]
    D2 = np.sum(diff * diff, axis=2)
    if aff == "rbf":
        W = np.exp(-g * D2)
    else:
        nb = int(n_neighbors)
        if not (1 <= nb < n):
            raise ValueError(f"geron_spectral_clustering: n_neighbors must lie in 1..{n - 1}, got {nb}")
        W = np.zeros((n, n))
        for i in range(n):
            order = np.argsort(D2[i], kind="mergesort")[1 : nb + 1]
            W[i, order] = 1.0
        W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)

    L = np.diag(W.sum(axis=1)) - W
    vals, vecs = np.linalg.eigh(L)
    tol = 1e-8 * max(1.0, float(np.max(np.abs(L))))
    n_comp = int(np.sum(vals < tol))
    U = vecs[:, :k]
    labels, centers = _lloyd(U, k, seed=seed)

    return RichResult(
        title="Spectral clustering",
        summary_lines=[
            ("Points", n),
            ("Clusters", k),
            ("Graph components", n_comp),
            ("Eigengap", float(vals[k] - vals[k - 1]) if k < n else float("nan")),
        ],
        interpretation=(
            "Spectral clustering finds groups a centroid method cannot, because it clusters in the "
            "eigenvector space of the graph rather than in the data space; the eigengap suggests k."
        ),
        payload={
            "labels": labels,
            "embedding": U,
            "centers": centers,
            "eigenvalues": vals,
            "affinity_matrix": W,
            "laplacian": L,
            "n_components": n_comp,
            "estimate": float(vals[k] - vals[k - 1]) if k < n else 0.0,
            "n": int(n),
            "method": f"Unnormalised Laplacian spectral clustering ({aff} affinity) + Lloyd k-means in the embedding",
        },
    )


def cheatsheet():
    return "hmspcl: Spectral clustering: eigenvectors of graph Laplacian"
