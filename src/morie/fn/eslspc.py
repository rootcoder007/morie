# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spectral clustering (ESL Ch 14.5.3)."""

import numpy as np

from ._richresult import RichResult
from .wsmkmn import wasserman_kmeans

__all__ = ["esl_spectral_cluster"]


def esl_spectral_cluster(W, k, normalized=True):
    """
    Spectral clustering on a similarity graph.

    Build the Laplacian from the similarity matrix W, take the
    eigenvectors of its k smallest eigenvalues, and cluster the rows
    of that embedding with k-means. The reason it beats k-means on
    raw coordinates is the reason it exists: the embedding is derived
    from CONNECTIVITY, so it separates groups that are connected but
    not compact — two interleaved rings, for instance, which k-means
    cannot split at all.

    normalized=True uses the symmetric Laplacian
    L_sym = I - D^-1/2 W D^-1/2 (Ng-Jordan-Weiss) with rows of the
    embedding renormalised to unit length; False uses the unnormalised
    L = D - W. The normalised version is the usual default because it
    is not dominated by high-degree vertices.

    The number of eigenvalues at (numerically) zero equals the number
    of connected components, which is reported — if that count already
    exceeds k, the clustering is decided by the graph rather than by
    k-means, and it is worth knowing that happened.

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric non-negative similarities, zero diagonal.
    k : int
        Clusters, 2 <= k <= n.
    normalized : bool
        Use the symmetric normalised Laplacian.

    Returns
    -------
    result : dict
        Keys: estimate (within-cluster sum of squares in the
        embedding), labels (0-based), eigenvalues, n_components,
        embedding (row-major n x k), normalized, n, k, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 14.5.3;
    Ng, Jordan & Weiss (2002).

    Examples
    --------
    Two disconnected triangles are separated exactly, and the graph
    announces its two components through the zero eigenvalues:

    >>> W = [[0,1,1,0,0,0],
    ...      [1,0,1,0,0,0],
    ...      [1,1,0,0,0,0],
    ...      [0,0,0,0,1,1],
    ...      [0,0,0,1,0,1],
    ...      [0,0,0,1,1,0]]
    >>> out = esl_spectral_cluster(W, 2)
    >>> out["n_components"]
    2
    >>> lab = out["labels"]
    >>> lab[0] == lab[1] == lab[2]
    True
    >>> lab[3] == lab[4] == lab[5]
    True
    >>> lab[0] != lab[3]
    True
    """
    W = np.atleast_2d(np.asarray(W, dtype=float))
    n = W.shape[0]
    k = int(k)
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square; got shape {W.shape}.")
    if not np.allclose(W, W.T):
        raise ValueError("the similarity matrix must be symmetric.")
    if np.any(W < 0):
        raise ValueError("similarities cannot be negative.")
    if not 2 <= k <= n:
        raise ValueError(f"k must lie in [2, {n}]; got {k}.")
    d = W.sum(axis=1)
    if np.any(d <= 0):
        raise ValueError("every vertex needs at least one positive similarity; "
                         "an isolated vertex has no defined normalised Laplacian.")
    if normalized:
        dm = 1.0 / np.sqrt(d)
        L = np.eye(n) - (W * dm[:, None]) * dm[None, :]
    else:
        L = np.diag(d) - W
    vals, vecs = np.linalg.eigh(L)
    U = vecs[:, :k]
    if normalized:
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        U = U / norms
    km = wasserman_kmeans(U, k)
    return RichResult(payload={
        "estimate": km["estimate"], "labels": km["labels"],
        "eigenvalues": [float(v) for v in vals[:k]],
        "n_components": int(np.sum(vals < 1e-8)),
        "embedding": [float(v) for v in U.ravel()],
        "normalized": bool(normalized), "n": int(n), "k": k,
        "method": "spectral clustering on the Laplacian embedding, k-means on rows"})


def cheatsheet():
    return "eslspc: cluster Laplacian eigenvectors; finds connected-not-compact groups"
