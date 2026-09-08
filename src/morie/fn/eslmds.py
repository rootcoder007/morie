# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classical multidimensional scaling (ESL Ch 14.8)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_mds"]


def esl_mds(D, k):
    """
    Classical (Torgerson) MDS: recover coordinates from distances.

    Double-centre the squared distances, B = -(1/2) J D^2 J with
    J = I - 11'/n, then take the top-k eigenpairs: Z = V_k L_k^(1/2).
    The identity underneath is that B is exactly the Gram matrix of
    the centred configuration, so when D really is Euclidean the
    reconstruction is exact up to rotation, reflection and
    translation — which is why the coordinates returned will not
    match any particular original orientation, only the distances.

    Negative eigenvalues are the diagnostic worth watching: they mean
    the input is NOT Euclidean-embeddable, and the routine reports
    them rather than silently discarding them.

    Parameters
    ----------
    D : array-like, shape (n, n)
        Symmetric non-negative dissimilarities, zero diagonal.
    k : int
        Embedding dimension, 1 <= k <= n - 1.

    Returns
    -------
    result : dict
        Keys: estimate (stress), coordinates (row-major n x k),
        eigenvalues, negative_eigenvalue_mass, is_euclidean, stress,
        n, k, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 14.8 (Eq. 14.100);
    Torgerson (1952).

    Examples
    --------
    Distances from points on a line are reproduced exactly:

    >>> import numpy as np
    >>> pts = np.array([[0.0], [1.0], [3.0]])
    >>> D = np.abs(pts - pts.T)
    >>> out = esl_mds(D, 1)
    >>> out["is_euclidean"]
    True
    >>> round(out["stress"], 10)
    0.0

    The recovered configuration has the same pairwise distances even
    though its orientation is arbitrary:

    >>> Z = np.asarray(out["coordinates"]).reshape(3, 1)
    >>> rec = np.abs(Z - Z.T)
    >>> bool(np.allclose(rec, D, atol=1e-9))
    True

    A non-Euclidean dissimilarity is flagged rather than hidden:

    >>> bad = [[0.0, 1.0, 9.0], [1.0, 0.0, 1.0], [9.0, 1.0, 0.0]]
    >>> esl_mds(bad, 1)["is_euclidean"]
    False
    """
    D = np.atleast_2d(np.asarray(D, dtype=float))
    n = D.shape[0]
    k = int(k)
    if D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square; got shape {D.shape}.")
    if not np.allclose(D, D.T):
        raise ValueError("the dissimilarity matrix must be symmetric.")
    if np.any(D < 0):
        raise ValueError("dissimilarities cannot be negative.")
    if not np.allclose(np.diag(D), 0):
        raise ValueError("the dissimilarity matrix must have a zero diagonal.")
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must lie in [1, {n - 1}]; got {k}.")
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    neg = float(-np.sum(vals[vals < -1e-9]))
    pos = np.clip(vals[:k], 0.0, None)
    Z = vecs[:, :k] * np.sqrt(pos)
    for j in range(k):                       # deterministic orientation
        i = int(np.argmax(np.abs(Z[:, j]))) if Z[:, j].any() else 0
        if Z[i, j] < 0:
            Z[:, j] = -Z[:, j]
    rec = np.sqrt(np.maximum(
        np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2), 0.0))
    denom = float(np.sum(D ** 2))
    stress = float(np.sqrt(np.sum((D - rec) ** 2) / denom)) if denom > 0 else 0.0
    return RichResult(payload={
        "estimate": stress, "coordinates": [float(v) for v in Z.ravel()],
        "eigenvalues": [float(v) for v in vals[:k]],
        "negative_eigenvalue_mass": neg,
        "is_euclidean": bool(neg <= 1e-9), "stress": stress,
        "n": int(n), "k": k,
        "method": "classical MDS via double-centred Gram; negative eigenvalues reported"})


def cheatsheet():
    return "eslmds: B = -1/2 J D^2 J, top-k eigenpairs; negatives => not Euclidean"


# compact alias per ledger/NAMING.md
eslmds = esl_mds
