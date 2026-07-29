# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classical multidimensional scaling (MDS) preserves pairwise distances."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_mds", "pairwise_distances", "double_center"]

_METHOD = "Classical (Torgerson) multidimensional scaling"


def pairwise_distances(X):
    """Euclidean distance matrix of the rows of ``X``."""
    X = np.asarray(X, dtype=float)
    sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.clip(sq, 0.0, None))


def double_center(D):
    """``B = -0.5 * J D^2 J`` with ``J = I - 11^T/n``.

    Double-centring turns a squared-distance matrix into the Gram matrix
    of a centred configuration -- the identity that makes classical MDS a
    single eigen-decomposition instead of an optimisation.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    return -0.5 * (J @ (D**2) @ J)


def geron_mds(X, n_components, precomputed=False):
    """
    Classical multidimensional scaling (MDS) preserves pairwise distances.

    Formula: min_Y sum_{ij} (d_ij - ||y_i - y_j||)^2

    Classical (Torgerson) MDS solves this in closed form rather than by
    iteration: double-centre the squared distances to get a Gram matrix
    ``B``, take its top eigenpairs, and set ``Y = V sqrt(Lambda)``.  On
    Euclidean input the answer is exactly PCA -- same eigenvalues, same
    configuration up to rotation and reflection -- which is the useful
    fact: MDS only differs when the distances are *not* Euclidean.

    A negative eigenvalue of ``B`` is the signature of a non-Euclidean
    distance matrix; those components are dropped and the count is
    reported, since silently taking ``sqrt`` of a negative eigenvalue is
    how this algorithm produces complex nonsense.

    Parameters
    ----------
    X : array-like
        Data, shape (m, n); or a square distance matrix if
        ``precomputed=True``.
    n_components : int
        Embedding dimension.
    precomputed : bool
        Treat ``X`` as a symmetric distance matrix with zero diagonal.

    Returns
    -------
    result : RichResult
        Keys: embedding, eigenvalues, stress, distance_matrix,
        n_negative_eigenvalues, estimate, n, method.

    Examples
    --------
    Four points on a line embed back onto a line with the distances
    exactly preserved, so the stress is zero:

    >>> X = [[0.0], [1.0], [3.0], [6.0]]
    >>> r = geron_mds(X, n_components=1)
    >>> round(r["stress"], 12)
    0.0
    >>> emb = r["embedding"][:, 0]
    >>> [round(float(v), 9) for v in np.abs(emb - emb[0])]
    [0.0, 1.0, 3.0, 6.0]

    From a precomputed distance matrix of an equilateral triangle, the
    two-dimensional embedding has all three sides equal to 1:

    >>> D = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    >>> t = geron_mds(D, n_components=2, precomputed=True)
    >>> from morie.fn.hmmds import pairwise_distances
    >>> [round(float(v), 9) for v in pairwise_distances(t["embedding"])[0]]
    [0.0, 1.0, 1.0]

    A distance matrix that violates the triangle inequality is not
    Euclidean and produces negative eigenvalues, which are reported:

    >>> bad = [[0.0, 1.0, 9.0], [1.0, 0.0, 1.0], [9.0, 1.0, 0.0]]
    >>> geron_mds(bad, n_components=1, precomputed=True)["n_negative_eigenvalues"] >= 1
    True

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_mds: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_mds: X contains non-finite values")

    if precomputed:
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"geron_mds: a precomputed distance matrix must be square, got shape {A.shape}")
        if not np.allclose(A, A.T):
            raise ValueError("geron_mds: the precomputed distance matrix is not symmetric")
        if np.any(np.diag(A) != 0):
            raise ValueError("geron_mds: the precomputed distance matrix must have a zero diagonal")
        if np.any(A < 0):
            raise ValueError("geron_mds: distances must be non-negative")
        D = A
    else:
        D = pairwise_distances(A)

    m = D.shape[0]
    d = int(n_components)
    if not (1 <= d <= m):
        raise ValueError(f"geron_mds: n_components must lie in 1..{m}, got {n_components!r}")

    B = double_center(D)
    B = 0.5 * (B + B.T)
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    tol = 1e-8 * max(1.0, float(np.max(np.abs(vals))))
    n_neg = int(np.count_nonzero(vals < -tol))

    keep = vals[:d]
    pos = keep > tol
    emb = np.zeros((m, d))
    emb[:, pos] = vecs[:, :d][:, pos] * np.sqrt(keep[pos])

    D_emb = pairwise_distances(emb)
    iu = np.triu_indices(m, 1)
    stress = float(np.sum((D[iu] - D_emb[iu]) ** 2))

    warns = []
    if n_neg:
        warns.append(
            f"{n_neg} eigenvalue(s) of the double-centred matrix are negative: the distances are not "
            f"Euclidean and those dimensions were dropped rather than square-rooted."
        )

    return RichResult(
        title="Classical MDS",
        summary_lines=[
            ("Points", int(m)),
            ("Dimensions", d),
            ("Raw stress", stress),
            ("Negative eigenvalues", n_neg),
        ],
        warnings=warns,
        interpretation=(
            "On Euclidean input classical MDS is PCA in disguise; it earns its keep only when the "
            "distances come from somewhere else."
        ),
        payload={
            "embedding": emb,
            "eigenvalues": vals,
            "stress": stress,
            "distance_matrix": D,
            "embedded_distances": D_emb,
            "n_negative_eigenvalues": n_neg,
            "estimate": stress,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmds: classical MDS -- double-centre D^2, eigendecompose, Y = V sqrt(Lambda); flags non-Euclidean D"
