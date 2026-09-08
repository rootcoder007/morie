# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Locally linear embedding (LLE): preserve local linear reconstruction weights."""

from . import _array_core as np

from ._richresult import RichResult
from .hmmds import pairwise_distances

__all__ = ["geron_locally_linear_embedding"]

_METHOD = "Locally linear embedding"


def geron_locally_linear_embedding(X, n_components, n_neighbors=5, reg=1e-3):
    """
    Locally linear embedding (LLE): preserve local linear reconstruction weights.

    Formula: min sum_i ||x_i - sum_j w_ij x_j||^2 then embed preserving W

    Two steps, both closed-form.  First, each point is written as a
    weighted average of its ``k`` neighbours, weights summing to 1 --
    that sum-to-one constraint is what makes the weights invariant to
    translation, rotation and scaling of the neighbourhood, and it is
    the reason the recipe works at all.  Second, a low-dimensional
    configuration is found that the *same* weights reconstruct, by
    taking the bottom eigenvectors of ``M = (I - W)^T (I - W)``.

    The bottom eigenvector of ``M`` is the constant vector with
    eigenvalue 0 (every row of ``W`` sums to one) and carries no
    information; it is discarded, and components ``1 .. d`` are kept.
    Forgetting to drop it is the classic LLE off-by-one.

    When ``n_neighbors > n_features`` the local Gram matrix is singular,
    so it is regularised by ``reg * trace(G) / k`` -- without that the
    weights are arbitrary within a null space.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    n_components : int
        Embedding dimension, ``1 <= n_components <= m - 2``.
    n_neighbors : int
        Neighbours per point.
    reg : float
        Relative ridge added to each local Gram matrix.

    Returns
    -------
    result : RichResult
        Keys: embedding, weights, reconstruction_error, eigenvalues,
        estimate, n, method.

    Examples
    --------
    The reconstruction weights sum to one for every point:

    >>> X = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
    >>> r = geron_locally_linear_embedding(X, n_components=1, n_neighbors=2)
    >>> [round(float(v), 9) for v in np.sum(r["weights"], axis=1)]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    Points evenly spaced on a line embed to a monotone 1-D coordinate:

    >>> e = r["embedding"][:, 0]
    >>> order = np.argsort(e)
    >>> bool(np.all(order == np.arange(6)) or np.all(order == np.arange(5, -1, -1)))
    True

    Interior points on a line are the exact midpoint of their two
    neighbours, so their reconstruction error is zero:

    >>> round(float(r["reconstruction_error"][2]), 9)
    0.0

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_locally_linear_embedding: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_locally_linear_embedding: X contains non-finite values")
    m = A.shape[0]
    k = int(n_neighbors)
    if not (1 <= k < m):
        raise ValueError(f"geron_locally_linear_embedding: n_neighbors must lie in 1..{m - 1}, got {n_neighbors!r}")
    d = int(n_components)
    if not (1 <= d <= m - 2):
        raise ValueError(
            f"geron_locally_linear_embedding: n_components must lie in 1..{m - 2} "
            f"(the constant bottom eigenvector is discarded), got {n_components!r}"
        )
    r = float(reg)
    if not np.isfinite(r) or r < 0:
        raise ValueError(f"geron_locally_linear_embedding: reg must be non-negative and finite, got {reg!r}")

    D = pairwise_distances(A)
    W = np.zeros((m, m))
    err = np.zeros(m)
    for i in range(m):
        nb = [j for j in np.argsort(D[i], kind="mergesort") if j != i][:k]
        Z = A[nb] - A[i]
        G = Z @ Z.T
        trace = float(np.trace(G))
        ridge = r * (trace if trace > 0 else 1.0)
        G = G + ridge * np.eye(k) / k
        try:
            w = np.linalg.solve(G, np.ones(k))
        except np.linalg.LinAlgError:
            raise ValueError(
                f"geron_locally_linear_embedding: the local Gram matrix at point {i} is singular even after "
                f"regularisation; raise reg or lower n_neighbors"
            ) from None
        s = float(np.sum(w))
        if s == 0:
            raise ValueError(
                f"geron_locally_linear_embedding: reconstruction weights at point {i} sum to zero, "
                f"so they cannot be normalised to one"
            )
        w = w / s
        W[i, nb] = w
        err[i] = float(np.sum((A[i] - w @ A[nb]) ** 2))

    I = np.eye(m)
    M = (I - W).T @ (I - W)
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    # vals ascending; index 0 is the constant vector with eigenvalue ~0.
    emb = vecs[:, 1 : d + 1]

    return RichResult(
        title="Locally linear embedding",
        summary_lines=[
            ("Points", int(m)),
            ("Neighbours", k),
            ("Dimensions", d),
            ("Total reconstruction error", float(np.sum(err))),
        ],
        interpretation=(
            "Weights summing to one make the local description invariant to translation, rotation and "
            "scale; the bottom eigenvector of M is constant and is dropped."
        ),
        payload={
            "embedding": emb,
            "weights": W,
            "reconstruction_error": err,
            "eigenvalues": vals[: d + 1],
            "M": M,
            "n_neighbors": k,
            "estimate": float(np.sum(err)),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlle: LLE -- sum-to-one local weights, then bottom eigenvectors of (I-W)^T(I-W) minus the constant one"
