# morie.fn -- function file (rootcoder007/morie)
"""Locally linear embedding -- Roweis & Saul (2000), ESL Sec 14.9."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_lle"]


def esl_lle(X, k=2, neighbors=5, reg=1e-3):
    r"""Embed data by preserving local linear reconstruction weights.

    Two steps. First, reconstruct each point from its neighbours,

    .. math::
        \min_{W} \sum_i \Big\lVert x_i - \sum_{j \in N(i)} w_{ij} x_j \Big\rVert^2
        \quad\text{s.t.}\quad \sum_j w_{ij} = 1 ,

    whose sum-to-one constraint makes :math:`W` invariant to rotation,
    rescaling and translation of each neighbourhood -- that invariance is
    exactly why the weights characterise local geometry rather than local
    position. Second, find low-dimensional :math:`Y` that the *same* weights
    reconstruct, as the bottom eigenvectors of
    :math:`M = (I-W)^\top (I-W)`.

    The bottom eigenvector is the constant vector with eigenvalue ~0, and is
    discarded; forgetting to do so is the classic LLE implementation bug and
    yields an embedding one dimension short.

    Where a neighbourhood has more members than the ambient dimension the
    Gram matrix is singular, so ``reg`` conditions it -- required, not
    cosmetic.

    Parameters
    ----------
    X : array-like
        Data ``(n, p)``.
    k : int
        Embedding dimension.
    neighbors : int
        Neighbourhood size.
    reg : float
        Relative ridge on the local Gram matrix.

    Returns
    -------
    RichResult
        ``embedding`` ``(n, k)``, ``weights`` ``(n, n)``, ``eigenvalues``,
        ``reconstruction_error``.

    References
    ----------
    Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality reduction
        by locally linear embedding. *Science*, 290(5500), 2323-2326.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    The reconstruction weights sum to one for every point, which is the
    constraint that buys the invariance.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = rng.uniform(0, 4 * np.pi, 300)
    >>> X = np.column_stack([t * np.cos(t), t * np.sin(t), rng.normal(0, 0.05, 300)])
    >>> r = esl_lle(X, k=2, neighbors=10)
    >>> bool(np.allclose(r["weights"].sum(axis=1), 1.0))
    True

    A spiral's embedding recovers its arclength parameter.

    >>> bool(abs(np.corrcoef(r["embedding"][:, 0], t)[0, 1]) > 0.85)
    True

    The constant eigenvector is dropped, so the embedding has exactly ``k``
    non-degenerate columns.

    >>> r["embedding"].shape
    (300, 2)
    >>> bool(np.all(r["embedding"].std(axis=0) > 1e-8))
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    k, m = int(k), int(neighbors)
    if not 1 <= k < n:
        raise ValueError(f"k must be between 1 and {n - 1}")
    if not 1 <= m < n:
        raise ValueError(f"neighbors must be between 1 and {n - 1}")

    D = ((X[:, None] - X[None]) ** 2).sum(-1)
    idx = np.argsort(D, axis=1)[:, 1: m + 1]

    W = np.zeros((n, n))
    err = 0.0
    for i in range(n):
        Zn = X[idx[i]] - X[i]
        C = Zn @ Zn.T
        C = C + reg * np.trace(C) * np.eye(m) if np.trace(C) > 0 else C + reg * np.eye(m)
        w = np.linalg.solve(C, np.ones(m))
        w /= w.sum()
        W[i, idx[i]] = w
        err += float(np.sum((X[i] - w @ X[idx[i]]) ** 2))

    I = np.eye(n)
    M = (I - W).T @ (I - W)
    w_eig, V = np.linalg.eigh((M + M.T) / 2)
    # Skip the first eigenvector: it is constant with eigenvalue ~0 and
    # carries no embedding information.
    emb = V[:, 1: k + 1]
    return RichResult(
        title="Locally linear embedding",
        summary_lines=[("n", n), ("k", k), ("neighbors", m),
                       ("reconstruction error", err / n)],
        payload={
            "embedding": emb * np.sqrt(n), "weights": W,
            "eigenvalues": w_eig[: k + 1],
            "reconstruction_error": float(err / n),
            "neighbors": m,
            "method": "esl_lle",
        },
    )


def cheatsheet():
    return "esllle: weights sum to 1 (invariance); DROP the constant bottom eigenvector or you lose a dimension"
