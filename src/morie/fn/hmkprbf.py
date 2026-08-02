# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kernel PCA with RBF kernel in reproducing kernel Hilbert space."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_kernel_pca_rbf", "kernel_pca_from_gram", "center_gram"]

_METHOD = "Kernel PCA (RBF kernel)"


def center_gram(K):
    """Centre a Gram matrix in feature space.

    ``Kc = K - 1_n K - K 1_n + 1_n K 1_n`` with ``1_n`` the matrix of
    ``1/n``.  Feature vectors cannot be centred directly -- they are
    never formed -- so the centring has to happen on the kernel.
    Skipping it is the most common kernel-PCA bug: the leading component
    then just encodes the mean.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    ones = np.full((n, n), 1.0 / n)
    return K - ones @ K - K @ ones + ones @ K @ ones


def kernel_pca_from_gram(K, n_components):
    """Eigen-decomposition step shared by every kernel-PCA variant.

    Returns ``(projection, eigenvalues, alphas, K_centered)``.  The
    eigenvectors are scaled so the feature-space components are unit
    norm, i.e. ``lambda_i * ||alpha_i||^2 = 1``, which makes the
    projection equal to ``sqrt(lambda_i) * alpha_i``.
    """
    Kc = center_gram(K)
    n = Kc.shape[0]
    d = int(n_components)
    if not (1 <= d <= n):
        raise ValueError(f"kernel_pca_from_gram: n_components must lie in 1..{n}, got {n_components!r}")
    # Symmetrise against round-off before eigh.
    Kc = 0.5 * (Kc + Kc.T)
    vals, vecs = np.linalg.eigh(Kc)
    order = np.argsort(vals)[::-1][:d]
    vals = vals[order]
    vecs = vecs[:, order]
    positive = vals > 1e-12
    if not np.any(positive):
        raise ValueError(
            "kernel_pca_from_gram: the centred kernel has no positive eigenvalue, so there is no "
            "component to project onto (the data may be a single point in feature space)"
        )
    alphas = np.zeros_like(vecs)
    alphas[:, positive] = vecs[:, positive] / np.sqrt(vals[positive])
    proj = np.zeros((n, d))
    proj[:, positive] = vecs[:, positive] * np.sqrt(vals[positive])
    return proj, vals, alphas, Kc


def geron_kernel_pca_rbf(X, n_components, gamma=None):
    """
    Kernel PCA with RBF kernel in reproducing kernel Hilbert space.

    Formula: K(x,y) = exp(-gamma ||x-y||^2); eig of centered K

    The RBF feature map is infinite-dimensional, so the components are
    never formed explicitly -- only their inner products, via the
    kernel.  The eigen-decomposition therefore runs on the ``m x m``
    Gram matrix rather than the ``n x n`` covariance, which is why
    kernel PCA scales with the number of *samples*, not features.

    Centring happens on the kernel (:func:`center_gram`); the shared
    eigen-step is :func:`kernel_pca_from_gram`, which the polynomial and
    sigmoid variants reuse.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    n_components : int
        Components to keep, ``1 <= n_components <= m``.
    gamma : float, optional
        Kernel width.  Defaults to ``1/n_features``, matching the usual
        convention.

    Returns
    -------
    result : RichResult
        Keys: X_projected, eigenvalues, alphas, K, explained_variance_ratio,
        gamma, estimate, n, method.

    Examples
    --------
    Every RBF kernel has a unit diagonal, so the trace of ``K`` is ``m``:

    >>> X = [[0.0], [1.0], [2.0], [5.0]]
    >>> r = geron_kernel_pca_rbf(X, n_components=2, gamma=0.5)
    >>> float(np.trace(r["K"]))
    4.0
    >>> r["X_projected"].shape
    (4, 2)

    The projection is centred: each component sums to zero.

    >>> [abs(round(float(v), 12)) for v in np.sum(r["X_projected"], axis=0)]
    [0.0, 0.0]

    Eigenvalues come back in descending order and are non-negative
    (the centred kernel is PSD):

    >>> ev = r["eigenvalues"]
    >>> bool(np.all(np.diff(ev) <= 1e-12)), bool(np.all(ev > -1e-9))
    (True, True)

    Two well-separated groups separate along the first component:

    >>> g = geron_kernel_pca_rbf([[0.0], [0.1], [9.0], [9.1]], n_components=1, gamma=1.0)
    >>> p = g["X_projected"][:, 0]
    >>> bool(np.sign(p[0]) == np.sign(p[1]) and np.sign(p[2]) == np.sign(p[3]))
    True
    >>> bool(np.sign(p[0]) != np.sign(p[2]))
    True

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_kernel_pca_rbf: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_kernel_pca_rbf: X contains non-finite values")
    m, n_feat = A.shape
    g = 1.0 / n_feat if gamma is None else float(gamma)
    if not np.isfinite(g) or g <= 0:
        raise ValueError(f"geron_kernel_pca_rbf: gamma must be positive and finite, got {gamma!r}")

    sq = np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=2)
    K = np.exp(-g * sq)
    proj, vals, alphas, Kc = kernel_pca_from_gram(K, n_components)

    total = float(np.sum(np.clip(np.linalg.eigvalsh(0.5 * (Kc + Kc.T)), 0.0, None)))
    ratio = np.clip(vals, 0.0, None) / total if total > 0 else np.zeros_like(vals)

    return RichResult(
        title="Kernel PCA (RBF)",
        summary_lines=[
            ("Samples", int(m)),
            ("Components", int(proj.shape[1])),
            ("gamma", g),
            ("Variance explained", float(np.sum(ratio))),
        ],
        interpretation=(
            "The feature space is infinite-dimensional but the eigenproblem is m x m; "
            "kernel PCA costs samples, not features."
        ),
        payload={
            "X_projected": proj,
            "eigenvalues": vals,
            "alphas": alphas,
            "K": K,
            "K_centered": Kc,
            "explained_variance_ratio": ratio,
            "gamma": g,
            "estimate": float(vals[0]),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkprbf: RBF kernel PCA -- centred Gram matrix eigendecomposition (shared core for poly/sigmoid)"
