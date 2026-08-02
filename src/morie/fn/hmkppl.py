# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kernel PCA with polynomial kernel."""

from . import _array_core as np

from ._richresult import RichResult
from .hmkprbf import kernel_pca_from_gram

__all__ = ["geron_kernel_pca_poly"]

_METHOD = "Kernel PCA (polynomial kernel)"


def geron_kernel_pca_poly(X, n_components, degree=3, gamma=None, coef0=1.0):
    """
    Kernel PCA with polynomial kernel.

    Formula: K(x,y) = (gamma * x^T y + coef0)^d

    Unlike the RBF kernel the polynomial feature map is *finite*: degree
    ``d`` over ``n`` features spans ``C(n+d, d)`` monomials, so at most
    that many components carry any variance no matter how many samples
    there are.  That count is returned, because asking for more
    components than the feature space has is a silent source of
    near-zero eigenvalues.

    ``coef0`` is not decoration: with ``coef0 = 0`` the kernel contains
    only degree-``d`` monomials, so the lower-order interactions
    disappear.  The centring and eigen-step are delegated to
    :func:`morie.fn.hmkprbf.kernel_pca_from_gram`.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    n_components : int
        Components to keep.
    degree : int
        Polynomial degree (positive).
    gamma : float, optional
        Scale on the inner product; defaults to ``1/n_features``.
    coef0 : float
        Constant term; ``0`` gives a homogeneous kernel.

    Returns
    -------
    result : RichResult
        Keys: X_projected, eigenvalues, alphas, K, feature_space_dim,
        estimate, n, method.

    Examples
    --------
    Degree 1 with gamma=1, coef0=0 is the linear kernel, so kernel PCA
    reduces to ordinary PCA -- the projection reproduces the centred
    data up to sign:

    >>> X = [[0.0], [1.0], [2.0], [4.0]]
    >>> r = geron_kernel_pca_poly(X, n_components=1, degree=1, gamma=1.0, coef0=0.0)
    >>> centred = np.asarray(X) - np.mean(X)
    >>> p = r["X_projected"][:, 0]
    >>> bool(np.allclose(np.abs(p), np.abs(centred.ravel())))
    True

    The kernel entry is exactly the formula:

    >>> k = geron_kernel_pca_poly([[1.0], [2.0]], n_components=1, degree=2,
    ...                           gamma=1.0, coef0=1.0)
    >>> float(k["K"][0, 1])
    9.0

    Degree 2 in one variable spans ``C(3,2) = 3`` monomials:

    >>> k["feature_space_dim"]
    3

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_kernel_pca_poly: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_kernel_pca_poly: X contains non-finite values")
    m, n_feat = A.shape
    deg = int(degree)
    if deg < 1:
        raise ValueError(f"geron_kernel_pca_poly: degree must be a positive integer, got {degree!r}")
    g = 1.0 / n_feat if gamma is None else float(gamma)
    if not np.isfinite(g) or g <= 0:
        raise ValueError(f"geron_kernel_pca_poly: gamma must be positive and finite, got {gamma!r}")
    c0 = float(coef0)
    if not np.isfinite(c0):
        raise ValueError(f"geron_kernel_pca_poly: coef0 must be finite, got {coef0!r}")

    K = (g * (A @ A.T) + c0) ** deg
    if not np.all(np.isfinite(K)):
        raise ValueError(
            f"geron_kernel_pca_poly: the degree-{deg} kernel overflowed; rescale X or lower gamma/degree"
        )
    proj, vals, alphas, Kc = kernel_pca_from_gram(K, n_components)

    # dim of the space of monomials of degree <= deg in n_feat variables
    # (or exactly deg when coef0 == 0): C(n+d, d) resp. C(n+d-1, d).
    def _comb(a, b):
        out = 1
        for i in range(b):
            out = out * (a - i) // (i + 1)
        return int(out)

    fdim = _comb(n_feat + deg, deg) if c0 != 0 else _comb(n_feat + deg - 1, deg)

    warns = []
    if int(n_components) > fdim:
        warns.append(
            f"asked for {int(n_components)} components but the degree-{deg} feature space has only "
            f"{fdim} dimensions; the extra eigenvalues are numerical noise."
        )
    if c0 == 0:
        warns.append("coef0 = 0 makes the kernel homogeneous: only degree-d monomials, no lower-order terms.")

    return RichResult(
        title="Kernel PCA (polynomial)",
        summary_lines=[
            ("Samples", int(m)),
            ("Degree", deg),
            ("gamma / coef0", f"{g} / {c0}"),
            ("Feature-space dimension", fdim),
        ],
        warnings=warns,
        payload={
            "X_projected": proj,
            "eigenvalues": vals,
            "alphas": alphas,
            "K": K,
            "K_centered": Kc,
            "feature_space_dim": fdim,
            "degree": deg,
            "gamma": g,
            "coef0": c0,
            "estimate": float(vals[0]),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkppl: polynomial kernel PCA (gamma x.y + coef0)^d, with the finite feature-space dimension"
