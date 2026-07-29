# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Principal components via SVD of the centred data matrix."""

import numpy as np

from ._richresult import RichResult
from .pcsvd import pca_svd

__all__ = ["geron_principal_components"]


def geron_principal_components(X, n_components=None, center=True, scale=False):
    """
    Principal components via SVD of the centred data matrix.

    Formula: X_c = U Sigma V^T; PCs = columns of V

    The decomposition is DELEGATED to the finished implementation
    :func:`morie.fn.pcsvd.pca_svd`. Two things this adds are worth the
    wrapper: the sign convention (each component is flipped so its
    largest-magnitude loading is positive, since V is only defined up to
    sign and unstable signs make runs look different when they are not),
    and the reconstruction error, which is the quantity PCA actually
    minimises.

    Centering is not optional in spirit -- without it the first component
    chases the mean rather than the variance.

    Parameters
    ----------
    X : array-like, shape (m, p)
    n_components : int, optional
        Components to keep; default all.
    center : bool, default True
        Subtract the column means.
    scale : bool, default False
        Standardise columns first (correlation-matrix PCA).

    Returns
    -------
    result : RichResult
        Keys: components, scores, explained_variance,
        explained_variance_ratio, reconstruction_error, estimate, n,
        method.

    Examples
    --------
    All the variance of this cloud lies on the first axis:

    >>> X = [[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]
    >>> r = geron_principal_components(X, 2)
    >>> [round(float(v), 12) for v in r["explained_variance_ratio"]]
    [1.0, 0.0]
    >>> [round(float(v), 12) for v in r["components"][:, 0]]
    [1.0, 0.0]

    Keeping that one component reconstructs the data exactly:

    >>> round(float(geron_principal_components(X, 1)["reconstruction_error"]), 12)
    0.0

    References
    ----------
    Geron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"geron_principal_components: X must be 2-D, got ndim={A.ndim}")
    m, p = A.shape
    if m < 2:
        raise ValueError(f"geron_principal_components: need at least 2 rows to have variance, got {m}")
    if p < 1:
        raise ValueError("geron_principal_components: X has no columns")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_principal_components: X contains non-finite values")
    k = min(m, p) if n_components is None else int(n_components)
    if not (1 <= k <= min(m, p)):
        raise ValueError(f"geron_principal_components: n_components must lie in [1, {min(m, p)}], got {n_components!r}")

    res = pca_svd(A, n_components=k, center=center, scale=scale)
    comps = np.asarray(res.components, dtype=float)
    scores = np.asarray(res.scores, dtype=float)

    # Sign convention: largest-magnitude loading of each component positive.
    for j in range(comps.shape[1]):
        if comps[np.argmax(np.abs(comps[:, j])), j] < 0:
            comps[:, j] *= -1.0
            scores[:, j] *= -1.0

    Xc = A - A.mean(axis=0) if center else A
    if scale:
        sd = Xc.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        Xc = Xc / sd
    err = float(np.linalg.norm(Xc - scores @ comps.T, "fro"))

    return RichResult(
        title="Principal components",
        summary_lines=[
            ("Components kept", k),
            ("Variance explained", float(np.sum(res.explained_variance_ratio))),
            ("Reconstruction error", err),
        ],
        interpretation="PCA minimises exactly this reconstruction error; components are unique only up to sign.",
        payload={
            "components": comps,
            "scores": scores,
            "explained_variance": np.asarray(res.explained_variance, dtype=float),
            "explained_variance_ratio": np.asarray(res.explained_variance_ratio, dtype=float),
            "reconstruction_error": err,
            "n_components": k,
            "estimate": comps,
            "n": int(m),
            "method": "PCA by SVD, delegated to morie.fn.pcsvd.pca_svd",
        },
    )


def cheatsheet():
    return "hmpcac: Principal components via SVD of centred data"
