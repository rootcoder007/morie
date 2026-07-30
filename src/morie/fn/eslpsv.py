# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PCA by SVD (ESL Ch 14.5.1)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_pca_svd", "esl_pca_transform"]


def esl_pca_svd(X, k, center=True, scale=False):
    """
    PCA through the SVD: X = U D V', principal components are the
    columns of V.

    Going through the SVD rather than eigendecomposing X'X is the
    point of this routine: forming X'X squares the condition number,
    so directions whose singular values are near the numerical floor
    are lost. The SVD reaches them.

    Centring is on by default because uncentred PCA finds the
    direction of the mean rather than of the variation. Scaling is
    OFF by default, matching ESL, but must be turned on when the
    columns are in different units — otherwise whichever variable has
    the largest numbers dominates the first component regardless of
    its importance. Both choices are reported.

    Sign convention: each component's largest-magnitude loading is
    made positive, so results are reproducible across LAPACK builds
    (singular vectors are only defined up to sign).

    Parameters
    ----------
    X : array-like, shape (n, p)
        Data, n >= 2.
    k : int
        Components to keep, 1 <= k <= min(n, p).
    center : bool
        Subtract the column means.
    scale : bool
        Divide by the column standard deviations.

    Returns
    -------
    result : dict
        Keys: estimate (first eigenvalue), eigenvalues,
        singular_values, components (row-major k x p), scores
        (row-major n x k), explained_variance_ratio,
        cumulative_ratio, centered, scaled, n, p, k, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 14.5.1 (Eq. 14.54).

    Examples
    --------
    Points on the line y = x load entirely on (1,1)/sqrt(2):

    >>> X = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
    >>> out = esl_pca_svd(X, 1)
    >>> [round(v, 12) for v in out["components"]]
    [0.707106781187, 0.707106781187]
    >>> round(out["explained_variance_ratio"][0], 12)
    1.0
    >>> round(out["estimate"], 12)
    2.0

    Scores are the data projected onto the components:

    >>> [round(v, 10) for v in out["scores"]]
    [-1.4142135624, 0.0, 1.4142135624]
    >>> [round(v, 10) for v in esl_pca_transform(out, [[2.0, 2.0]])]
    [2.8284271247]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    k = int(k)
    if n < 2:
        raise ValueError("PCA needs at least 2 observations.")
    kmax = min(n, p)
    if not 1 <= k <= kmax:
        raise ValueError(f"k must lie in [1, {kmax}]; got {k}.")
    mu = X.mean(axis=0) if center else np.zeros(p)
    Z = X - mu
    sd = np.ones(p)
    if scale:
        sd = Z.std(axis=0, ddof=1)
        if np.any(sd == 0):
            raise ValueError("a constant column cannot be scaled; drop it or set scale=False.")
        Z = Z / sd
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt.T[:, :k].copy()
    for j in range(k):
        i = int(np.argmax(np.abs(V[:, j])))
        if V[i, j] < 0:
            V[:, j] = -V[:, j]
    eig = (S ** 2) / (n - 1)
    total = float(np.sum(eig))
    ratio = (eig[:k] / total) if total > 0 else np.full(k, np.nan)
    scores = Z @ V
    return RichResult(payload={
        "estimate": float(eig[0]),
        "eigenvalues": [float(v) for v in eig[:k]],
        "singular_values": [float(v) for v in S[:k]],
        "components": [float(v) for v in V.T.ravel()],
        "scores": [float(v) for v in scores.ravel()],
        "explained_variance_ratio": [float(v) for v in ratio],
        "cumulative_ratio": [float(v) for v in np.cumsum(ratio)],
        "mean": [float(v) for v in mu], "sd": [float(v) for v in sd],
        "centered": bool(center), "scaled": bool(scale),
        "n": int(n), "p": int(p), "k": k,
        "method": "PCA via SVD; sign fixed by max-|loading|; scaling off by default"})


def esl_pca_transform(model, X):
    """
    Project new rows onto components from [esl_pca_svd], reusing the
    training mean and scale.

    Parameters
    ----------
    model : dict
        Payload from esl_pca_svd.
    X : array-like, shape (m, p)

    Returns
    -------
    list of float
        Scores, row-major m x k.

    Examples
    --------
    >>> m = esl_pca_svd([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], 1)
    >>> [round(v, 10) for v in esl_pca_transform(m, [[0.0, 0.0]])]
    [0.0]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu = np.asarray(model["mean"], dtype=float)
    sd = np.asarray(model["sd"], dtype=float)
    k, p = model["k"], model["p"]
    V = np.asarray(model["components"], dtype=float).reshape(k, p).T
    Z = (X - mu) / sd
    return [float(v) for v in (Z @ V).ravel()]


def cheatsheet():
    return "eslpsv: SVD not eig(X'X); centre by default, scale when units differ"
