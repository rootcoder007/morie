# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Principal component analysis."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_pca"]


def wasserman_pca(X, k):
    """
    PCA of the sample covariance matrix.

    Formula: Sigma v_j = lambda_j v_j; the top-k eigenvectors of the
    (n-1)-divisor covariance of column-centred X. Sign convention:
    each component's largest-magnitude coordinate is made positive,
    so results are deterministic across LAPACK builds (eigenvector
    sign freedom documented, not ignored). Components come back as
    a row-major list of k rows.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Data, n >= 2.
    k : int
        Components to keep, 1 <= k <= d.

    Returns
    -------
    result : dict
        Keys: estimate (first eigenvalue), eigenvalues, components
        (k x d row-major), explained_ratio, scores (n x k row-major),
        n, d, k, method.

    References
    ----------
    Wasserman (2004), Ch 14 (multivariate); Pearson (1901).

    Examples
    --------
    Points on the line y = x load on the (1,1)/sqrt(2) direction:

    >>> X = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
    >>> out = wasserman_pca(X, 1)
    >>> [round(v, 12) for v in out["components"][:2]]
    [0.707106781187, 0.707106781187]
    >>> round(out["estimate"], 12)
    2.0
    >>> out["explained_ratio"]
    1.0
    >>> wasserman_pca(X, 3)
    Traceback (most recent call last):
        ...
    ValueError: k must lie in [1, d]; got k=3, d=2.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    k = int(k)
    if n < 2:
        raise ValueError("PCA needs at least 2 observations.")
    if not 1 <= k <= d:
        raise ValueError(f"k must lie in [1, d]; got k={k}, d={d}.")
    Xc = X - np.mean(X, axis=0)
    S = Xc.T @ Xc / (n - 1)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals = vals[order][:k]
    vecs = vecs[:, order][:, :k]
    for j in range(k):
        i = int(np.argmax(np.abs(vecs[:, j])))
        if vecs[i, j] < 0:
            vecs[:, j] = -vecs[:, j]
    scores = Xc @ vecs
    total = float(np.trace(S))
    return RichResult(payload={
        "estimate": float(vals[0]),
        "eigenvalues": [float(v) for v in vals],
        "components": [float(v) for v in vecs.T.ravel()],
        "explained_ratio": float(np.sum(vals) / total) if total > 0 else float("nan"),
        "scores": [float(v) for v in scores.ravel()],
        "n": int(n), "d": int(d), "k": k,
        "method": "eigh of (n-1)-covariance; sign fixed by max-|coord| positive"})


def cheatsheet():
    return "wsmpca: top-k eigh of cov; deterministic sign; components row-major k x d"
