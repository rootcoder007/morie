# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Randomized PCA via a random projection of the data matrix."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_randomized_pca"]


def _lcg_matrix(rows, cols, seed):
    """Uniform(-1, 1) matrix from the integer LCG, identical on every machine."""
    s = int(seed) % 2**32
    out = np.empty(rows * cols)
    for i in range(rows * cols):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (s + 0.5) / 2**32 * 2.0 - 1.0
    return out.reshape(rows, cols)


def geron_randomized_pca(X, n_components, seed=0, n_oversamples=10, n_power_iter=2):
    """
    Randomized PCA using a random projection to approximate top components.

    Formula: X_approx = X * Q; PCA on X_approx

    A random sketch of k + p columns almost surely spans the leading
    subspace, so a full SVD of an m x n matrix (O(m n min(m,n))) becomes
    an SVD of an m x (k+p) one. That is the whole trick, and it is only
    a good one when k is much smaller than the rank: the reported
    ``spectral_gap`` says how fast the tail decays, and with a flat
    spectrum the sketch has nothing to lock onto. ``n_power_iter``
    sharpens the separation by pushing the sketch through (X X^T)^q.

    Parameters
    ----------
    X : array-like, shape (m, p)
    n_components : int
        Components to recover (>= 1).
    seed : int, default 0
        Seed of the integer LCG used for the sketch.
    n_oversamples : int, default 10
        Extra sketch columns.
    n_power_iter : int, default 2
        Power iterations.

    Returns
    -------
    result : RichResult
        Keys: components, scores, explained_variance,
        explained_variance_ratio, singular_values, spectral_gap,
        estimate, n, method.

    Examples
    --------
    A rank-1 cloud: the centred rows are (-1,-1), (0,0), (1,1), whose
    only singular value is 2, so the variance is 4/(3-1) = 2:

    >>> r = geron_randomized_pca([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], 1)
    >>> round(float(r["singular_values"][0]), 9)
    2.0
    >>> round(float(r["explained_variance"][0]), 9)
    2.0
    >>> round(float(r["explained_variance_ratio"][0]), 9)
    1.0

    References
    ----------
    Geron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"geron_randomized_pca: X must be 2-D, got ndim={A.ndim}")
    m, p = A.shape
    if m < 2:
        raise ValueError(f"geron_randomized_pca: need at least 2 rows, got {m}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_randomized_pca: X contains non-finite values")
    k = int(n_components)
    if not (1 <= k <= min(m, p)):
        raise ValueError(f"geron_randomized_pca: n_components must lie in [1, {min(m, p)}], got {n_components!r}")
    over = int(n_oversamples)
    if over < 0:
        raise ValueError(f"geron_randomized_pca: n_oversamples must be >= 0, got {n_oversamples!r}")
    q = int(n_power_iter)
    if q < 0:
        raise ValueError(f"geron_randomized_pca: n_power_iter must be >= 0, got {n_power_iter!r}")

    Xc = A - A.mean(axis=0)
    ell = min(p, k + over)
    Omega = _lcg_matrix(p, ell, seed)
    Y = Xc @ Omega
    Q, _ = np.linalg.qr(Y)
    for _ in range(q):
        Q, _ = np.linalg.qr(Xc.T @ Q)
        Q, _ = np.linalg.qr(Xc @ Q)
    B = Q.T @ Xc
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)

    comps = Vt[:k].T.copy()
    sv = S[:k].copy()
    for j in range(comps.shape[1]):
        if comps[np.argmax(np.abs(comps[:, j])), j] < 0:
            comps[:, j] *= -1.0
    scores = Xc @ comps
    var = sv**2 / (m - 1)
    total = float(np.sum(Xc**2) / (m - 1))
    ratio = var / total if total > 0 else np.zeros_like(var)
    gap = float(S[k - 1] / S[k]) if S.size > k and S[k] > 0 else float("inf")

    return RichResult(
        title="Randomized PCA",
        summary_lines=[("Components", k), ("Sketch width", int(ell)), ("Variance explained", float(np.sum(ratio)))],
        interpretation="The sketch only helps when the spectrum decays; a flat spectrum has no dominant subspace.",
        payload={
            "components": comps,
            "scores": scores,
            "singular_values": sv,
            "explained_variance": var,
            "explained_variance_ratio": ratio,
            "spectral_gap": gap,
            "sketch_width": int(ell),
            "estimate": comps,
            "n": int(m),
            "method": "Randomized range finder (LCG sketch) plus SVD of the projected matrix",
        },
    )


def cheatsheet():
    return "hmrpca: Randomized PCA via random projection"
