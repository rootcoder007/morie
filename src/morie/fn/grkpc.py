# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kernel PCA with an RBF kernel."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_kernel_pca_rbf"]

_METHOD = "Kernel PCA (RBF kernel)"


def geron_kernel_pca_rbf(X, gamma=1.0, d=2):
    r"""PCA in the feature space, without ever visiting it.

    .. math::
        K_{ij} = \exp(-\gamma \|x_i - x_j\|^2),\qquad
        K_c = K - \mathbf 1_m K - K \mathbf 1_m + \mathbf 1_m K \mathbf 1_m

    then take the top ``d`` eigenvectors of :math:`K_c`.

    The double centering is not optional bookkeeping.  PCA is defined on
    *centred* data, and the feature-space mean cannot be subtracted
    directly because the feature map is never computed -- centering the
    Gram matrix is the algebraically equivalent move.  Skip it and the
    leading component is the mean direction rather than the leading
    direction of variance.

    Projections are scaled by :math:`\sqrt{\lambda}` so that the
    variance along component ``k`` equals :math:`\lambda_k / m`, matching
    ordinary PCA's convention.

    Parameters
    ----------
    X : array-like, shape (m, n)
    gamma : float, optional
        RBF width, positive. Default 1.
    d : int, optional
        Components to keep, ``1 <= d <= m``. Default 2.

    Returns
    -------
    RichResult
        Payload keys ``projected``, ``eigenvalues``, ``eigenvectors``,
        ``explained_variance_ratio``, ``kernel``, ``kernel_centered``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 7, Kernel PCA section.

    Examples
    --------
    The RBF kernel has a unit diagonal, and centering drives every row
    sum of the Gram matrix to zero -- the property the algorithm turns
    on:

    >>> X = [[0.0], [1.0], [2.0], [4.0]]
    >>> r = geron_kernel_pca_rbf(X, gamma=0.5, d=2)
    >>> r["kernel"][0][0]
    1.0
    >>> round(max(abs(sum(row)) for row in r["kernel_centered"]), 12)
    0.0

    Eigenvalues come back sorted and non-negative (the centred Gram
    matrix is positive semi-definite):

    >>> ev = r["eigenvalues"]
    >>> ev == sorted(ev, reverse=True) and min(ev) >= -1e-12
    True
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(f"X must be a non-empty 2-D array, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X must be finite.")
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"gamma must be a positive finite float, got {gamma}.")
    m = A.shape[0]
    d = int(d)
    if not (1 <= d <= m):
        raise ValueError(f"d must lie in [1, {m}], got {d}.")

    sq = np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=2)
    K = np.exp(-gamma * sq)
    one = np.full((m, m), 1.0 / m)
    Kc = K - one @ K - K @ one + one @ K @ one
    Kc = (Kc + Kc.T) / 2.0                   # symmetrise away round-off

    vals, vecs = np.linalg.eigh(Kc)
    idx = np.argsort(vals)[::-1][:d]
    lam = vals[idx]
    V = vecs[:, idx]
    lam_pos = np.clip(lam, 0.0, None)
    Z = V * np.sqrt(lam_pos)

    total = float(np.sum(np.clip(vals, 0.0, None)))
    evr = (lam_pos / total).tolist() if total > 0 else [0.0] * d

    return RichResult(
        title="Kernel PCA (RBF)",
        summary_lines=[("gamma", gamma), ("Components", d),
                       ("Top eigenvalue", float(lam[0]))],
        payload={
            "projected": Z.tolist(),
            "eigenvalues": lam.tolist(),
            "eigenvectors": V.tolist(),
            "explained_variance_ratio": evr,
            "kernel": K.tolist(),
            "kernel_centered": Kc.tolist(),
            "gamma": gamma,
            "estimate": Z.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grkpc: RBF Gram matrix, double-centred, top-d eigenvectors scaled by sqrt(lambda)"
