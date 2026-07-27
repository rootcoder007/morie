# morie.fn -- function file (rootcoder007/morie)
"""Classical metric MDS via the Torgerson double-centering decomposition."""

import numpy as np

from ._richresult import RichResult

__all__ = ["metric_mds_torgerson"]


def metric_mds_torgerson(D_matrix, n_dims=2):
    r"""Torgerson classical (metric) multidimensional scaling.

    .. math:: B = -\tfrac12 H D^{(2)} H, \qquad H = I - \tfrac1n 11',

    followed by the eigendecomposition :math:`B = Q \Lambda Q'` and
    coordinates :math:`X = Q_k \Lambda_k^{1/2}` on the k largest
    positive eigenvalues. When D is exactly Euclidean, B is PSD and
    the configuration reproduces D up to rotation and translation;
    negative eigenvalues measure the non-Euclidean part.

    Parameters
    ----------
    D_matrix : array-like, shape (n, n)
        Symmetric dissimilarity matrix with a zero diagonal.
    n_dims : int, default 2
        Dimensions retained.

    Returns
    -------
    RichResult
        keys: ``coordinates`` (n, n_dims), ``eigenvalues`` (all n,
        descending), ``explained`` (share of the positive-eigenvalue
        mass retained), ``n``, ``method``.

    References
    ----------
    Torgerson, W. S. (1952). Multidimensional scaling: I. Theory and
    method. *Psychometrika*, 17(4), 401-419.

    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Sec. 3.1 (Classical Metric MDS),
    p. 68.
    """
    D = np.asarray(D_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D_matrix must be square.")
    n = D.shape[0]
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("D_matrix must be symmetric.")
    if not np.allclose(np.diag(D), 0.0, atol=1e-8):
        raise ValueError("D_matrix must have a zero diagonal.")
    k = int(n_dims)
    if not 1 <= k <= n - 1:
        raise ValueError(f"n_dims must lie in [1, {n - 1}], got {k}.")

    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D**2) @ H
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    pos = np.maximum(vals[:k], 0.0)
    X = vecs[:, :k] * np.sqrt(pos)

    pos_mass = vals[vals > 0].sum()
    explained = float(pos[pos > 0].sum() / pos_mass) if pos_mass > 0 else 0.0

    return RichResult(
        payload={
            "coordinates": X,
            "eigenvalues": vals,
            "explained": explained,
            "n": int(n),
            "method": "Classical metric MDS (Torgerson double centering)",
        }
    )


def cheatsheet():
    return "mmdsf: B = -1/2 H D^2 H; X = Q_k Lambda_k^(1/2) (Torgerson 1952)"
