# morie.fn -- function file (rootcoder007/morie)
"""SMACOF majorization algorithm for (weighted) metric MDS."""

from . import _array_core as np

from ._richresult import RichResult
from .mmdsf import metric_mds_torgerson

__all__ = ["smacof_algorithm"]


def smacof_algorithm(delta, n_dims=2, weights=None, max_iter=300, eps=1e-8):
    r"""Scaling by MAjorizing a COmplicated Function.

    Minimises the (weighted) raw stress

    .. math:: \sigma(X) = \sum_{i<j} w_{ij}\,
              (d_{ij}(X) - \delta_{ij})^2

    by the Guttman transform :math:`X \leftarrow n^{-1} B(X) X` (equal
    weights), where :math:`B(X)` has off-diagonal entries
    :math:`-w_{ij}\,\delta_{ij}/d_{ij}(X)`. Majorization guarantees
    the stress sequence is non-increasing -- asserted in the tests --
    which gradient methods on this objective do not.

    Parameters
    ----------
    delta : array-like, shape (n, n)
        Symmetric target dissimilarities, zero diagonal.
    n_dims : int, default 2
        Embedding dimension.
    weights : array-like, optional
        Symmetric nonnegative weight matrix; default all ones.
    max_iter : int, default 300
    eps : float, default 1e-8
        Stop when the stress improvement falls below eps.

    Returns
    -------
    RichResult
        keys: ``coordinates`` (n, n_dims), ``stress`` (final raw
        stress), ``stress_path``, ``n_iter``, ``converged``, ``n``,
        ``method``.

    References
    ----------
    de Leeuw, J. (1977). Applications of convex analysis to
    multidimensional scaling. In *Recent Developments in Statistics*
    (Barra et al., eds.), North-Holland, 133-145. (the majorization
    step)

    Borg, I. & Groenen, P. J. F. (2005). *Modern Multidimensional
    Scaling* (2nd ed.). Springer. Ch. 8 (SMACOF and the Guttman
    transform).
    """
    Delta = np.asarray(delta, dtype=float)
    if Delta.ndim != 2 or Delta.shape[0] != Delta.shape[1]:
        raise ValueError("delta must be square.")
    n = Delta.shape[0]
    if not np.allclose(Delta, Delta.T, atol=1e-8):
        raise ValueError("delta must be symmetric.")
    k = int(n_dims)
    if not 1 <= k <= n - 1:
        raise ValueError(f"n_dims must lie in [1, {n - 1}], got {k}.")
    if weights is None:
        W = np.ones((n, n)) - np.eye(n)
    else:
        W = np.asarray(weights, dtype=float)
        if W.shape != (n, n) or np.any(W < 0):
            raise ValueError("weights must be a nonnegative (n, n) matrix.")
        W = W * (1 - np.eye(n))

    # V matrix of the weighted problem; Moore-Penrose inverse handles rank n-1
    V = -W.copy()
    np.fill_diagonal(V, W.sum(axis=1))
    Vinv = np.linalg.pinv(V)

    X = metric_mds_torgerson(Delta, n_dims=k)["coordinates"]

    def dists(X):
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff**2).sum(axis=2))

    def stress(X):
        d = dists(X)
        iu = np.triu_indices(n, k=1)
        return float((W[iu] * (d[iu] - Delta[iu]) ** 2).sum())

    path = [stress(X)]
    converged = False
    for it in range(int(max_iter)):
        d = dists(X)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(d > 1e-12, Delta / d, 0.0)
        B = -W * ratio
        np.fill_diagonal(B, -B.sum(axis=1) + np.diag(B))
        X = Vinv @ (B @ X)
        path.append(stress(X))
        if path[-2] - path[-1] < eps:
            converged = True
            break

    return RichResult(
        payload={
            "coordinates": X,
            "stress": path[-1],
            "stress_path": np.array(path),
            "n_iter": len(path) - 1,
            "converged": converged,
            "n": int(n),
            "method": "SMACOF (Guttman-transform majorization)",
        }
    )


def cheatsheet():
    return "smacf: iterate X <- Vinv B(X) X; stress never increases (de Leeuw 1977)"
