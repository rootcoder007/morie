# morie.fn -- function file (rootcoder007/morie)
"""Nonmetric MDS via isotonic regression (Kruskal 1964)."""

from . import _array_core as np

from ._richresult import RichResult
from .isotr import isotonic_regression_disparity
from .krust import kruskal_stress
from .mmdsf import metric_mds_torgerson

__all__ = ["nonmetric_mds"]


def nonmetric_mds(delta, n_dims=2, max_iter=100, eps=1e-6):
    r"""Kruskal's nonmetric MDS: alternate disparities and configuration.

    Only the *rank order* of the dissimilarities is honoured:

    1. compute configuration distances :math:`d_{ij}(X)`;
    2. monotone-regress them on the ranks of :math:`\delta_{ij}`
       (PAV) to get disparities :math:`\hat d_{ij}`;
    3. move X one Guttman step toward the disparities;
    4. repeat until stress-1 stops improving.

    Parameters
    ----------
    delta : array-like, shape (n, n)
        Symmetric dissimilarities, zero diagonal; only their order
        matters.
    n_dims : int, default 2
    max_iter : int, default 100
    eps : float, default 1e-6
        Stop when stress-1 improves by less than eps.

    Returns
    -------
    RichResult
        keys: ``coordinates``, ``stress`` (final stress-1),
        ``stress_path``, ``n_iter``, ``converged``, ``n``, ``method``.

    References
    ----------
    Kruskal, J. B. (1964). Multidimensional scaling by optimizing
    goodness of fit to a nonmetric hypothesis. *Psychometrika*, 29(1),
    1-27; and Nonmetric multidimensional scaling: a numerical method.
    *Psychometrika*, 29(2), 115-129.
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

    iu = np.triu_indices(n, k=1)
    ranks = Delta[iu]

    X = metric_mds_torgerson(Delta, n_dims=k)["coordinates"]

    def dmat(X):
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff**2).sum(axis=2))

    def s1(X, Dhat):
        return kruskal_stress(Dhat, dmat(X))["stress"]

    path = []
    converged = False
    Dhat = None
    for it in range(int(max_iter)):
        d = dmat(X)
        disp = isotonic_regression_disparity(d[iu], ranks)["disparities"]
        Dhat = np.zeros((n, n))
        Dhat[iu] = disp
        Dhat = Dhat + Dhat.T
        path.append(s1(X, Dhat))
        if len(path) > 1 and path[-2] - path[-1] < eps:
            converged = True
            break
        # Guttman step toward the disparities
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(d > 1e-12, Dhat / d, 0.0)
        B = -ratio
        np.fill_diagonal(B, -B.sum(axis=1) + np.diag(B))
        X = (B @ X) / n

    return RichResult(
        payload={
            "coordinates": X,
            "stress": path[-1],
            "stress_path": np.array(path),
            "n_iter": len(path),
            "converged": converged,
            "n": int(n),
            "method": "Nonmetric MDS (Kruskal: PAV disparities + Guttman steps)",
        }
    )


def cheatsheet():
    return "nmdsf: alternate PAV disparities and Guttman steps; only ranks of delta matter"
