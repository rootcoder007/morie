# morie.fn -- function file (rootcoder007/morie)
"""Differentially private PCA."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from .dpcov import dp_covariance

__all__ = ["dp_pca"]


def dp_pca(X, k=2, epsilon=1.0, delta=1e-5, C=1.0, seed=None):
    r"""Top-``k`` principal directions from a privately released covariance.

    Releases :math:`\hat\Sigma` by the Gaussian mechanism and eigendecomposes
    it. Everything after the noisy covariance is **post-processing**, so the
    eigenvectors, the projections and any downstream model inherit the same
    :math:`(\varepsilon, \delta)` without further cost. That is the property
    that makes this worth doing rather than privatising each downstream
    quantity separately.

    Accuracy is governed by the **eigengap**. The perturbation of a subspace
    is bounded by noise divided by the gap :math:`\lambda_k - \lambda_{k+1}`
    (Davis-Kahan), so when two eigenvalues are close their directions mix
    almost freely and the released subspace is unstable even at a generous
    budget. ``eigengap`` is returned for exactly that reason -- a small one
    means the individual components should not be interpreted, even though the
    subspace they span may still be fine.

    Parameters
    ----------
    X : array-like
        Data ``(n, p)``.
    k : int
        Number of components, 1 to ``p``.
    epsilon, delta : float
        Privacy budget.
    C : float
        Row-norm clipping bound.
    seed : int, optional
        Seed; leave ``None`` for a real release.

    Returns
    -------
    RichResult
        ``components`` ``(p, k)``, ``eigenvalues``, ``eigengap``,
        ``explained_variance_ratio``, ``scores``.

    References
    ----------
    Chaudhuri, K., Sarwate, A. D., & Sinha, K. (2013). Near-optimal
        algorithm for differentially-private principal components. *JMLR*, 14,
        2905-2943.

    Examples
    --------
    With a strong signal direction and a workable budget, the private first
    component aligns with the true one.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> v = np.array([1.0, 0.0, 0.0, 0.0])
    >>> X = rng.normal(size=(4000, 1)) * v * 3 + rng.normal(size=(4000, 4)) * 0.1
    >>> r = dp_pca(X, k=1, epsilon=8.0, C=6.0, seed=0)
    >>> bool(abs(float(r["components"][:, 0] @ v)) > 0.9)
    True

    Components are orthonormal, since they come from a symmetric matrix.

    >>> r2 = dp_pca(X, k=2, epsilon=8.0, C=6.0, seed=0)
    >>> bool(abs(float(r2["components"][:, 0] @ r2["components"][:, 1])) < 1e-8)
    True

    The eigengap is reported because a small one makes individual components
    meaningless however large the budget.

    >>> bool(r2["eigengap"] >= 0)
    True

    >>> dp_pca(X, k=99, epsilon=1.0)
    Traceback (most recent call last):
        ...
    ValueError: k must be between 1 and 4
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    p = X.shape[1]
    k = int(k)
    if not 1 <= k <= p:
        raise ValueError(f"k must be between 1 and {p}")
    cov = dp_covariance(X, C=C, epsilon=epsilon, delta=delta, seed=seed)
    vals, vecs = np.linalg.eigh(cov["release"])
    order = np.argsort(vals, kind="stable")[::-1]
    vals, vecs = vals[order], vecs[:, order]
    gap = float(vals[k - 1] - vals[k]) if k < p else float("inf")
    total = float(np.sum(np.clip(vals, 0, None)))
    return RichResult(
        title="DP PCA",
        summary_lines=[("epsilon", cov["epsilon"]), ("k", k),
                       ("eigengap", gap)],
        warnings=(list(cov.warnings)
                  + (["the eigengap is small relative to the leading eigenvalue; "
                      "individual components are unstable, though the subspace "
                      "they span may not be"]
                     if np.isfinite(gap) and vals[0] > 0 and gap < 0.05 * vals[0] else [])),
        payload={
            "components": vecs[:, :k], "eigenvalues": vals,
            "eigengap": gap,
            "explained_variance_ratio": (np.clip(vals[:k], 0, None) / total
                                         if total > 0 else np.full(k, np.nan)),
            "scores": X @ vecs[:, :k], "covariance": cov["release"],
            "epsilon": cov["epsilon"], "delta": cov["delta"],
            "method": "dp_pca",
        },
    )


def cheatsheet():
    return "dppca: post-processing is free, so downstream costs nothing; accuracy is set by the EIGENGAP"


# compact alias per ledger/NAMING.md
dppca = dp_pca
