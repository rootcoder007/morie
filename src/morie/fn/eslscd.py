# morie.fn -- function file (rootcoder007/morie)
"""Sparse principal components -- ESL Sec 14.5.5."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_sparse_pca"]


def esl_sparse_pca(X, k=2, lambda_=0.1, max_iter=500, tol=1e-8, center=True, scale=False):
    r"""Sparse PCA by L1-penalised alternating maximisation.

    Each component solves

    .. math::
        \max_{v} \; v^\top \Sigma v - \lambda \lVert v \rVert_1
        \quad\text{s.t.}\quad \lVert v \rVert_2 = 1,

    which is the penalised-matrix-decomposition form: iterate
    :math:`v \leftarrow S_\lambda(\Sigma v)`, soft-thresholding then
    renormalising, until the loading stabilises. Later components are found
    on the deflated covariance.

    Ordinary PCA loadings are almost never zero, so every component is a
    combination of all ``p`` variables and hard to interpret. The L1 penalty
    zeroes loadings outright, at a cost that must be stated plainly: sparse
    components are **not orthogonal** and their variances do not sum to the
    total, so "percent variance explained" is not additive here the way it is
    for PCA. The reported ``adjusted_variance`` uses the Zou-Hastie-Tibshirani
    correction that accounts for the correlation between components.

    Parameters
    ----------
    X : array-like
        Data ``(n, p)``.
    k : int
        Number of components, from 1 to ``min(n, p)``.
    lambda_ : float
        Soft-threshold level. Zero recovers ordinary PCA.
    max_iter, tol
        Convergence controls for the power iteration.
    center, scale
        Centre columns, and optionally scale to unit variance.

    Returns
    -------
    RichResult
        ``loadings`` ``(p, k)``, ``scores`` ``(n, k)``, ``sparsity`` (fraction
        of zero loadings), ``adjusted_variance``, ``explained``,
        ``n_iter``.

    References
    ----------
    Zou, H., Hastie, T., & Tibshirani, R. (2006). Sparse principal component
        analysis. *Journal of Computational and Graphical Statistics*, 15(2),
        265-286.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    With ``lambda_ = 0`` this reduces to ordinary PCA, so the leading loading
    matches the top eigenvector up to sign.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 6)) @ np.diag([5.0, 3.0, 1, 1, 1, 1])
    >>> v = esl_sparse_pca(X, k=1, lambda_=0.0)["loadings"][:, 0]
    >>> w = np.linalg.eigh(np.cov(X, rowvar=False))[1][:, -1]
    >>> bool(abs(abs(v @ w) - 1) < 1e-5)
    True

    Raising the penalty zeroes loadings outright, which is the point.

    >>> r = esl_sparse_pca(X, k=1, lambda_=2.0)
    >>> bool(r["sparsity"] > 0)
    True
    >>> bool(np.isclose(np.linalg.norm(r["loadings"][:, 0]), 1.0))
    True

    >>> esl_sparse_pca(X, k=0)
    Traceback (most recent call last):
        ...
    ValueError: k must be between 1 and min(n, p)
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    k = int(k)
    if not 1 <= k <= min(n, p):
        raise ValueError("k must be between 1 and min(n, p)")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")

    Z = X - X.mean(axis=0) if center else X.astype(float).copy()
    if scale:
        sd = Z.std(axis=0, ddof=1)
        Z = Z / np.where(sd > 0, sd, 1.0)
    S = np.cov(Z, rowvar=False).reshape(p, p)

    loadings = np.zeros((p, k))
    iters = []
    Sd = S.copy()
    for j in range(k):
        v = np.linalg.eigh(Sd)[1][:, -1]
        it = 0
        for it in range(1, max_iter + 1):
            t = Sd @ v
            t = np.sign(t) * np.maximum(np.abs(t) - lambda_, 0.0)
            nrm = np.linalg.norm(t)
            if nrm < 1e-12:
                break
            new = t / nrm
            if np.linalg.norm(new - v) < tol or np.linalg.norm(new + v) < tol:
                v = new
                break
            v = new
        if np.abs(v).sum() > 0 and v[np.argmax(np.abs(v))] < 0:
            v = -v
        loadings[:, j] = v
        iters.append(it)
        Sd = Sd - (v @ Sd @ v) * np.outer(v, v)

    scores = Z @ loadings
    # Zou-Hastie-Tibshirani adjusted variance: QR of the scores removes the
    # part each component shares with the earlier ones, since sparse
    # components are not orthogonal and raw variances would double-count.
    _, R = np.linalg.qr(scores)
    adj = np.diag(R) ** 2 / max(n - 1, 1)
    total = float(np.trace(S))
    return RichResult(
        title="Sparse PCA",
        summary_lines=[("n", n), ("p", p), ("k", k), ("lambda", float(lambda_)),
                       ("sparsity", float(np.mean(loadings == 0)))],
        payload={
            "loadings": loadings, "scores": scores,
            "sparsity": float(np.mean(loadings == 0)),
            "adjusted_variance": adj,
            "explained": adj / total if total > 0 else np.full(k, np.nan),
            "total_variance": total,
            "lambda_": float(lambda_), "n_iter": np.array(iters),
            "method": "esl_sparse_pca",
        },
    )


def cheatsheet():
    return "eslscd: L1 sparse PCA; components are NOT orthogonal, so use adjusted_variance not raw sums"


# compact alias per ledger/NAMING.md
eslsparsepca = esl_sparse_pca
