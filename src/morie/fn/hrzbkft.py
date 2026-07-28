# morie.fn -- function file (rootcoder007/morie)
"""Backfitting for additive models."""

import numpy as np

from ._horowitz import local_linear, silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_backfitting"]


def hrz_backfitting(X, y, h=None, max_iter=50, tol=1e-6, kernel_name="gaussian"):
    r"""Backfitting for additive models (Horowitz Ch. 2):

    iterate :math:`g_j \leftarrow S_j\big(Y - \mu -
    \sum_{k \ne j} g_k\big)` until convergence, with each
    :math:`S_j` a one-dimensional smoother.

    The additive restriction :math:`E[Y|X] = \mu + \sum_j g_j(X_j)`
    buys back the one-dimensional rate :math:`n^{-2/5}` in ANY
    dimension -- the escape from the curse that
    :mod:`morie.fn.hrzkd2` quantifies. Each component is centred at
    every sweep, since only the sum is identified: shifting a constant
    between components leaves the fit unchanged.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Covariates.
    y : array-like, shape (n,)
        Response.
    h : float, optional
        Common bandwidth.
    max_iter : int, default 50
        Sweeps.
    tol : float, default 1e-6
        Convergence tolerance.
    kernel_name : str
        Kernel.

    Returns
    -------
    RichResult
        keys: ``mu``, ``components`` (n, d), ``fitted``,
        ``n_iter``, ``converged``, ``rate_exponent`` (-2/5),
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (additive models and backfitting).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        X = X.T
    if X.shape[0] != y.size:
        raise ValueError("X must have one row per entry of y.")
    n, d = X.shape
    mu = float(y.mean())
    G = np.zeros((n, d))
    conv = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        prev = G.copy()
        for j in range(d):
            partial = y - mu - (G.sum(axis=1) - G[:, j])
            hj = silverman_bw(X[:, j]) if h is None else float(h)
            _, gj, _, _ = local_linear(X[:, j], partial, grid=X[:, j], h=hj,
                                       name=kernel_name)
            gj = np.where(np.isfinite(gj), gj, 0.0)
            G[:, j] = gj - gj.mean()  # only the sum is identified
        if np.max(np.abs(G - prev)) < tol:
            conv = True
            break
    return RichResult(payload={"mu": mu, "components": G,
                               "fitted": mu + G.sum(axis=1), "n_iter": it,
                               "converged": conv, "rate_exponent": -0.4,
                               "d": int(d), "n": int(n),
                               "method": "Backfitting; additivity restores n^{-2/5} in any d"})


def cheatsheet():
    return "hrzbkft: components centred each sweep -- only their SUM is identified"
