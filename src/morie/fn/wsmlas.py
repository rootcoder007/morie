# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lasso regression by coordinate descent."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_lasso"]


def wasserman_lasso(X, y, lambda_, max_iter=10000, tol=1e-12):
    """
    Lasso: argmin (1/2)|y - X beta|^2 + lambda |beta|_1.

    Solved by cyclic coordinate descent with the exact
    soft-threshold update
    beta_j = S(x_j'(y - X_{-j} beta_{-j}), lambda) / (x_j' x_j),
    S(z, t) = sign(z) max(|z| - t, 0). Note the objective uses the
    UNSCALED (1/2) RSS, so this lambda is n times glmnet's. The
    active set (nonzero pattern) and objective value ship in the
    payload; convergence is by max coordinate change.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix (no all-zero columns).
    y : array-like, shape (n,)
        Response.
    lambda_ : float
        Penalty, >= 0.
    max_iter, tol
        Coordinate-descent controls.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, n_nonzero,
        objective, iterations, converged, lambda, n, p, method.

    References
    ----------
    Wasserman (2004), Ch 13; Tibshirani (1996).

    Examples
    --------
    Orthogonal design: lasso soft-thresholds the OLS solution.

    >>> X = [[1.0, 0.0], [0.0, 1.0]]
    >>> y = [3.0, -1.0]
    >>> out = wasserman_lasso(X, y, 0.5)
    >>> [round(b, 12) for b in out["beta"]]
    [2.5, -0.5]
    >>> out["n_nonzero"]
    2
    >>> wasserman_lasso(X, y, 1.5)["beta"]
    [1.5, 0.0]
    >>> wasserman_lasso(X, y, 0.0)["beta"]
    [3.0, -1.0]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    lam = float(lambda_)
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if lam < 0:
        raise ValueError(f"the lasso penalty must be non-negative; got {lam}.")
    colsq = np.sum(X ** 2, axis=0)
    if np.any(colsq == 0):
        raise ValueError("an all-zero column cannot be penalised meaningfully.")
    beta = np.zeros(p)
    r = y.copy()
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        delta = 0.0
        for j in range(p):
            old = beta[j]
            rho = X[:, j] @ r + colsq[j] * old
            new = np.sign(rho) * max(abs(rho) - lam, 0.0) / colsq[j]
            if new != old:
                r += X[:, j] * (old - new)
                beta[j] = new
                delta = max(delta, abs(new - old))
        if delta < tol:
            converged = True
            break
    obj = 0.5 * float(r @ r) + lam * float(np.sum(np.abs(beta)))
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "n_nonzero": int(np.sum(beta != 0)), "objective": float(obj),
        "iterations": int(it), "converged": bool(converged),
        "lambda": lam, "n": int(n), "p": int(p),
        "method": "lasso cyclic coordinate descent, soft threshold"})


def cheatsheet():
    return "wsmlas: coord descent S(rho,lam)/||x_j||^2; lambda unscaled (n x glmnet's)"
