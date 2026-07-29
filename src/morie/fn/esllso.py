# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lasso regression (ESL Ch 3.4.2)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_lasso"]


def esl_lasso(X, y, lambda_, max_iter=10000, tol=1e-12):
    """
    Lasso: argmin (1/2)|y - X beta|^2 + lambda |beta|_1.

    Cyclic coordinate descent with the exact soft-threshold update
    S(rho, lambda)/||x_j||^2. As in ridge, a CONSTANT column is left
    unpenalised -- an L1 penalty on the intercept would shrink the
    fit toward y = 0 rather than toward the mean. Note the objective
    uses the unscaled (1/2)RSS, so this lambda is n times glmnet's;
    mixing the two conventions is the usual source of "the lasso
    gave me different coefficients".

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix with no all-zero column.
    y : array-like, shape (n,)
        Response.
    lambda_ : float
        Penalty, >= 0.
    max_iter, tol
        Coordinate-descent controls.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, n_nonzero, active_set
        (0-based), objective, iterations, converged, lambda, n, p,
        method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.4.2 (Eq. 3.51-3.52);
    Tibshirani (1996).

    Examples
    --------
    Orthonormal design: the lasso soft-thresholds the OLS solution,
    and the intercept is untouched by the penalty.

    >>> X = [[1.0, 0.0], [0.0, 1.0]]
    >>> [round(b, 12) for b in esl_lasso(X, [3.0, -1.0], 0.5)["beta"]]
    [2.5, -0.5]
    >>> esl_lasso(X, [3.0, -1.0], 1.5)["active_set"]
    [0]
    >>> Xi = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> b = esl_lasso(Xi, [1.0, 3.0, 5.0], 1e6)["beta"]
    >>> round(b[0], 6), abs(b[1]) < 1e-9
    (3.0, True)
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
    const = np.array([bool(np.ptp(X[:, j]) == 0) for j in range(p)])
    beta = np.zeros(p)
    r = y.copy()
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        delta = 0.0
        for j in range(p):
            old = beta[j]
            rho = X[:, j] @ r + colsq[j] * old
            pen = 0.0 if const[j] else lam
            new = np.sign(rho) * max(abs(rho) - pen, 0.0) / colsq[j]
            if new != old:
                r += X[:, j] * (old - new)
                beta[j] = new
                delta = max(delta, abs(new - old))
        if delta < tol:
            converged = True
            break
    active = [int(j) for j in np.flatnonzero(beta != 0)]
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "n_nonzero": len(active), "active_set": active,
        "objective": float(0.5 * (r @ r) + lam * np.sum(np.abs(beta[~const]))),
        "iterations": int(it), "converged": bool(converged), "lambda": lam,
        "n": int(n), "p": int(p),
        "method": "lasso coordinate descent; constant columns unpenalised; lambda is n x glmnet's"})


def cheatsheet():
    return "esllso: soft-threshold CD; intercept unpenalised; lambda scale != glmnet"
