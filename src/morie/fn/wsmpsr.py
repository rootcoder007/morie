# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Poisson regression by Newton-Raphson."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_poisson_regression"]


def wasserman_poisson_regression(X, y, max_iter=100, tol=1e-10):
    """
    Poisson log-linear regression MLE.

    Formula: log mu = X beta, Y ~ Poisson(mu). Newton update
    beta <- beta + (X'WX)^{-1} X'(y - mu) with W = diag(mu).
    Standard errors from the inverse information at convergence.
    The log-likelihood INCLUDES the -sum log(y_i!) constant (via
    lgamma), so values are true log-likelihoods comparable across
    models.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Non-negative integer counts.
    max_iter, tol
        Newton controls.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, se,
        log_likelihood, deviance, iterations, converged, n, p,
        method.

    References
    ----------
    Wasserman (2004), Ch 13 (GLMs).

    Examples
    --------
    Intercept-only: beta0 = log(ybar).

    >>> import math
    >>> out = wasserman_poisson_regression([[1.0]] * 4, [1, 2, 3, 2])
    >>> abs(out["beta"][0] - math.log(2.0)) < 1e-10
    True
    >>> abs(out["se"][0] - 1.0 / math.sqrt(8.0)) < 1e-10
    True
    >>> wasserman_poisson_regression([[1.0]] * 2, [1.0, -2.0])
    Traceback (most recent call last):
        ...
    ValueError: Poisson counts must be non-negative integers.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if np.any(y < 0) or np.any(y != np.round(y)):
        raise ValueError("Poisson counts must be non-negative integers.")
    from math import lgamma
    beta = np.zeros(p)
    beta[0] = np.log(np.mean(y)) if np.mean(y) > 0 else 0.0
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        mu = np.exp(np.clip(X @ beta, -30, 30))
        H = X.T @ (X * mu[:, None])
        try:
            step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            raise ValueError("the information matrix is singular; check the design.")
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    mu = np.exp(X @ beta)
    ll = float(np.sum(y * np.log(mu) - mu) - sum(lgamma(v + 1.0) for v in y))
    with np.errstate(divide="ignore", invalid="ignore"):
        dev_terms = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    deviance = float(2.0 * np.sum(dev_terms))
    cov = np.linalg.inv(X.T @ (X * mu[:, None]))
    se = np.sqrt(np.diag(cov))
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "se": [float(v) for v in se], "log_likelihood": ll,
        "deviance": deviance, "iterations": int(it),
        "converged": bool(converged), "n": int(n), "p": int(p),
        "method": "Poisson GLM Newton; ll includes lgamma constant"})


def cheatsheet():
    return "wsmpsr: log mu = X beta; W = diag(mu); deviance + full ll"
