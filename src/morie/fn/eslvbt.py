# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sampling covariance of the OLS estimator (ESL Ch 3.2)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_var_beta_hat"]


def esl_var_beta_hat(X, sigma2):
    """
    Var(beta_hat) = sigma^2 (X'X)^{-1}.

    Depends only on the design and the noise level, not on y -- which
    is why an experiment's precision can be planned before any data
    are collected. The matrix comes back ROW-MAJOR; the diagonal
    (variances) and their square roots (standard errors) ship
    separately so callers need not re-derive the shape.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix of full column rank.
    sigma2 : float
        Noise variance, > 0.

    Returns
    -------
    result : dict
        Keys: estimate (se of the first coefficient), covariance
        (row-major p x p), variances, se, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.2 (Eq. 3.8).

    Examples
    --------
    Orthogonal design with n = 4, sigma^2 = 1: each variance is 1/4.

    >>> X = [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]]
    >>> out = esl_var_beta_hat(X, 1.0)
    >>> [round(v, 12) for v in out["variances"]]
    [0.25, 0.25]
    >>> round(out["estimate"], 12)
    0.5
    >>> esl_var_beta_hat(X, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: the noise variance must be positive; got 0.0.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    sigma2 = float(sigma2)
    n, p = X.shape
    if sigma2 <= 0:
        raise ValueError(f"the noise variance must be positive; got {sigma2}.")
    G = X.T @ X
    if np.linalg.matrix_rank(G) < p:
        raise ValueError(f"X'X is singular (rank < {p}); the design is collinear.")
    V = sigma2 * np.linalg.inv(G)
    var = np.diag(V)
    return RichResult(payload={
        "estimate": float(np.sqrt(var[0])),
        "covariance": [float(v) for v in V.ravel()],
        "variances": [float(v) for v in var],
        "se": [float(v) for v in np.sqrt(var)],
        "n": int(n), "p": int(p),
        "method": "Var(beta_hat) = sigma^2 (X'X)^-1, row-major"})


def cheatsheet():
    return "eslvbt: sigma^2 (X'X)^-1; y-free, so usable for design planning"
