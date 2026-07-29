# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Least squares regression."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_least_squares"]


def wasserman_least_squares(X, y):
    """
    Ordinary least squares with classical standard errors.

    Formula: beta_hat = (X'X)^{-1} X'y;
    Cov(beta_hat) = sigma_hat^2 (X'X)^{-1} with
    sigma_hat^2 = RSS / (n - p). Solved by QR (lstsq), not the
    normal equations, for numerical stability; a rank-deficient
    design is refused rather than silently pseudo-inverted.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix (add your own intercept column).
    y : array-like, shape (n,)
        Response, with n > p.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, se, sigma2, rss,
        r_squared, n, p, method.

    References
    ----------
    Wasserman (2004), Ch 13, Theorem 13.4.

    Examples
    --------
    Exact line y = 1 + 2x has zero residuals:

    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> out = wasserman_least_squares(X, [1.0, 3.0, 5.0])
    >>> [round(b, 12) for b in out["beta"]]
    [1.0, 2.0]
    >>> round(out["rss"], 12)
    0.0
    >>> out2 = wasserman_least_squares([[1.0], [1.0], [1.0], [1.0]], [1.0, 2.0, 3.0, 4.0])
    >>> out2["beta"]
    [2.5]
    >>> round(out2["se"][0], 15)
    0.645497224367903
    >>> wasserman_least_squares([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0])
    Traceback (most recent call last):
        ...
    ValueError: the design matrix is rank deficient (rank 1 < p = 2).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if n <= p:
        raise ValueError(f"OLS needs n > p; got n={n}, p={p}.")
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < p:
        raise ValueError(f"the design matrix is rank deficient (rank {rank} < p = {p}).")
    resid = y - X @ beta
    rss = float(resid @ resid)
    sigma2 = rss / (n - p)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "se": [float(v) for v in se], "sigma2": float(sigma2),
        "rss": rss, "r_squared": float(r2), "n": int(n), "p": int(p),
        "method": "OLS via QR; classical se sigma2 (X'X)^-1"})


def cheatsheet():
    return "wsmlsr: beta = lstsq(X,y); se = sqrt(diag(RSS/(n-p) (X'X)^-1))"
