# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""White-Huber sandwich covariance."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_white_huber"]


def wasserman_white_huber(X, y, f=None):
    """
    Heteroskedasticity-robust (White-Huber) sandwich covariance for
    least squares.

    Formula: V = A^{-1} B A^{-1} with A = (1/n) X'X and
    B = (1/n) sum_i e_i^2 x_i x_i', where e are residuals from the
    fit y ~ X. This is HC0 scaled for theta_hat's covariance:
    Cov(beta_hat) = A^{-1} B A^{-1} / n. ``f`` optionally maps
    (X, beta) to fitted values for a nonlinear mean; None = linear
    X beta with beta the OLS solution.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix (include the intercept column yourself).
    y : array-like, shape (n,)
        Response.
    f : callable or None
        Mean function f(X, beta); None = linear.

    Returns
    -------
    result : dict
        Keys: estimate (robust se of the first coefficient), beta,
        robust_se, covariance (row-major flat), bread (A, flat),
        meat (B, flat), n, p, method.

    References
    ----------
    Wasserman (2004), Ch 9 (sandwich estimator); White (1980).

    Examples
    --------
    Homoskedastic equal-leverage design: sandwich matches classical.

    >>> import numpy as np
    >>> X = np.array([[1.0, -1.0], [1.0, 1.0]] * 50)
    >>> beta = np.array([1.0, 2.0])
    >>> e = np.tile([0.5, -0.5, -0.5, 0.5], 25)
    >>> y = X @ beta + e
    >>> out = wasserman_white_huber(X, y)
    >>> [round(b, 12) for b in out["beta"]]
    [1.0, 2.0]
    >>> round(out["robust_se"][0], 12) == round(0.5 / 10.0, 12)
    True
    >>> out["n"], out["p"]
    (100, 2)
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if n <= p:
        raise ValueError(f"sandwich needs n > p; got n={n}, p={p}.")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta if f is None else np.asarray(f(X, beta), dtype=float)
    e = y - fitted
    A = X.T @ X / n
    B = (X * (e ** 2)[:, None]).T @ X / n
    Ainv = np.linalg.inv(A)
    V = Ainv @ B @ Ainv / n
    rse = np.sqrt(np.diag(V))
    return RichResult(payload={
        "estimate": float(rse[0]),
        "beta": [float(v) for v in beta],
        "robust_se": [float(v) for v in rse],
        "covariance": [float(v) for v in V.ravel()],
        "bread": [float(v) for v in A.ravel()],
        "meat": [float(v) for v in B.ravel()],
        "n": int(n), "p": int(p),
        "method": "White-Huber HC0 sandwich A^-1 B A^-1 / n"})


def cheatsheet():
    return "wsmwhz: Cov = A^-1 B A^-1/n, A = X'X/n, B = X'diag(e^2)X/n; matrices row-major"
