# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ridge regression (ESL Ch 3.4.1)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_ridge"]


def esl_ridge(X, y, lambda_):
    """
    Ridge: beta_hat = (X'X + lambda I)^{-1} X'y.

    ESL penalises the SLOPES only, never the intercept, because
    shrinking the intercept would make the fit depend on the origin
    of y. This implementation follows that convention: pass the
    intercept as column 0 and it is left unpenalised (set
    ``penalize_intercept=True`` to override deliberately). Ridge is
    also not scale-invariant, so the payload reports whether the
    non-intercept columns are standardised -- unstandardised inputs
    silently penalise large-unit variables harder.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix; column 0 treated as the intercept if constant.
    y : array-like, shape (n,)
        Response.
    lambda_ : float
        Penalty, >= 0.
    penalize_intercept : bool
        Penalise every column including a constant one.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, effective_df, rss,
        lambda, intercept_penalised, columns_standardised, n, p,
        method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.4.1 (Eq. 3.44, 3.50).

    Examples
    --------
    lambda = 0 reproduces OLS; a large penalty shrinks the slope but
    leaves the intercept free to track the mean:

    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    >>> y = [1.0, 3.0, 5.0, 7.0]
    >>> [round(b, 10) for b in esl_ridge(X, y, 0.0)["beta"]]
    [1.0, 2.0]
    >>> heavy = esl_ridge(X, y, 1e8)["beta"]
    >>> abs(heavy[1]) < 1e-6
    True
    >>> round(heavy[0], 6)
    4.0
    >>> esl_ridge(X, y, -1.0)
    Traceback (most recent call last):
        ...
    ValueError: the ridge penalty must be non-negative; got -1.0.
    """
    return _ridge(X, y, lambda_, False)


def _ridge(X, y, lambda_, penalize_intercept):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    lam = float(lambda_)
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if lam < 0:
        raise ValueError(f"the ridge penalty must be non-negative; got {lam}.")
    const = np.array([bool(np.ptp(X[:, j]) == 0) for j in range(p)])
    P = np.eye(p)
    if not penalize_intercept:
        P[const, const] = 0.0
    G = X.T @ X + lam * P
    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        raise ValueError("X'X + lambda P is singular; increase lambda or fix the design.")
    beta = Ginv @ X.T @ y
    H = X @ Ginv @ X.T
    resid = y - X @ beta
    free = ~const
    if free.any():
        sd = X[:, free].std(axis=0)
        standardised = bool(np.allclose(sd, 1.0, atol=1e-8))
    else:
        standardised = True
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "effective_df": float(np.trace(H)), "rss": float(resid @ resid),
        "lambda": lam, "intercept_penalised": bool(penalize_intercept),
        "columns_standardised": standardised, "n": int(n), "p": int(p),
        "method": "ridge (X'X + lambda P)^-1 X'y; P excludes constant columns"})


def cheatsheet():
    return "eslrdg: intercept UNpenalised; not scale-invariant, standardisation reported"
