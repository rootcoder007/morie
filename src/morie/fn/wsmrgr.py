# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ridge regression."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_ridge"]


def wasserman_ridge(X, y, lambda_):
    """
    Ridge regression.

    Formula: beta_hat = (X'X + lambda I)^{-1} X'y. lambda = 0
    reduces to OLS (allowed only when X'X is invertible); lambda < 0
    is refused. Effective degrees of freedom
    tr(X (X'X + lambda I)^{-1} X') come along — the model-size
    currency for ridge.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    lambda_ : float
        Penalty, >= 0.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, effective_df, rss,
        lambda, n, p, method.

    References
    ----------
    Wasserman (2004), Ch 13; Hoerl & Kennard (1970).

    Examples
    --------
    lambda -> 0 gives OLS; large lambda shrinks toward zero:

    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> y = [1.0, 3.0, 5.0]
    >>> [round(b, 10) for b in wasserman_ridge(X, y, 0.0)["beta"]]
    [1.0, 2.0]
    >>> b = wasserman_ridge(X, y, 1e6)["beta"]
    >>> abs(b[0]) < 1e-4 and abs(b[1]) < 1e-4
    True
    >>> round(wasserman_ridge(X, y, 0.0)["effective_df"], 12)
    2.0
    >>> wasserman_ridge(X, y, -1.0)
    Traceback (most recent call last):
        ...
    ValueError: the ridge penalty must be non-negative; got -1.0.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    lam = float(lambda_)
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if lam < 0:
        raise ValueError(f"the ridge penalty must be non-negative; got {lam}.")
    G = X.T @ X + lam * np.eye(p)
    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        raise ValueError("X'X + lambda I is singular; increase lambda or fix the design.")
    beta = Ginv @ X.T @ y
    H = X @ Ginv @ X.T
    resid = y - X @ beta
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "effective_df": float(np.trace(H)), "rss": float(resid @ resid),
        "lambda": lam, "n": int(n), "p": int(p),
        "method": "ridge (X'X + lambda I)^-1 X'y; edf = tr(H)"})


def cheatsheet():
    return "wsmrgr: closed-form ridge; effective df = tr(X G^-1 X')"
