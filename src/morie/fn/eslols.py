# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OLS via the normal equations (ESL Ch 3.2)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_ols_normal_equations"]


def esl_ols_normal_equations(X, y):
    """
    Least squares beta_hat = (X'X)^{-1} X'y.

    ESL Eq. 3.6 is written with the explicit inverse and that is the
    formula named here, but the solve is done by QR: the normal
    equations square the condition number, so forming (X'X)^{-1}
    loses roughly twice the digits for no benefit. A rank-deficient
    design is refused rather than pseudo-inverted, since with
    collinear columns beta is not unique and any single answer would
    be arbitrary.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix with n > p; include your own intercept.
    y : array-like, shape (n,)
        Response.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, se, sigma2, rss,
        df_residual, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.2 (Eq. 3.6, 3.8).

    Examples
    --------
    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    >>> out = esl_ols_normal_equations(X, [1.0, 3.0, 5.0, 7.0])
    >>> [round(b, 12) for b in out["beta"]]
    [1.0, 2.0]
    >>> round(out["rss"], 12)
    0.0
    >>> out["df_residual"]
    2
    >>> esl_ols_normal_equations([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0])
    Traceback (most recent call last):
        ...
    ValueError: the design matrix is rank deficient (rank 1 < p = 2); beta is not unique.
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
        raise ValueError(f"the design matrix is rank deficient (rank {rank} < p = {p}); "
                         "beta is not unique.")
    resid = y - X @ beta
    rss = float(resid @ resid)
    dfr = n - p
    sigma2 = rss / dfr
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "se": [float(v) for v in se], "sigma2": float(sigma2), "rss": rss,
        "df_residual": int(dfr), "n": int(n), "p": int(p),
        "method": "OLS (Eq. 3.6) solved by QR; rank-deficient designs refused"})


def cheatsheet():
    return "eslols: beta = (X'X)^-1 X'y stated, QR-solved; se from sigma2 (X'X)^-1"
