# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standard errors of fitted coefficients (ESL Ch 3.2)."""

from . import _array_core as np

from ._richresult import RichResult
from .eslrss import esl_residual_sum_squares
from .eslvbt import esl_var_beta_hat

__all__ = ["esl_se_beta"]


def esl_se_beta(X, y, beta):
    """
    se(beta_j) = sqrt(sigma_hat^2 v_jj), v = (X'X)^{-1}.

    Unlike eslvbt, sigma^2 is ESTIMATED from the residuals at the
    supplied beta with the unbiased divisor n - p (ESL Eq. 3.8), so
    these are the standard errors actually reported with a fit.
    Passing a non-OLS beta inflates sigma_hat^2 honestly rather than
    silently re-fitting.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix with n > p.
    y : array-like, shape (n,)
        Response.
    beta : array-like, shape (p,)
        Coefficients whose residuals estimate sigma^2.

    Returns
    -------
    result : dict
        Keys: estimate (se of the first coefficient), se, sigma2_hat,
        rss, df_residual, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.2 (Eq. 3.8).

    Examples
    --------
    >>> X = [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]]
    >>> y = [3.0, -1.0, 3.0, -1.0]
    >>> out = esl_se_beta(X, y, [1.0, 2.0])
    >>> out["sigma2_hat"]
    0.0
    >>> out["se"]
    [0.0, 0.0]
    >>> worse = esl_se_beta(X, y, [0.0, 2.0])
    >>> round(worse["sigma2_hat"], 12)
    2.0
    >>> esl_se_beta(X, y, [1.0, 2.0, 3.0])
    Traceback (most recent call last):
        ...
    ValueError: X has 2 columns but beta has 3 entries.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    if n <= p:
        raise ValueError(f"estimating sigma^2 needs n > p; got n={n}, p={p}.")
    rss = esl_residual_sum_squares(X, y, beta)
    dfr = n - p
    sigma2 = rss["estimate"] / dfr
    if sigma2 == 0:
        se = [0.0] * p
    else:
        se = esl_var_beta_hat(X, sigma2)["se"]
    return RichResult(payload={
        "estimate": float(se[0]), "se": [float(v) for v in se],
        "sigma2_hat": float(sigma2), "rss": rss["estimate"],
        "df_residual": int(dfr), "n": int(n), "p": int(p),
        "method": "se(beta_j) = sqrt(sigma_hat^2 v_jj), sigma_hat^2 = RSS/(n-p)"})


def cheatsheet():
    return "eslsbt: sigma_hat^2 = RSS/(n-p) at the SUPPLIED beta, then sqrt(diag)"
