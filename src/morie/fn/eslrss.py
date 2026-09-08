# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Residual sum of squares (ESL Ch 3.2)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_residual_sum_squares"]


def esl_residual_sum_squares(X, y, beta):
    """
    RSS(beta) = sum_i (y_i - x_i' beta)^2.

    Evaluated at the SUPPLIED beta, which need not be the least
    squares solution -- that is the point of the function: it scores
    an arbitrary coefficient vector, so ridge, lasso and OLS fits can
    be compared on the same scale.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix (include your own intercept column).
    y : array-like, shape (n,)
        Response.
    beta : array-like, shape (p,)
        Coefficient vector to score.

    Returns
    -------
    result : dict
        Keys: estimate (RSS), residuals, mean_squared_error, n, p,
        method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.2 (Eq. 3.2).

    Examples
    --------
    An exact fit has RSS 0; a unit shift of the intercept costs n.

    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> esl_residual_sum_squares(X, [1.0, 3.0, 5.0], [1.0, 2.0])["estimate"]
    0.0
    >>> esl_residual_sum_squares(X, [1.0, 3.0, 5.0], [0.0, 2.0])["estimate"]
    3.0
    >>> esl_residual_sum_squares(X, [1.0, 3.0, 5.0], [1.0])
    Traceback (most recent call last):
        ...
    ValueError: X has 2 columns but beta has 1 entries.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if beta.size != p:
        raise ValueError(f"X has {p} columns but beta has {beta.size} entries.")
    resid = y - X @ beta
    rss = float(resid @ resid)
    return RichResult(payload={
        "estimate": rss, "residuals": [float(v) for v in resid],
        "mean_squared_error": rss / n, "n": int(n), "p": int(p),
        "method": "RSS(beta) = sum (y_i - x_i' beta)^2 at the supplied beta"})


def cheatsheet():
    return "eslrss: RSS at an ARBITRARY beta, not necessarily the OLS fit"
