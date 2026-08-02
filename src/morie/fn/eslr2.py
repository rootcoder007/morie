# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coefficient of determination (ESL Ch 3.2)."""

from . import _array_core as np

from ._richresult import RichResult
from .eslrss import esl_residual_sum_squares
from .eslrss2 import esl_total_sum_squares

__all__ = ["esl_r_squared"]


def esl_r_squared(X, y, beta):
    """
    R^2 = 1 - RSS/TSS at the supplied beta.

    Delegates both sums to eslrss and eslrss2 so the three functions
    cannot drift apart. Because beta is arbitrary rather than the OLS
    fit, R^2 CAN be negative: a model worse than the mean has
    RSS > TSS. That is reported honestly instead of being clipped to
    zero, and a constant response leaves R^2 undefined (nan) rather
    than dividing by zero.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    beta : array-like, shape (p,)
        Coefficients to score.

    Returns
    -------
    result : dict
        Keys: estimate (R^2), rss, tss, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.2.

    Examples
    --------
    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> esl_r_squared(X, [1.0, 3.0, 5.0], [1.0, 2.0])["estimate"]
    1.0
    >>> esl_r_squared(X, [1.0, 3.0, 5.0], [3.0, 0.0])["estimate"]
    0.0
    >>> esl_r_squared(X, [1.0, 3.0, 5.0], [10.0, 0.0])["estimate"] < 0
    True
    >>> esl_r_squared(X, [2.0, 2.0, 2.0], [2.0, 0.0])["estimate"]
    nan
    """
    rss = esl_residual_sum_squares(X, y, beta)
    tss = esl_total_sum_squares(y)
    r2 = float("nan") if tss["is_degenerate"] else 1.0 - rss["estimate"] / tss["estimate"]
    return RichResult(payload={
        "estimate": r2, "rss": rss["estimate"], "tss": tss["estimate"],
        "n": rss["n"], "p": rss["p"],
        "method": "R^2 = 1 - RSS/TSS; negative when worse than the mean, nan for constant y"})


def cheatsheet():
    return "eslr2: 1 - RSS/TSS; may be negative (arbitrary beta), nan if y constant"
