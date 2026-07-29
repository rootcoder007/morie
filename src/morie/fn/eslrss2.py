# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Total sum of squares (ESL Ch 3.2)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_total_sum_squares"]


def esl_total_sum_squares(y):
    """
    TSS = sum_i (y_i - y_bar)^2.

    The baseline against which RSS is judged: the residual sum of
    squares of the intercept-only model. A constant response gives
    TSS 0, which makes R^2 undefined rather than 1 -- reported here
    as the ``is_degenerate`` flag so callers can branch instead of
    dividing by zero downstream.

    Parameters
    ----------
    y : array-like
        Response, at least one observation.

    Returns
    -------
    result : dict
        Keys: estimate (TSS), mean, n, is_degenerate, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.2.

    Examples
    --------
    >>> esl_total_sum_squares([1.0, 3.0, 5.0])["estimate"]
    8.0
    >>> out = esl_total_sum_squares([2.0, 2.0, 2.0])
    >>> out["estimate"]
    0.0
    >>> out["is_degenerate"]
    True
    >>> esl_total_sum_squares([])
    Traceback (most recent call last):
        ...
    ValueError: the total sum of squares needs at least one observation.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if y.size == 0:
        raise ValueError("the total sum of squares needs at least one observation.")
    mu = float(np.mean(y))
    tss = float(np.sum((y - mu) ** 2))
    return RichResult(payload={
        "estimate": tss, "mean": mu, "n": int(y.size),
        "is_degenerate": bool(tss == 0.0),
        "method": "TSS = sum (y_i - y_bar)^2"})


def cheatsheet():
    return "eslrss2: TSS about the mean; is_degenerate flags constant y (R^2 undefined)"
