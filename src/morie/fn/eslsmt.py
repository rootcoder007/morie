# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoothing spline (ESL Ch 5.4)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_smoothing_spline"]


def esl_smoothing_spline(x, y, lambda_):
    """
    Smoothing spline:
    min sum (y_i - f(x_i))^2 + lambda int f''(t)^2 dt.

    ESL Ch 5.4 shows the minimiser over ALL twice-differentiable
    functions is a natural cubic spline with a knot at every unique
    x_i — an infinite-dimensional problem with a finite-dimensional
    answer. The fit is therefore linear in y, f_hat = S_lambda y,
    and the smoother matrix S gives the effective degrees of freedom
    as its trace.

    Implemented in the standard discrete form (I + lambda K)^-1 y with
    K = D' W D built from the second-difference operator on the
    unequally spaced design, which is the Reinsch algorithm's penalty.
    Its two limits are exact and both are checked in the doctests:
    lambda = 0 interpolates, and lambda -> infinity gives the least
    squares LINE, not a constant, because the penalty annihilates
    anything linear.

    Parameters
    ----------
    x : array-like
        Design points, strictly increasing, at least 3.
    y : array-like
        Responses.
    lambda_ : float
        Roughness penalty, >= 0.

    Returns
    -------
    result : dict
        Keys: estimate (fitted values), effective_df, rss, lambda,
        n, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 5.4 (Eq. 5.9, 5.17).

    Examples
    --------
    >>> x = [0.0, 1.0, 2.0, 3.0]
    >>> y = [0.0, 2.0, 0.0, 2.0]
    >>> zero = esl_smoothing_spline(x, y, 0.0)
    >>> [round(v, 10) for v in zero["estimate"]]
    [0.0, 2.0, 0.0, 2.0]
    >>> round(zero["effective_df"], 8)
    4.0

    Heavy penalty gives the OLS line through the data, and its
    effective degrees of freedom fall to 2 — an intercept and a slope:

    >>> stiff = esl_smoothing_spline(x, y, 1e10)
    >>> [round(v, 4) for v in stiff["estimate"]]
    [0.4, 0.8, 1.2, 1.6]
    >>> round(stiff["effective_df"], 4)
    2.0
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    lam = float(lambda_)
    n = x.size
    if y.size != n:
        raise ValueError(f"x ({n}) and y ({y.size}) lengths differ.")
    if n < 3:
        raise ValueError("a smoothing spline needs at least 3 points.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("the design points must be strictly increasing.")
    if lam < 0:
        raise ValueError(f"the penalty must be non-negative; got {lam}.")
    h = np.diff(x)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -(1.0 / h[i] + 1.0 / h[i + 1])
        D[i, i + 2] = 1.0 / h[i + 1]
    W = np.zeros((n - 2, n - 2))
    for i in range(n - 2):
        W[i, i] = (h[i] + h[i + 1]) / 3.0
        if i < n - 3:
            W[i, i + 1] = W[i + 1, i] = h[i + 1] / 6.0
    K = D.T @ W @ D
    S = np.linalg.inv(np.eye(n) + lam * K)
    fit = S @ y
    resid = y - fit
    return RichResult(payload={
        "estimate": [float(v) for v in fit],
        "effective_df": float(np.trace(S)), "rss": float(resid @ resid),
        "lambda": lam, "n": int(n),
        "method": "smoothing spline (I + lambda D'WD)^-1 y; df = tr(S)"})


def cheatsheet():
    return "eslsmt: knot at every x; lam=0 interpolates (df=n), lam=inf -> line (df=2)"
