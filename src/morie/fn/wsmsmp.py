# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoothing spline (discrete second-difference penalty)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_smoothing_spline"]


def wasserman_smoothing_spline(x, y, lambda_):
    """
    Penalised regression smoother at the observation sites.

    Formula: min_m sum_i (Y_i - m(X_i))^2 + lambda int m''(t)^2 dt.
    Implemented in its standard discrete form on the sorted design
    points: m_hat = (I + lambda D'D)^{-1} y with D the second-
    difference operator scaled by the (possibly uneven) spacings —
    the Whittaker-Henderson / discrete-spline estimator whose limits
    are interpolation (lambda -> 0) and the least-squares LINE
    (lambda -> inf, since D annihilates linear trends).

    Parameters
    ----------
    x : array-like
        Design points, strictly increasing, n >= 3.
    y : array-like
        Responses, same length.
    lambda_ : float
        Roughness penalty, >= 0.

    Returns
    -------
    result : dict
        Keys: estimate (fitted values at x, in x order),
        effective_df, rss, lambda, n, method.

    References
    ----------
    Wasserman (2004), Ch 20, section 20.5 (penalised regression);
    Whittaker (1923).

    Examples
    --------
    lambda = 0 interpolates; huge lambda flattens a line exactly
    through linear data (D annihilates it, so any lambda keeps it):

    >>> x = [0.0, 1.0, 2.0, 3.0]
    >>> y = [1.0, 3.0, 5.0, 7.0]
    >>> out = wasserman_smoothing_spline(x, y, 1e8)
    >>> [round(v, 6) for v in out["estimate"]]
    [1.0, 3.0, 5.0, 7.0]
    >>> bumpy = [0.0, 2.0, 0.0, 2.0]
    >>> sm = wasserman_smoothing_spline(x, bumpy, 1e10)["estimate"]
    >>> [round(v, 4) for v in sm]     # the OLS line through the data
    [0.4, 0.8, 1.2, 1.6]
    >>> wasserman_smoothing_spline(x, bumpy, 0.0)["estimate"]
    [0.0, 2.0, 0.0, 2.0]
    >>> round(wasserman_smoothing_spline(x, y, 0.0)["effective_df"], 10)
    4.0
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
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        h1 = x[i + 1] - x[i]
        h2 = x[i + 2] - x[i + 1]
        D[i, i] = 2.0 / (h1 * (h1 + h2))
        D[i, i + 1] = -2.0 / (h1 * h2)
        D[i, i + 2] = 2.0 / (h2 * (h1 + h2))
    S = np.linalg.inv(np.eye(n) + lam * D.T @ D)
    fit = S @ y
    resid = y - fit
    return RichResult(payload={
        "estimate": [float(v) for v in fit],
        "effective_df": float(np.trace(S)), "rss": float(resid @ resid),
        "lambda": lam, "n": int(n),
        "method": "discrete smoothing spline (I + lam D'D)^-1 y, uneven spacings"})


def cheatsheet():
    return "wsmsmp: Whittaker smoother; lam=0 interpolates, lam=inf -> line"
