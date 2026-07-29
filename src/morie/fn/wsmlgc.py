# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Log-linear model for a 2-way table."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_log_linear"]


def wasserman_log_linear(table):
    """
    Saturated log-linear decomposition of an I x J table.

    Formula: log mu_ij = lambda + lambda_i^A + lambda_j^B +
    lambda_ij^{AB}, identified by zero-SUM constraints
    (sum_i lambda_i^A = 0 etc.). Computed exactly from the observed
    counts (the saturated model fits perfectly): the lambdas are the
    two-way ANOVA decomposition of log n_ij. The independence
    (no-interaction) fitted counts mu_ij = n_i. n_.j / n and the
    G^2 likelihood-ratio statistic against independence come along.

    Parameters
    ----------
    table : array-like, shape (I, J)
        Strictly positive counts.

    Returns
    -------
    result : dict
        Keys: estimate (G^2 vs independence), lambda0, lambda_row,
        lambda_col, lambda_int (row-major), independence_fit
        (row-major), df, n, method.

    References
    ----------
    Wasserman (2004), Ch 17 (log-linear models).

    Examples
    --------
    An independent table has zero interaction and G^2 = 0:

    >>> out = wasserman_log_linear([[10.0, 20.0], [30.0, 60.0]])
    >>> [round(abs(v), 12) for v in out["lambda_int"]]
    [0.0, 0.0, 0.0, 0.0]
    >>> round(out["estimate"], 12)
    0.0
    >>> dep = wasserman_log_linear([[30, 10], [15, 45]])
    >>> dep["estimate"] > 20
    True
    >>> dep["df"]
    1
    >>> wasserman_log_linear([[1, 0], [2, 3]])
    Traceback (most recent call last):
        ...
    ValueError: the saturated log-linear model needs strictly positive counts.
    """
    T = np.atleast_2d(np.asarray(table, dtype=float))
    I, J = T.shape
    if I < 2 or J < 2:
        raise ValueError(f"a two-way table needs at least 2x2 cells; got {I}x{J}.")
    if np.any(T <= 0):
        raise ValueError("the saturated log-linear model needs strictly positive counts.")
    L = np.log(T)
    lam0 = float(np.mean(L))
    lr = np.mean(L, axis=1) - lam0
    lc = np.mean(L, axis=0) - lam0
    lint = L - lam0 - lr[:, None] - lc[None, :]
    n = float(np.sum(T))
    mu_ind = np.outer(T.sum(axis=1), T.sum(axis=0)) / n
    g2 = float(2.0 * np.sum(T * np.log(T / mu_ind)))
    return RichResult(payload={
        "estimate": g2, "lambda0": lam0,
        "lambda_row": [float(v) for v in lr],
        "lambda_col": [float(v) for v in lc],
        "lambda_int": [float(v) for v in lint.ravel()],
        "independence_fit": [float(v) for v in mu_ind.ravel()],
        "df": int((I - 1) * (J - 1)), "n": n,
        "method": "saturated log-linear (zero-sum ANOVA of log counts) + G^2"})


def cheatsheet():
    return "wsmlgc: lambdas = ANOVA of log n_ij; G^2 = 2 sum n log(n/mu_ind)"
