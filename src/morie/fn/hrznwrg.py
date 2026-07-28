# morie.fn -- function file (rootcoder007/morie)
"""Index-model kernel regression."""

import numpy as np

from ._horowitz import kernel, silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_index_nw"]


def hrz_index_nw(X, y, beta, h=None, grid=None, kernel_name="gaussian"):
    r"""Nadaraya-Watson regression on a single index (Horowitz Ch. 2):

    .. math:: \hat G(v) = \frac{\sum_i K_h(X_i'\beta - v) Y_i}
                                {\sum_i K_h(X_i'\beta - v)}.

    Regressing on the scalar :math:`X'\beta` instead of the full
    vector X collapses a d-dimensional smoothing problem to one
    dimension, restoring the :math:`n^{-2/5}` rate no matter how large
    d is. That is the entire payoff of the single-index restriction.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Covariates.
    y : array-like, shape (n,)
        Response.
    beta : array-like, shape (d,)
        Index coefficients.
    h : float, optional
        Bandwidth on the index scale.
    grid : array-like, optional
        Index values at which to evaluate.
    kernel_name : str
        Kernel.

    Returns
    -------
    RichResult
        keys: ``index_grid``, ``G``, ``index``, ``bandwidth``,
        ``rate_exponent`` (-2/5, independent of d), ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (single-index models).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    b = np.atleast_1d(np.asarray(beta, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    if X.shape[0] != y.size:
        raise ValueError("X must have one row per entry of y.")
    if X.shape[1] != b.size:
        raise ValueError(f"beta must have {X.shape[1]} entries.")
    v = X @ b
    h = silverman_bw(v) if h is None else float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.linspace(v.min(), v.max(), 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    W = kernel((g[:, None] - v[None, :]) / h, kernel_name)
    den = W.sum(axis=1)
    with np.errstate(invalid="ignore"):
        G = np.where(den > 0, (W @ y) / np.maximum(den, 1e-300), np.nan)
    return RichResult(payload={"index_grid": g, "G": G, "index": v,
                               "bandwidth": h, "rate_exponent": -0.4,
                               "d": int(X.shape[1]),
                               "method": "NW on X'beta; n^{-2/5} rate regardless of d"})


def cheatsheet():
    return "hrznwrg: index collapses d dimensions to 1, restoring the n^{-2/5} rate"
