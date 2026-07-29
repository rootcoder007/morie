# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bayes theorem on a discretised parameter grid."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_posterior"]


def wasserman_posterior(data, f, prior):
    """
    Grid posterior p(theta | x) = p(x | theta) p(theta) / p(x).

    The prior comes as (theta_grid, prior_density) evaluated on an
    increasing grid; the likelihood of the whole i.i.d. sample is
    computed in the log domain at each grid point, and the posterior
    is normalised by trapezoid quadrature (p(x) = the normalising
    integral, reported as ``evidence``). ``f = None`` means the
    N(theta, 1) model.

    Parameters
    ----------
    data : array-like
        Sample (non-empty).
    f : callable or None
        Density f(x, theta) vectorised over x; None = N(theta, 1).
    prior : tuple (theta_grid, prior_density)
        Grid (increasing, >= 2 points) and non-negative density.

    Returns
    -------
    result : dict
        Keys: estimate (posterior mean), posterior, theta_grid,
        evidence, map_theta, n, method.

    References
    ----------
    Wasserman (2004), Ch 11, section 11.2.

    Examples
    --------
    N(theta, 1) with flat prior: posterior is N(xbar, 1/n) — mean
    equals the sample mean.

    >>> import numpy as np
    >>> grid = np.linspace(-5.0, 5.0, 4001)
    >>> out = wasserman_posterior([1.0, 0.5, 1.5], None, (grid, np.ones_like(grid)))
    >>> abs(out["estimate"] - 1.0) < 1e-9
    True
    >>> abs(out["map_theta"] - 1.0) < 0.005
    True
    >>> bool(abs(np.trapezoid(out["posterior"], grid) - 1.0) < 1e-9)
    True
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    if data.size == 0:
        raise ValueError("a posterior needs data.")
    grid, pd = prior
    grid = np.atleast_1d(np.asarray(grid, dtype=float))
    pd = np.atleast_1d(np.asarray(pd, dtype=float))
    if grid.size != pd.size or grid.size < 2:
        raise ValueError("the prior needs matching grid/density arrays with >= 2 points.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("the parameter grid must be strictly increasing.")
    if np.any(pd < 0):
        raise ValueError("a prior density cannot be negative.")
    if f is None:
        f = lambda x, th: np.exp(-0.5 * (x - th) ** 2) / np.sqrt(2.0 * np.pi)
    with np.errstate(divide="ignore"):
        ll = np.array([float(np.sum(np.log(np.asarray(f(data, th), dtype=float))))
                       for th in grid])
        logpost = ll + np.log(pd)
    m = np.max(logpost[np.isfinite(logpost)])
    unnorm = np.where(np.isfinite(logpost), np.exp(logpost - m), 0.0)
    dx = np.diff(grid)
    Z = float(0.5 * np.sum(dx * (unnorm[1:] + unnorm[:-1])))
    if Z <= 0:
        raise ValueError("the posterior normalising integral is zero; prior and likelihood do not overlap.")
    post = unnorm / Z
    tp = grid * post
    pmean = float(0.5 * np.sum(dx * (tp[1:] + tp[:-1])))
    return RichResult(payload={
        "estimate": pmean, "posterior": [float(v) for v in post],
        "theta_grid": [float(v) for v in grid],
        "evidence": float(Z * np.exp(m)),
        "map_theta": float(grid[int(np.argmax(post))]),
        "n": int(data.size),
        "method": "grid posterior, log-domain likelihood, trapezoid normalisation"})


def cheatsheet():
    return "wsmbay: post = exp(ll + log prior - max)/Z on grid; evidence = Z e^max"
