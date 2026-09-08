# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bayesian credible interval from a grid posterior."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_credible_interval"]


def wasserman_credible_interval(posterior, alpha):
    """
    Equal-tail 1-alpha credible interval from a grid posterior.

    Formula: (a, b) with P(theta <= a | x) = alpha/2 and
    P(theta >= b | x) = alpha/2, read off the trapezoid CDF of the
    (theta_grid, density) pair. The density is renormalised if its
    integral drifts from 1 by more than 1e-6 (drift reported).

    Parameters
    ----------
    posterior : tuple (theta_grid, density)
        Grid posterior as produced by wsmbay.
    alpha : float
        Level in (0, 1).

    Returns
    -------
    result : dict
        Keys: estimate (interval length), lower, upper, mass_drift,
        alpha, method.

    References
    ----------
    Wasserman (2004), Ch 11, section 11.3.

    Examples
    --------
    Uniform posterior on [0, 1], alpha = 0.1 -> (0.05, 0.95).

    >>> import numpy as np
    >>> g = np.linspace(0.0, 1.0, 100001)
    >>> out = wasserman_credible_interval((g, np.ones_like(g)), 0.10)
    >>> round(out["lower"], 6)
    0.05
    >>> round(out["upper"], 6)
    0.95
    >>> round(out["estimate"], 6)
    0.9
    >>> wasserman_credible_interval((g, np.ones_like(g)), 1.0)
    Traceback (most recent call last):
        ...
    ValueError: alpha must lie in (0, 1); got 1.0.
    """
    grid, dens = posterior
    grid = np.atleast_1d(np.asarray(grid, dtype=float))
    dens = np.atleast_1d(np.asarray(dens, dtype=float))
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")
    if grid.size != dens.size or grid.size < 2:
        raise ValueError("the posterior needs matching grid/density arrays with >= 2 points.")
    dx = np.diff(grid)
    seg = 0.5 * dx * (dens[1:] + dens[:-1])
    total = float(np.sum(seg))
    if total <= 0:
        raise ValueError("the posterior has zero mass.")
    drift = abs(total - 1.0)
    cdf = np.concatenate([[0.0], np.cumsum(seg)]) / total
    lo = float(np.interp(alpha / 2.0, cdf, grid))
    hi = float(np.interp(1.0 - alpha / 2.0, cdf, grid))
    return RichResult(payload={
        "estimate": float(hi - lo), "lower": lo, "upper": hi,
        "mass_drift": float(drift), "alpha": alpha,
        "method": "equal-tail credible interval via trapezoid CDF inversion"})


def cheatsheet():
    return "wsmbcr: invert trapezoid CDF at alpha/2, 1-alpha/2; drift reported"
