# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Posterior mean from a grid posterior."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_posterior_mean"]


def wasserman_posterior_mean(posterior):
    """
    Posterior mean theta_bayes = int theta p(theta | x) d theta.

    Trapezoid quadrature over the (theta_grid, density) pair, with
    the density renormalised by its own trapezoid mass so a slightly
    unnormalised posterior still yields the correct mean; the drift
    is reported. The posterior second moment and sd come along.

    Parameters
    ----------
    posterior : tuple (theta_grid, density)
        Grid posterior as produced by wsmbay.

    Returns
    -------
    result : dict
        Keys: estimate (posterior mean), posterior_sd, mass_drift,
        method.

    References
    ----------
    Wasserman (2004), Ch 11, section 11.2.

    Examples
    --------
    Uniform on [0, 1]: mean 1/2, sd 1/sqrt(12).

    >>> import numpy as np
    >>> g = np.linspace(0.0, 1.0, 100001)
    >>> out = wasserman_posterior_mean((g, np.ones_like(g)))
    >>> round(out["estimate"], 9)
    0.5
    >>> abs(out["posterior_sd"] - 1.0 / 12.0 ** 0.5) < 1e-9
    True
    >>> round(out["mass_drift"], 12)
    0.0
    """
    grid, dens = posterior
    grid = np.atleast_1d(np.asarray(grid, dtype=float))
    dens = np.atleast_1d(np.asarray(dens, dtype=float))
    if grid.size != dens.size or grid.size < 2:
        raise ValueError("the posterior needs matching grid/density arrays with >= 2 points.")
    dx = np.diff(grid)
    def _quad(y):
        return float(0.5 * np.sum(dx * (y[1:] + y[:-1])))
    Z = _quad(dens)
    if Z <= 0:
        raise ValueError("the posterior has zero mass.")
    m1 = _quad(grid * dens) / Z
    m2 = _quad(grid ** 2 * dens) / Z
    var = max(m2 - m1 ** 2, 0.0)
    return RichResult(payload={
        "estimate": float(m1), "posterior_sd": float(var ** 0.5),
        "mass_drift": float(abs(Z - 1.0)),
        "method": "posterior mean by trapezoid quadrature (self-normalising)"})


def cheatsheet():
    return "wsmpst1: E[theta|x] = quad(theta p)/quad(p); sd + drift alongside"
