# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Expectation E[X] = int x f(x) dx."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_expectation"]


def wasserman_expectation(x, f):
    """
    Expectation E[X] over a discretised density: E[X] ~ sum x f(x) dx.

    Formula: E[X] = int x f(x) dx, evaluated by the trapezoidal rule
    on the grid ``x`` with density values ``f``. The density's own
    integral is reported so a badly normalised f is visible; it is
    NOT silently renormalised.

    Parameters
    ----------
    x : array-like
        Increasing grid of support points (>= 2 points).
    f : array-like
        Non-negative density values on the grid, same length.

    Returns
    -------
    result : dict
        Keys: estimate (E[X]), density_mass (int f dx), n, method.

    References
    ----------
    Wasserman (2004), Ch 3, Definition 3.1.

    Examples
    --------
    Uniform(0,1) on a fine grid has mean 1/2:

    >>> import numpy as np
    >>> g = np.linspace(0.0, 1.0, 100001)
    >>> out = wasserman_expectation(g, np.ones_like(g))
    >>> round(out["estimate"], 10)
    0.5
    >>> round(out["density_mass"], 10)
    1.0
    >>> wasserman_expectation([0.0], [1.0])
    Traceback (most recent call last):
        ...
    ValueError: the grid needs at least 2 points for a trapezoid rule.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    f = np.atleast_1d(np.asarray(f, dtype=float))
    if x.size < 2:
        raise ValueError("the grid needs at least 2 points for a trapezoid rule.")
    if x.size != f.size:
        raise ValueError(f"grid ({x.size}) and density ({f.size}) lengths differ.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("the grid must be strictly increasing.")
    if np.any(f < 0):
        raise ValueError("a density cannot be negative.")
    dx = np.diff(x)
    xf = x * f
    est = float(0.5 * np.sum(dx * (xf[1:] + xf[:-1])))
    mass = float(0.5 * np.sum(dx * (f[1:] + f[:-1])))
    return RichResult(payload={
        "estimate": est, "density_mass": mass, "n": int(x.size),
        "method": "E[X] = int x f(x) dx (trapezoid)"})


def cheatsheet():
    return "wsmexp: E[X] = int x f(x) dx by trapezoid; density mass reported"
