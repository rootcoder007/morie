# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fisher information I(theta)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_fisher_info"]


def wasserman_fisher_info(f, theta, x_grid=None, h=1e-5):
    """
    Fisher information by numeric expectation of the curvature.

    Formula: I(theta) = -E_theta[d^2 log f(X;theta) / d theta^2].
    The second derivative in theta is a central finite difference
    with step ``h``; the expectation integrates it against
    f(x; theta) by the trapezoid rule on ``x_grid``. ``f = None``
    means the exponential model e^{-x/theta}/theta, whose exact
    information 1/theta^2 anchors the numerics; its default grid
    covers [0, 40 theta].

    Parameters
    ----------
    f : callable or None
        Density f(x, theta), vectorised in x; None = exponential.
    theta : float
        Parameter value (> 0 for the default model).
    x_grid : array-like, optional
        Support grid for the expectation. REQUIRED for a custom f —
        the function cannot guess an arbitrary model's support.
    h : float
        Finite-difference step in theta.

    Returns
    -------
    result : dict
        Keys: estimate (I), se_one_obs (1/sqrt(I)), theta, h,
        grid_points, method.

    References
    ----------
    Wasserman (2004), Ch 9, Definition 9.26.

    Examples
    --------
    Exponential: I(theta) = 1/theta^2.

    >>> out = wasserman_fisher_info(None, 2.0)
    >>> abs(out["estimate"] - 0.25) < 1e-4
    True
    >>> out2 = wasserman_fisher_info(None, 1.0)
    >>> abs(out2["estimate"] - 1.0) < 1e-4
    True
    >>> def g(x, th): return None
    >>> wasserman_fisher_info(g, 1.0)
    Traceback (most recent call last):
        ...
    ValueError: a custom density needs an explicit x_grid for the expectation.
    """
    theta = float(theta)
    if f is None:
        if theta <= 0:
            raise ValueError(f"the exponential model needs theta > 0; got {theta}.")
        f = lambda x, th: np.where(x >= 0, np.exp(-x / th) / th, 0.0)
        if x_grid is None:
            x_grid = np.linspace(0.0, 40.0 * theta, 200001)
    if x_grid is None:
        raise ValueError("a custom density needs an explicit x_grid for the expectation.")
    x = np.atleast_1d(np.asarray(x_grid, dtype=float))
    with np.errstate(divide="ignore"):
        lp = np.log(np.asarray(f(x, theta + h), dtype=float))
        l0 = np.log(np.asarray(f(x, theta), dtype=float))
        lm = np.log(np.asarray(f(x, theta - h), dtype=float))
    d2 = (lp - 2.0 * l0 + lm) / (h * h)
    w = np.asarray(f(x, theta), dtype=float)
    good = np.isfinite(d2) & np.isfinite(w) & (w > 0)
    xg, integ = x[good], -d2[good] * w[good]
    dx = np.diff(xg)
    info = float(0.5 * np.sum(dx * (integ[1:] + integ[:-1])))
    if info <= 0:
        raise ValueError(f"numeric information came out non-positive ({info}); check the model/grid.")
    return RichResult(payload={
        "estimate": info, "se_one_obs": float(info ** -0.5),
        "theta": theta, "h": float(h), "grid_points": int(x.size),
        "method": "I = -int f(x;th) d2 log f/dth2 dx (central diff + trapezoid)"})


def cheatsheet():
    return "wsmfis: I(theta) = -E[d2 log f/dtheta2]; exponential default anchors 1/theta^2"
