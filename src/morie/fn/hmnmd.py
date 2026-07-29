# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Numerical differentiation via finite differences."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_numerical_diff"]


def geron_numerical_diff(f, x, h=1e-5):
    """
    Numerical differentiation via finite differences.

    Formula: df/dx approx (f(x+h) - f(x-h)) / (2h)

    The central difference is used, whose truncation error is O(h^2)
    against the O(h) of the forward difference. For a vector ``x`` the
    full gradient is returned, one perturbed pair per coordinate, which
    is the 2n function evaluations that make finite differences useless
    for neural networks and perfect for CHECKING an analytic gradient.
    A Richardson extrapolation from a half-step gives an error estimate
    for free.

    Parameters
    ----------
    f : callable
        ``f(x) -> float``; must accept the same shape as ``x``.
    x : float or array-like
        Point of evaluation.
    h : float, default 1e-5
        Step size (positive).

    Returns
    -------
    result : RichResult
        Keys: derivative, richardson, error_estimate, n_evals, estimate,
        n, method.

    Examples
    --------
    d/dx x^3 at x = 2 is 12:

    >>> r = geron_numerical_diff(lambda x: x**3, 2.0)
    >>> round(float(r["derivative"]), 6)
    12.0

    The gradient of x0^2 + 3*x1 is (2*x0, 3):

    >>> g = geron_numerical_diff(lambda v: v[0]**2 + 3*v[1], [4.0, 1.0])["derivative"]
    >>> [round(float(v), 6) for v in g]
    [8.0, 3.0]
    >>> int(geron_numerical_diff(lambda v: v[0] + v[1], [0.0, 0.0])["n_evals"])
    8

    References
    ----------
    Geron Appendix A
    """
    if not callable(f):
        raise ValueError("geron_numerical_diff: f must be callable")
    step = float(h)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"geron_numerical_diff: h must be a positive finite step, got {h!r}")
    scalar = np.ndim(x) == 0
    xv = np.atleast_1d(np.asarray(x, dtype=float)).astype(float)
    if xv.size == 0:
        raise ValueError("geron_numerical_diff: x is empty")

    calls = [0]

    def _f(v):
        calls[0] += 1
        out = f(float(v[0]) if scalar else v)
        out = np.asarray(out, dtype=float)
        if out.size != 1:
            raise ValueError(f"geron_numerical_diff: f must return a scalar, got shape {out.shape}")
        if not np.isfinite(out):
            raise ValueError("geron_numerical_diff: f returned a non-finite value")
        return float(out)

    def _grad(hh):
        g = np.empty(xv.size)
        for i in range(xv.size):
            up = xv.copy()
            dn = xv.copy()
            up[i] += hh
            dn[i] -= hh
            g[i] = (_f(up) - _f(dn)) / (2.0 * hh)
        return g

    d1 = _grad(step)
    d2 = _grad(step / 2.0)
    rich = (4.0 * d2 - d1) / 3.0  # cancels the leading h^2 term
    err = np.abs(rich - d1)

    out = float(d1[0]) if scalar else d1
    rout = float(rich[0]) if scalar else rich
    return RichResult(
        title="Central finite difference",
        summary_lines=[("h", step), ("Function evaluations", calls[0])],
        interpretation="Too small an h loses precision to cancellation; the Richardson gap flags that.",
        payload={
            "derivative": out,
            "richardson": rout,
            "error_estimate": float(err[0]) if scalar else err,
            "n_evals": int(calls[0]),
            "h": step,
            "estimate": out,
            "n": int(xv.size),
            "method": "Central difference with Richardson error estimate",
        },
    )


def cheatsheet():
    return "hmnmd: Numerical differentiation via finite differences"
