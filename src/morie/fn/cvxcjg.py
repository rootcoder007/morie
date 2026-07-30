# morie.fn -- function file (rootcoder007/morie)
"""Convex conjugate -- Boyd & Vandenberghe Sec. 3.3."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_conjugate"]


def boyd_conjugate(f, y, x_grid=None):
    r"""The Legendre-Fenchel conjugate
    :math:`f^*(y) = \sup_x (y^\top x - f(x))`.

    Convex in y ALWAYS, whatever f is, because it is a pointwise supremum
    of affine functions of y. That is why conjugation is the standard route
    into a dual: it manufactures convexity rather than requiring it.

    Geometrically :math:`f^*(y)` is the largest vertical gap by which the
    line of slope y clears f -- equivalently, the negated intercept of the
    supporting line with that slope. Where no such line exists the
    supremum is :math:`+\infty`, which is the conjugate's way of saying
    the slope is outside the domain rather than failing.

    Parameters
    ----------
    f : callable
        The function to conjugate.
    y : float or array-like
        Slope(s) at which to evaluate the conjugate.
    x_grid : array-like, optional
        Grid for the supremum. Defaults to a wide symmetric grid.

    Returns
    -------
    RichResult
        ``value``, ``argmax``, ``unbounded``, ``supporting_intercept``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    For :math:`f(x) = x^2/2` the conjugate is :math:`y^2/2` -- the
    quadratic is self-conjugate.

    >>> import numpy as np
    >>> g = np.linspace(-10, 10, 20001)
    >>> r = boyd_conjugate(lambda x: 0.5 * x ** 2, 2.0, x_grid=g)
    >>> round(r["value"], 6)
    2.0

    The maximiser satisfies f'(x) = y, which for this f means x = y.

    >>> round(r["argmax"], 6)
    2.0

    For :math:`f(x) = |x|` the conjugate is 0 on [-1, 1] and infinite
    outside -- the indicator of the dual-norm ball.

    >>> round(boyd_conjugate(abs, 0.5, x_grid=g)["value"], 6)
    0.0
    >>> bool(boyd_conjugate(abs, 2.0, x_grid=g)["unbounded"])
    True

    Vectorised over several slopes.

    >>> vals = boyd_conjugate(lambda x: 0.5 * x ** 2, [1.0, 2.0, 3.0],
    ...                       x_grid=g)["value"]
    >>> [round(float(v), 6) for v in vals]
    [0.5, 2.0, 4.5]
    """
    if not callable(f):
        raise ValueError("f must be callable")
    xs = (np.linspace(-50.0, 50.0, 20001) if x_grid is None
          else np.atleast_1d(np.asarray(x_grid, dtype=float)).ravel())
    fx = np.asarray([float(f(x)) for x in xs], dtype=float)
    ys = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    vals = np.empty(ys.size)
    args = np.empty(ys.size)
    unb = np.zeros(ys.size, dtype=bool)
    for i, yy in enumerate(ys):
        obj = yy * xs - fx
        j = int(np.argmax(obj))
        vals[i] = obj[j]
        args[i] = xs[j]
        # A supremum attained at a grid endpoint, still climbing, is the
        # signature of an unbounded conjugate: the slope y lies outside
        # the domain of f*.
        unb[i] = bool(j in (0, xs.size - 1) and xs.size > 2
                      and obj[j] > obj[1 if j == 0 else -2] + 1e-12)
    scalar = ys.size == 1
    return RichResult(
        title="Convex conjugate",
        summary_lines=[("slopes", int(ys.size)),
                       ("value", float(vals[0]) if scalar else float(vals.mean())),
                       ("unbounded", bool(unb.any()))],
        warnings=["the supremum is unbounded at one or more slopes; those "
                  "y lie outside the domain of the conjugate"]
        if unb.any() else [],
        payload={
            "value": float(vals[0]) if scalar else vals,
            "argmax": float(args[0]) if scalar else args,
            "unbounded": bool(unb[0]) if scalar else unb,
            "supporting_intercept": -vals[0] if scalar else -vals,
            "y": ys, "method": "boyd_conjugate",
        },
    )


def cheatsheet():
    return "cvxcjg: convex in y even when f is not; +inf just means the slope is outside dom f*"
