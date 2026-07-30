# morie.fn -- function file (rootcoder007/morie)
"""Convex combination -- Boyd & Vandenberghe Sec. 2.1.4."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_convex_combination"]


def boyd_convex_combination(x, theta=None, tol=1e-09):
    r"""The point :math:`\sum_i \theta_i x_i` with
    :math:`\theta \ge 0` and :math:`\sum_i \theta_i = 1`.

    Both conditions are load-bearing and they buy different things.
    Non-negativity restricts the result to the segment or hull rather than
    the whole affine span; summing to one keeps it affine, so a convex
    combination of points in a convex set stays in the set. Drop the
    non-negativity and you have an affine combination, which can land
    anywhere on the extended line.

    Parameters
    ----------
    x : array-like
        Points, one per row.
    theta : array-like, optional
        Weights. Defaults to uniform, i.e. the centroid.
    tol : float
        Tolerance on the simplex constraint.

    Returns
    -------
    RichResult
        ``value`` (the combination), ``theta``, ``is_centroid``,
        ``n_points``, ``support`` (indices with nonzero weight).

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Uniform weights give the centroid.

    >>> import numpy as np
    >>> P = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]])
    >>> [float(v) for v in boyd_convex_combination(P)["value"]]
    [1.0, 1.0]

    A convex combination of points in a convex set stays inside it: here
    every weight is non-negative and the result lies within the triangle's
    bounding box.

    >>> v = boyd_convex_combination(P, [0.5, 0.25, 0.25])["value"]
    >>> bool(np.all(v >= P.min(axis=0)) and np.all(v <= P.max(axis=0)))
    True

    Weights that do not sum to one are rejected rather than silently
    renormalised, because renormalising would answer a different question.

    >>> boyd_convex_combination(P, [0.5, 0.25, 0.1])
    Traceback (most recent call last):
        ...
    ValueError: theta must sum to 1, got 0.85

    >>> boyd_convex_combination(P, [1.5, -0.25, -0.25])
    Traceback (most recent call last):
        ...
    ValueError: theta must be non-negative; with negative weights this is an affine, not convex, combination
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    n = X.shape[0]
    if n == 0:
        raise ValueError("x must contain at least one point")
    if theta is None:
        th = np.full(n, 1.0 / n)
        uniform = True
    else:
        th = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
        uniform = False
    if th.size != n:
        raise ValueError(f"theta has {th.size} entries but x has {n} points")
    if np.any(th < -tol):
        raise ValueError(
            "theta must be non-negative; with negative weights this is an "
            "affine, not convex, combination")
    s = float(th.sum())
    if abs(s - 1.0) > tol:
        raise ValueError(f"theta must sum to 1, got {s:g}")
    val = th @ X
    return RichResult(
        title="Convex combination",
        summary_lines=[("points", int(n)), ("uniform", uniform),
                       ("support", int(np.sum(th > tol)))],
        payload={
            "value": val.ravel() if val.ndim else float(val),
            "theta": th, "is_centroid": bool(uniform or np.allclose(th, 1.0 / n)),
            "n_points": int(n), "support": np.flatnonzero(th > tol),
            "method": "boyd_convex_combination",
        },
    )


def cheatsheet():
    return "cvxcvc: theta >= 0 keeps it in the HULL; sum = 1 keeps it affine. Drop either and it escapes"
