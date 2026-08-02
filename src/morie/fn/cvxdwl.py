# morie.fn -- function file (rootcoder007/morie)
"""Lagrange dual function -- Boyd & Vandenberghe Sec. 5.1.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_dual_function"]


def boyd_dual_function(L, lambda_=None, nu=None, x_grid=None):
    r"""The dual function :math:`g(\lambda, \nu) = \inf_x L(x, \lambda, \nu)`.

    The infimum is taken over ALL x, feasible or not, and that is what makes
    the dual function useful: it is concave in :math:`(\lambda, \nu)`
    regardless of whether the primal problem is convex, because a pointwise
    infimum of affine functions is concave. Weak duality --
    :math:`g(\lambda, \nu) \le p^\star` for every
    :math:`\lambda \ge 0` -- then holds with no convexity assumption at
    all.

    The price is that the infimum is often :math:`-\infty`. That is not a
    failure; it means those multipliers give a vacuous bound and the dual
    problem will simply avoid them.

    Parameters
    ----------
    L : callable or array-like
        Either :math:`L(x)` evaluated on ``x_grid``, or an array of
        Lagrangian values whose minimum is taken directly.
    lambda_, nu : array-like, optional
        Multipliers, recorded on the result. ``lambda_`` must be
        non-negative for the weak-duality bound to hold.
    x_grid : array-like, optional
        Points at which to evaluate a callable ``L``.

    Returns
    -------
    RichResult
        ``value`` (the dual value), ``argmin``, ``unbounded``,
        ``bound_valid``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    For :math:`L(x) = x^2 - 2x` the infimum is -1, attained at 1.

    >>> import numpy as np
    >>> g = boyd_dual_function(lambda x: x ** 2 - 2 * x,
    ...                        x_grid=np.linspace(-4, 4, 801))
    >>> round(g["value"], 6)
    -1.0
    >>> round(g["argmin"], 6)
    1.0

    A Lagrangian unbounded below gives a vacuous bound, flagged rather
    than returned as a number to be trusted.

    >>> u = boyd_dual_function(lambda x: -x, x_grid=np.linspace(0, 1e12, 5))
    >>> bool(u["unbounded"])
    True

    A negative inequality multiplier voids weak duality, and the result
    says so.

    >>> boyd_dual_function([1.0, 2.0], lambda_=[-1.0])["bound_valid"]
    False
    """
    if callable(L):
        if x_grid is None:
            raise ValueError("x_grid is required when L is callable")
        xs = np.atleast_1d(np.asarray(x_grid, dtype=float)).ravel()
        vals = np.asarray([float(L(x)) for x in xs], dtype=float)
    else:
        vals = np.atleast_1d(np.asarray(L, dtype=float)).ravel()
        xs = np.arange(vals.size, dtype=float)
    if vals.size == 0:
        raise ValueError("L must supply at least one value")
    i = int(np.argmin(vals))
    lam = (np.zeros(0) if lambda_ is None
           else np.atleast_1d(np.asarray(lambda_, dtype=float)).ravel())
    nuv = (np.zeros(0) if nu is None
           else np.atleast_1d(np.asarray(nu, dtype=float)).ravel())
    valid = bool(lam.size == 0 or np.all(lam >= 0))
    # A grid minimum sitting at an endpoint, with the function still
    # falling, is the signature of an unbounded infimum on a finite grid.
    unbounded = bool(
        not np.isfinite(vals[i])
        or (i in (0, vals.size - 1) and vals.size > 2
            and abs(vals[i] - vals[1 if i == 0 else -2]) > 1e6)
    )
    return RichResult(
        title="Lagrange dual function",
        summary_lines=[("dual value", float(vals[i])),
                       ("argmin", float(xs[i])),
                       ("weak-duality bound valid", valid)],
        warnings=["the infimum appears unbounded below; this multiplier "
                  "choice gives a vacuous bound"] if unbounded else [],
        payload={
            "value": float(vals[i]), "argmin": float(xs[i]),
            "unbounded": unbounded, "bound_valid": valid,
            "lambda": lam, "nu": nuv, "values": vals,
            "method": "boyd_dual_function",
        },
    )


def cheatsheet():
    return "cvxdwl: concave in (lambda,nu) even for a NONCONVEX primal; -inf just means a vacuous bound"
