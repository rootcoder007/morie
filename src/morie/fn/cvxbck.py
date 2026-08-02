# morie.fn -- function file (rootcoder007/morie)
"""Backtracking line search -- Boyd & Vandenberghe (2004) Alg 9.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_backtracking"]


def boyd_backtracking(f, grad, x, dx, alpha=0.25, beta=0.5, max_iter=100):
    r"""Backtracking line search satisfying the Armijo sufficient-decrease rule.

    Starting from :math:`t = 1`, shrink :math:`t \leftarrow \beta t` until

    .. math::
        f(x + t\,\Delta x) \le f(x) + \alpha\, t\, \nabla f(x)^\top \Delta x .

    The condition only accepts a step that captures at least a fraction
    :math:`\alpha` of the decrease the linear model predicts, which is what
    rules out the long steps that make plain gradient descent oscillate.
    Boyd recommends :math:`\alpha \in [0.01, 0.3]` and
    :math:`\beta \in [0.1, 0.8]`.

    Parameters
    ----------
    f : callable
        Objective, ``f(x) -> float``.
    grad : array-like
        Gradient at ``x``.
    x : array-like
        Current point.
    dx : array-like
        Search direction. Must be a descent direction.
    alpha : float
        Sufficient-decrease fraction, in (0, 0.5).
    beta : float
        Shrink factor, in (0, 1).
    max_iter : int
        Cap on backtracking steps before giving up.

    Returns
    -------
    RichResult
        ``t`` (accepted step size), ``x_new``, ``f_new``, ``n_backtracks``,
        and ``converged``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Steepest descent on a quadratic; from a point where the full step
    overshoots, the search backs off.

    >>> import numpy as np
    >>> f = lambda z: float(z @ z)
    >>> x = np.array([1.0])
    >>> r = boyd_backtracking(f, 2 * x, x, -2 * x)   # dx = -grad, full step overshoots
    >>> bool(r["t"] < 1.0 and r["f_new"] < f(x))
    True

    An ascent direction is rejected rather than searched.

    >>> boyd_backtracking(f, 2 * x, x, 2 * x)
    Traceback (most recent call last):
        ...
    ValueError: dx is not a descent direction (grad @ dx = 4)
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    g = np.atleast_1d(np.asarray(grad, dtype=float)).ravel()
    d = np.atleast_1d(np.asarray(dx, dtype=float)).ravel()
    if not (x.size == g.size == d.size):
        raise ValueError("x, grad and dx must all have the same length")
    slope = float(g @ d)
    if slope >= 0:
        raise ValueError(f"dx is not a descent direction (grad @ dx = {slope:g})")
    f0 = float(f(x))
    t = 1.0
    for k in range(max_iter):
        fx = float(f(x + t * d))
        if np.isfinite(fx) and fx <= f0 + alpha * t * slope:
            return RichResult(
                title="Backtracking line search",
                summary_lines=[("t", t), ("backtracks", k)],
                payload={
                    "t": float(t),
                    "x_new": x + t * d,
                    "f_new": fx,
                    "f_old": f0,
                    "slope": slope,
                    "n_backtracks": int(k),
                    "converged": True,
                    "method": "boyd_backtracking",
                },
            )
        t *= beta
    return RichResult(
        title="Backtracking line search",
        summary_lines=[("t", t), ("backtracks", max_iter)],
        warnings=[f"no step satisfied the Armijo condition in {max_iter} backtracks"],
        payload={
            "t": float(t),
            "x_new": x + t * d,
            "f_new": float(f(x + t * d)),
            "f_old": f0,
            "slope": slope,
            "n_backtracks": int(max_iter),
            "converged": False,
            "method": "boyd_backtracking",
        },
    )


def cheatsheet():
    return "cvxbck: Armijo backtracking; rejects non-descent dx instead of looping forever"
