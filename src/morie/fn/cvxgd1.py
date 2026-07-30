# morie.fn -- function file (rootcoder007/morie)
"""Projected gradient descent -- Boyd & Vandenberghe Sec. 8.1."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .cvxprc import boyd_projection

__all__ = ["boyd_grad_proj"]


def boyd_grad_proj(f, grad_f, x0, C="ball", t=0.05, max_iter=500,
                   tol=1e-08, **set_kw):
    r"""Iterate :math:`x^{k+1} = P_C\!\left(x^k - t\nabla f(x^k)\right)`.

    The projection is what makes the method feasible at every iterate, not
    merely at the end -- which matters whenever f is undefined outside C,
    as with a log or a square root of a constrained quantity.

    Because projection onto a convex set is nonexpansive, the composed map
    inherits the convergence of plain gradient descent: the step condition
    is still :math:`t < 2/L`, and the projection cannot make things worse.
    On a NONCONVEX C neither statement survives.

    The fixed point characterises the solution: :math:`x = P_C(x - t\nabla
    f(x))` is exactly the first-order optimality condition for a
    constrained minimum, so a converged iterate is a certificate rather
    than just a stopping point.

    Parameters
    ----------
    f, grad_f : callable
        Objective and gradient.
    x0 : array-like
        Start point.
    C : str
        Constraint set, passed to
        :func:`~morie.fn.cvxprc.boyd_projection`.
    t : float
        Step size.
    max_iter, tol
        Stopping controls.
    **set_kw
        Extra arguments for the projection (``radius``, ``lo``, ``hi``,
        ``A``, ``b``).

    Returns
    -------
    RichResult
        ``x``, ``f``, ``n_iter``, ``converged``, ``feasible``,
        ``fixed_point_residual``, ``trajectory``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*,
        Sec. 8.1 (projection on a set). The book covers the PROJECTION
        but not the projected-gradient method itself.
    Parikh, N., & Boyd, S. (2014). Proximal algorithms. *Foundations
        and Trends in Optimization*, 1(3), 123-231. Projected gradient
        is proximal gradient with the indicator of C as the penalty.

    Examples
    --------
    Minimising a quadratic whose unconstrained optimum is outside the
    ball drives the iterate to the boundary.

    >>> import numpy as np
    >>> f = lambda z: 0.5 * np.sum((z - np.array([3.0, 4.0])) ** 2)
    >>> gf = lambda z: z - np.array([3.0, 4.0])
    >>> r = boyd_grad_proj(f, gf, [0.0, 0.0], "ball", radius=1.0, t=0.5)
    >>> [round(float(v), 4) for v in r["x"]]
    [0.6, 0.8]

    Every iterate is feasible, not just the last one.

    >>> bool(np.all(np.linalg.norm(r["trajectory"], axis=1) <= 1 + 1e-9))
    True

    Convergence is certified by the fixed-point residual, which IS the
    first-order optimality condition here.

    >>> bool(r["fixed_point_residual"] < 1e-6)
    True

    On the simplex the iterates stay a probability vector throughout.

    >>> g = lambda z: np.array([1.0, 2.0, 3.0])
    >>> s = boyd_grad_proj(lambda z: float(z @ np.array([1.0, 2.0, 3.0])),
    ...                    g, [0.4, 0.4, 0.2], "simplex", t=0.1)
    >>> bool(abs(s["x"].sum() - 1) < 1e-9 and np.all(s["x"] >= -1e-12))
    True
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).ravel().copy()
    x = boyd_projection(x, C, **set_kw)["x"]
    traj = [x.copy()]
    conv = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        g = np.atleast_1d(np.asarray(grad_f(x), dtype=float)).ravel()
        x_new = boyd_projection(x - t * g, C, **set_kw)["x"]
        traj.append(x_new.copy())
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            conv = True
            break
        x = x_new
    g = np.atleast_1d(np.asarray(grad_f(x), dtype=float)).ravel()
    fp = float(np.max(np.abs(boyd_projection(x - t * g, C, **set_kw)["x"] - x)))
    return RichResult(
        title="Projected gradient descent",
        summary_lines=[("iterations", int(it)), ("f", float(f(x))),
                       ("converged", conv),
                       ("fixed-point residual", fp)],
        payload={
            "x": x, "f": float(f(x)), "n_iter": int(it), "converged": conv,
            "feasible": True, "fixed_point_residual": fp,
            "trajectory": np.asarray(traj), "set": C, "step": float(t),
            "method": "boyd_grad_proj",
        },
    )


def cheatsheet():
    return "cvxgd1: feasible at EVERY iterate; x = P_C(x - t grad f) IS the optimality condition"
