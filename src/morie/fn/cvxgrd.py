# morie.fn -- function file (rootcoder007/morie)
"""Gradient descent -- Boyd & Vandenberghe Sec. 9.3."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_gradient_descent"]


def boyd_gradient_descent(f, grad_f, x0, t=0.01, max_iter=1000, tol=1e-08):
    r"""Iterate :math:`x^{k+1} = x^k - t\,\nabla f(x^k)`.

    Convergence for a fixed step needs :math:`t < 2/L` with L the Lipschitz
    constant of the gradient -- for a quadratic, :math:`L` is the largest
    eigenvalue of the Hessian. Past that the iterates diverge, and they do
    so monotonically in magnitude, which is easy to mistake for a bad
    starting point.

    Even inside the stable range the rate is governed by the CONDITION
    NUMBER: error falls by a factor :math:`((\kappa-1)/(\kappa+1))^2` per
    step, so a poorly scaled problem crawls no matter how carefully t is
    tuned. That is the argument for Newton's method, not a reason to
    tune harder.

    Parameters
    ----------
    f : callable
        Objective, for monitoring.
    grad_f : callable
        Gradient.
    x0 : array-like
        Starting point.
    t : float
        Fixed step size.
    max_iter, tol
        Stopping controls (tol on the gradient norm).

    Returns
    -------
    RichResult
        ``x``, ``f``, ``n_iter``, ``converged``, ``grad_norm``,
        ``trajectory``, ``diverged``, ``monotone``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A well-conditioned quadratic converges quickly.

    >>> import numpy as np
    >>> Q = np.diag([1.0, 2.0])
    >>> r = boyd_gradient_descent(lambda x: 0.5 * x @ Q @ x,
    ...                           lambda x: Q @ x, [1.0, 1.0], t=0.4)
    >>> bool(r["converged"] and np.max(np.abs(r["x"])) < 1e-6)
    True

    A step past 2/L diverges rather than converging slowly, and is
    reported as such.

    >>> d = boyd_gradient_descent(lambda x: 0.5 * x @ Q @ x,
    ...                           lambda x: Q @ x, [1.0, 1.0], t=1.5,
    ...                           max_iter=200)
    >>> bool(d["diverged"])
    True

    Conditioning, not the step size, sets the rate: the same solver needs
    far more iterations on a stretched quadratic.

    >>> Q2 = np.diag([1.0, 100.0])
    >>> s = boyd_gradient_descent(lambda x: 0.5 * x @ Q2 @ x,
    ...                           lambda x: Q2 @ x, [1.0, 1.0], t=0.019)
    >>> bool(s["n_iter"] > r["n_iter"])
    True
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).ravel().copy()
    t = float(t)
    if t <= 0:
        raise ValueError("t must be positive")
    traj = [x.copy()]
    fs = [float(f(x))]
    conv = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        g = np.atleast_1d(np.asarray(grad_f(x), dtype=float)).ravel()
        if np.max(np.abs(g)) < tol:
            conv = True
            break
        x = x - t * g
        traj.append(x.copy())
        fs.append(float(f(x)))
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e12:
            break
    fs = np.asarray(fs)
    diverged = bool(not np.all(np.isfinite(fs)) or fs[-1] > fs[0] * 1.0 + 1e-9
                    and fs[-1] > fs[0])
    return RichResult(
        title="Gradient descent",
        summary_lines=[("iterations", int(it)), ("f", float(fs[-1])),
                       ("converged", conv), ("diverged", diverged)],
        warnings=["the iterates diverged; a fixed step needs t < 2/L, and "
                  "L is the largest Hessian eigenvalue"] if diverged else [],
        payload={
            "x": x, "f": float(fs[-1]), "n_iter": int(it),
            "converged": conv, "diverged": diverged,
            "grad_norm": float(np.max(np.abs(
                np.atleast_1d(np.asarray(grad_f(x), dtype=float))))),
            "trajectory": np.asarray(traj), "objective_path": fs,
            "monotone": bool(np.all(np.diff(fs) <= 1e-12)),
            "step": t, "method": "boyd_gradient_descent",
        },
    )


def cheatsheet():
    return "cvxgrd: needs t < 2/L to converge at all; the RATE is set by the condition number"
