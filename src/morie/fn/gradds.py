# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Gradient descent.

Cauchy (1847), "Methode generale pour la resolution des systemes
d'equations simultanees", C. R. Acad. Sci. Paris 25:536-538.  The
iteration is

    x_{t+1} = x_t - lr * grad f(x_t),

and on the quadratic f(x) = x^2 it has the exact closed-form solution
x_t = x_0 (1 - 2 lr)^t, which is what the tests check: convergence for
lr < 1, oscillation at lr = 1 and divergence beyond.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gradient_descent"]


def gradient_descent(f, grad_f, x0, lr=0.1, steps=100, tol=1e-12):
    """Fixed-step gradient descent from x0.

    Parameters
    ----------
    f : callable
        Objective, taking a list of floats and returning a float.
    grad_f : callable
        Gradient, taking a list and returning a list of the same length.
    x0 : array-like
        Starting point.
    lr : float
        Step size.
    steps : int
        Number of iterations.
    tol : float
        Stop early when the gradient norm falls below this.
    """
    x = core.vec(x0)
    if len(x) == 0:
        raise ValueError("gradient_descent: x0 is empty")
    if not callable(f) or not callable(grad_f):
        raise ValueError("gradient_descent: f and grad_f must be callable")
    if float(lr) <= 0:
        raise ValueError("gradient_descent: lr must be positive")
    ns = int(steps)
    if ns < 1:
        raise ValueError("gradient_descent: steps must be at least 1")
    path = [float(f(x))]
    gn = float("inf")
    used = 0
    for _ in range(ns):
        g = core.vec(grad_f(x))
        if len(g) != len(x):
            raise ValueError("gradient_descent: gradient has the wrong length")
        gn = math.sqrt(sum(v * v for v in g))
        if gn <= tol:
            break
        x = [x[i] - float(lr) * g[i] for i in range(len(x))]
        path.append(float(f(x)))
        used += 1
    return RichResult(
        title="Gradient descent",
        summary_lines=[("steps", used), ("lr", float(lr))],
        payload={
            "estimate": float(f(x)),
            "x": x,
            "f_path": path,
            "grad_norm": gn,
            "steps_used": used,
            "converged": bool(gn <= tol),
            "n": len(x),
            "method": "x <- x - lr grad f(x), Cauchy (1847)",
        },
    )


def cheatsheet():
    return "gradds: gradient descent"
