"""Sequential Quadratic Programming (SQP) for equality-constrained problems.

Thin interface over :func:`morie.fn.sqpmin.sequential_quadratic_programming` (Nocedal and
Wright 2006, Algorithm 18.3). The earlier body took full Newton steps without a merit
function and updated B with the objective gradient instead of the Lagrangian gradient.
"""

from __future__ import annotations

from collections.abc import Callable

from . import _array_core as np
from ._containers import DescriptiveResult
from .sqpmin import sequential_quadratic_programming


def sqp_optimize(
    f: Callable,
    grad_f: Callable,
    constraints: list[Callable],
    x0: np.ndarray,
    *,
    tol: float = 1e-6,
    maxiter: int = 100,
    h: float = 1e-7,
) -> DescriptiveResult:
    """Sequential Quadratic Programming for equality-constrained problems.

    Solves: min f(x) s.t. c_i(x) = 0 by line-search SQP with damped BFGS on the
    Lagrangian and an l1 merit function.

    Parameters
    ----------
    f : callable
        Objective function.
    grad_f : callable
        Gradient of the objective.
    constraints : list of callable
        Equality constraints c_i(x) = 0.
    x0 : ndarray
        Initial point.
    tol : float
        KKT tolerance.
    maxiter : int
        Maximum iterations.
    h : float
        Forward-difference step for the constraint Jacobian.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final objective; ``extra`` has x, iterations, constraint_violation,
        multipliers and converged.
    """
    cons = list(constraints)

    def jac(x):
        rows = []
        for c in cons:
            c0 = float(c(x))
            row = []
            for j in range(len(x)):
                xp = list(x)
                xp[j] += h
                row.append((float(c(xp)) - c0) / h)
            rows.append(row)
        return rows

    r = sequential_quadratic_programming(
        f, [float(v) for v in np.asarray(x0, dtype=float)], grad=lambda x: [float(v) for v in grad_f(np.asarray(x))],
        eq=cons, eq_jac=jac, tol=tol, max_iter=maxiter,
    )
    return DescriptiveResult(
        name="SQP",
        value=float(r["fun"]),
        extra={
            "x": np.asarray(r["x"]),
            "iterations": r["n_iter"],
            "constraint_violation": r["violation"],
            "multipliers": r["multipliers_eq"],
            "converged": r["converged"],
        },
    )


sqprg = sqp_optimize


# compact alias per ledger/NAMING.md
sqpoptimize = sqp_optimize


def cheatsheet() -> str:
    return "sqprg: sqp_optimize(f, grad_f, constraints, x0) -> Sequential Quadratic Programming for equality-constrained problems."
