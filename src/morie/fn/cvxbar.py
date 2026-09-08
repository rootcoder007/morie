# morie.fn -- function file (rootcoder007/morie)
"""Logarithmic barrier -- Boyd & Vandenberghe Sec. 11.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_log_barrier"]


def boyd_log_barrier(f, t=1.0, f0=None):
    r"""The barrier :math:`\phi(x) = -\sum_i \log(-f_i(x))` and the
    centering objective :math:`t f_0(x) + \phi(x)`.

    The barrier replaces hard constraints with a penalty that is finite
    inside the feasible region and :math:`+\infty` outside it, so an
    unconstrained method can solve a constrained problem. It is a
    SUBSTITUTE for the indicator function, and the substitution is exact
    only in the limit: the central point is suboptimal by at most
    :math:`m/t` for m inequality constraints, which is what makes the
    accuracy controllable by t rather than by luck.

    Strict feasibility is required, not preferred. At :math:`f_i(x) = 0`
    the logarithm diverges, so a point ON the boundary has no barrier
    value at all -- the function raises rather than returning inf, because
    an infinite objective silently poisons every downstream arithmetic.

    Parameters
    ----------
    f : array-like
        Inequality constraint values :math:`f_i(x)`, all strictly
        negative.
    t : float
        Barrier parameter. Larger t means a sharper approximation and a
        worse-conditioned problem -- that trade is the whole method.
    f0 : float, optional
        Objective value; when given, the centering objective is returned
        too.

    Returns
    -------
    RichResult
        ``barrier``, ``centering_objective``, ``gradient_factor``
        (:math:`-1/f_i`), ``suboptimality_bound`` (m/t), ``m``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Strictly interior points have a finite barrier.

    >>> r = boyd_log_barrier([-1.0, -2.0], t=10.0, f0=5.0)
    >>> round(r["barrier"], 6)
    -0.693147

    The suboptimality bound is m/t, so ten constraints at t = 100 put the
    central point within 0.1 of optimal -- accuracy set by a dial, not by
    hope.

    >>> boyd_log_barrier([-1.0] * 10, t=100.0)["suboptimality_bound"]
    0.1

    Approaching the boundary blows the barrier up, which is what keeps
    the iterate inside.

    >>> bool(boyd_log_barrier([-1e-8])["barrier"] > 18)
    True

    A point on or outside the boundary has no barrier and raises.

    >>> boyd_log_barrier([-1.0, 0.0])
    Traceback (most recent call last):
        ...
    ValueError: the barrier needs strictly feasible constraints (f_i < 0); entry 1 is 0
    """
    fv = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    if fv.size == 0:
        raise ValueError("f must contain at least one constraint")
    bad = np.flatnonzero(fv >= 0)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            "the barrier needs strictly feasible constraints (f_i < 0); "
            f"entry {i} is {fv[i]:g}")
    t = float(t)
    if t <= 0:
        raise ValueError("t must be positive")
    barrier = float(-np.sum(np.log(-fv)))
    m = int(fv.size)
    out = {
        "barrier": barrier, "gradient_factor": -1.0 / fv,
        "suboptimality_bound": m / t, "m": m, "t": t,
        "method": "boyd_log_barrier",
    }
    if f0 is not None:
        out["centering_objective"] = t * float(f0) + barrier
        out["objective"] = float(f0)
    else:
        out["centering_objective"] = None
    return RichResult(
        title="Log barrier",
        summary_lines=[("constraints", m), ("t", t), ("barrier", barrier),
                       ("suboptimality <=", m / t)],
        payload=out,
    )


def cheatsheet():
    return "cvxbar: central point is within m/t of optimal; strict feasibility is required, not preferred"


# compact alias per ledger/NAMING.md
boydlogbarrier = boyd_log_barrier
