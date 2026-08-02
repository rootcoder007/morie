# morie.fn -- function file (rootcoder007/morie)
"""Linear program in standard form -- Boyd & Vandenberghe Sec. 4.3."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_linear_program"]


def boyd_linear_program(c, A=None, b=None, G=None, h=None, bounds=None):
    r"""Solve :math:`\min c^\top x` subject to :math:`Ax = b`,
    :math:`Gx \le h`, :math:`x \ge 0` by default.

    The optimum of an LP is always attained at a VERTEX of the feasible
    polyhedron when one exists -- a linear objective has no interior
    stationary point, so it is pushed to the boundary and then to a
    corner. That is why the simplex method walks vertices at all, and why
    an LP with a bounded feasible set can be solved exactly in rational
    arithmetic while a general convex program cannot.

    Infeasible and unbounded are DIFFERENT failures and are reported
    separately: an infeasible LP has no answer, while an unbounded one has
    an answer of :math:`-\infty` and a direction of recession worth
    inspecting.

    Parameters
    ----------
    c : array-like
        Objective coefficients.
    A, b : array-like, optional
        Equality constraints.
    G, h : array-like, optional
        Inequality constraints :math:`Gx \le h`.
    bounds : list of tuple, optional
        Per-variable bounds; defaults to :math:`x \ge 0`.

    Returns
    -------
    RichResult
        ``x``, ``value``, ``status``, ``feasible``, ``bounded``,
        ``n_active`` (constraints tight at the optimum).

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    >>> r = boyd_linear_program([-1.0, -2.0], G=[[1.0, 1.0]], h=[4.0])
    >>> [round(float(v), 6) for v in r["x"]], round(r["value"], 6)
    ([0.0, 4.0], -8.0)

    The optimum sits at a vertex, so the binding constraints are tight.

    >>> int(r["n_active"]) >= 1
    True

    Unbounded and infeasible are distinguished rather than both reported
    as "no solution".

    >>> u = boyd_linear_program([-1.0], bounds=[(None, None)])
    >>> u["status"], bool(u["bounded"])
    ('unbounded', False)

    >>> i = boyd_linear_program([1.0], A=[[1.0]], b=[-5.0])
    >>> i["status"], bool(i["feasible"])
    ('infeasible', False)
    """
    from scipy.optimize import linprog

    cv = np.atleast_1d(np.asarray(c, dtype=float)).ravel()
    n = cv.size
    kw = {}
    if A is not None:
        kw["A_eq"] = np.atleast_2d(np.asarray(A, dtype=float))
        kw["b_eq"] = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    if G is not None:
        kw["A_ub"] = np.atleast_2d(np.asarray(G, dtype=float))
        kw["b_ub"] = np.atleast_1d(np.asarray(h, dtype=float)).ravel()
    bnds = [(0.0, None)] * n if bounds is None else list(bounds)
    res = linprog(cv, bounds=bnds, method="highs", **kw)
    status = {0: "optimal", 2: "infeasible", 3: "unbounded"}.get(
        int(res.status), "failed")
    ok = status == "optimal"
    x = np.asarray(res.x, dtype=float) if ok and res.x is not None else np.full(n, np.nan)
    n_active = 0
    if ok and G is not None:
        Gm = np.atleast_2d(np.asarray(G, dtype=float))
        hv = np.atleast_1d(np.asarray(h, dtype=float)).ravel()
        n_active = int(np.sum(np.abs(Gm @ x - hv) <= 1e-9))
    return RichResult(
        title="Linear program",
        summary_lines=[("n", int(n)), ("status", status),
                       ("value", float(res.fun) if ok else float("nan"))],
        warnings=[] if ok else [f"the LP is {status}"],
        payload={
            "x": x, "value": float(res.fun) if ok else (
                float("-inf") if status == "unbounded" else float("nan")),
            "status": status, "feasible": status != "infeasible",
            "bounded": status != "unbounded", "n_active": n_active,
            "message": str(res.message), "method": "boyd_linear_program",
        },
    )


def cheatsheet():
    return "cvxlin: optimum is at a VERTEX; infeasible and unbounded are different failures"
