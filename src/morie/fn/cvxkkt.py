# morie.fn -- function file (rootcoder007/morie)
"""KKT conditions -- Boyd & Vandenberghe Sec. 5.5.3."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_kkt"]


def boyd_kkt(grad_L, f=None, h=None, lambda_=None, nu=None, tol=1e-08):
    r"""Check the four KKT conditions at a candidate point.

    Stationarity :math:`\nabla f_0 + \sum\lambda_i\nabla f_i +
    \sum\nu_j\nabla h_j = 0`; primal feasibility :math:`f_i \le 0`,
    :math:`h_j = 0`; dual feasibility :math:`\lambda \ge 0`; and
    complementary slackness :math:`\lambda_i f_i = 0`.

    For a CONVEX problem satisfying a constraint qualification the four
    are necessary AND sufficient -- so a point that passes is optimal,
    full stop, with no further search. Without convexity they are only
    necessary, and a point that passes may be a saddle or a local
    maximum. Reporting "KKT satisfied" without saying which case applies
    is the error this function exists to prevent.

    Complementary slackness is the informative one: it says a constraint
    either binds or is free, never both, so the multipliers read off which
    constraints are actually doing work.

    Parameters
    ----------
    grad_L : array-like
        Gradient of the Lagrangian at the point.
    f : array-like, optional
        Inequality constraint values.
    h : array-like, optional
        Equality constraint values.
    lambda_, nu : array-like, optional
        Multipliers.
    tol : float
        Tolerance for each condition.

    Returns
    -------
    RichResult
        ``satisfied``, ``stationarity``, ``primal_feasible``,
        ``dual_feasible``, ``complementary_slackness``, ``violations``,
        ``active_constraints``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A point satisfying all four conditions.

    >>> r = boyd_kkt([0.0, 0.0], f=[-1.0, 0.0], lambda_=[0.0, 2.0])
    >>> bool(r["satisfied"])
    True

    Complementary slackness identifies which constraint is doing work:
    the second one binds and carries the multiplier.

    >>> [int(i) for i in r["active_constraints"]]
    [1]

    A negative multiplier on a SLACK constraint breaks two conditions at
    once -- dual feasibility, and complementary slackness, since the
    product lambda*f is then nonzero. Both are named rather than the
    first one found.

    >>> b = boyd_kkt([0.0], f=[-1.0], lambda_=[-0.5])
    >>> bool(b["dual_feasible"]), b["violations"]
    (False, ['dual feasibility', 'complementary slackness'])

    A negative multiplier on an ACTIVE constraint breaks only dual
    feasibility, because lambda*f is still zero there.

    >>> boyd_kkt([0.0], f=[0.0], lambda_=[-0.5])["violations"]
    ['dual feasibility']

    A nonzero multiplier on a slack constraint fails complementary
    slackness -- the condition that catches a point which is feasible and
    stationary but still not optimal.

    >>> c = boyd_kkt([0.0], f=[-3.0], lambda_=[1.0])
    >>> c["violations"]
    ['complementary slackness']
    """
    g = np.atleast_1d(np.asarray(grad_L, dtype=float)).ravel()
    fv = np.atleast_1d(np.asarray(f, dtype=float)).ravel() if f is not None else np.zeros(0)
    hv = np.atleast_1d(np.asarray(h, dtype=float)).ravel() if h is not None else np.zeros(0)
    lam = (np.zeros(fv.size) if lambda_ is None
           else np.atleast_1d(np.asarray(lambda_, dtype=float)).ravel())
    nuv = (np.zeros(hv.size) if nu is None
           else np.atleast_1d(np.asarray(nu, dtype=float)).ravel())
    if lam.size != fv.size:
        raise ValueError(f"lambda_ has {lam.size} entries but f has {fv.size}")
    stat = bool(np.max(np.abs(g)) <= tol) if g.size else True
    pf = bool((np.all(fv <= tol) if fv.size else True)
              and (np.all(np.abs(hv) <= tol) if hv.size else True))
    df = bool(np.all(lam >= -tol)) if lam.size else True
    cs_vec = lam * fv if fv.size else np.zeros(0)
    cs = bool(np.all(np.abs(cs_vec) <= tol)) if cs_vec.size else True
    viol = []
    if not stat:
        viol.append("stationarity")
    if not pf:
        viol.append("primal feasibility")
    if not df:
        viol.append("dual feasibility")
    if not cs:
        viol.append("complementary slackness")
    return RichResult(
        title="KKT conditions",
        summary_lines=[("stationarity", stat), ("primal feasible", pf),
                       ("dual feasible", df),
                       ("complementary slackness", cs)],
        warnings=["these are necessary AND sufficient only for a convex "
                  "problem under a constraint qualification; otherwise a "
                  "point satisfying them may be a saddle or a maximum"],
        payload={
            "satisfied": bool(not viol), "stationarity": stat,
            "primal_feasible": pf, "dual_feasible": df,
            "complementary_slackness": cs, "violations": viol,
            "slackness_products": cs_vec,
            "active_constraints": np.flatnonzero(np.abs(fv) <= tol)
            if fv.size else np.zeros(0, dtype=int),
            "stationarity_residual": float(np.max(np.abs(g))) if g.size else 0.0,
            "method": "boyd_kkt",
        },
    )


def cheatsheet():
    return "cvxkkt: sufficient only under convexity + a CQ; complementary slackness names the binding set"
