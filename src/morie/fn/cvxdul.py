# morie.fn -- function file (rootcoder007/morie)
"""Lagrangian of a constrained problem -- Boyd & Vandenberghe Sec. 5.1."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_lagrangian"]


def boyd_lagrangian(f0, f=None, h=None, lambda_=None, nu=None):
    r"""The Lagrangian

    .. math::
        L(x, \lambda, \nu) = f_0(x) + \sum_i \lambda_i f_i(x)
                            + \sum_j \nu_j h_j(x).

    The multipliers price the constraints: the Lagrangian is the objective
    a solver would minimise if, instead of forbidding constraint violation,
    it charged :math:`\lambda_i` per unit of it.

    The sign convention is not decorative. Inequality multipliers must be
    NON-NEGATIVE for the dual function to lower-bound the optimum -- a
    negative :math:`\lambda_i` would pay the solver to violate
    :math:`f_i(x) \le 0`, and the bound collapses. Equality multipliers
    :math:`\nu_j` are free in sign, because there is no direction of
    violation to reward.

    Parameters
    ----------
    f0 : float
        Objective value at the point.
    f : array-like, optional
        Inequality constraint values :math:`f_i(x)`, each required
        :math:`\le 0`.
    h : array-like, optional
        Equality constraint values :math:`h_j(x)`, each required
        :math:`= 0`.
    lambda_ : array-like, optional
        Inequality multipliers, non-negative. Defaults to zeros.
    nu : array-like, optional
        Equality multipliers, free in sign. Defaults to zeros.

    Returns
    -------
    RichResult
        ``value`` (the Lagrangian), ``objective``, ``ineq_term``,
        ``eq_term``, ``feasible``, ``complementary_slackness``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    At a feasible point with zero multipliers the Lagrangian is just the
    objective.

    >>> boyd_lagrangian(3.0, f=[-1.0, -2.0], h=[0.0])["value"]
    3.0

    Charging for a satisfied constraint LOWERS the Lagrangian, which is
    why the dual function bounds the optimum from below.

    >>> r = boyd_lagrangian(3.0, f=[-1.0, -2.0], lambda_=[2.0, 1.0])
    >>> r["value"]
    -1.0

    Complementary slackness at an optimum: a multiplier is nonzero only
    where its constraint is active. Here it is violated, and reported.

    >>> bool(r["complementary_slackness"] > 0)
    True

    >>> boyd_lagrangian(1.0, f=[-1.0], lambda_=[-0.5])
    Traceback (most recent call last):
        ...
    ValueError: inequality multipliers must be non-negative
    """
    f0 = float(f0)
    fv = np.atleast_1d(np.asarray(f, dtype=float)).ravel() if f is not None else np.zeros(0)
    hv = np.atleast_1d(np.asarray(h, dtype=float)).ravel() if h is not None else np.zeros(0)
    lam = (np.zeros(fv.size) if lambda_ is None
           else np.atleast_1d(np.asarray(lambda_, dtype=float)).ravel())
    nuv = (np.zeros(hv.size) if nu is None
           else np.atleast_1d(np.asarray(nu, dtype=float)).ravel())
    if lam.size != fv.size:
        raise ValueError(f"lambda_ has {lam.size} entries but f has {fv.size}")
    if nuv.size != hv.size:
        raise ValueError(f"nu has {nuv.size} entries but h has {hv.size}")
    if np.any(lam < 0):
        raise ValueError("inequality multipliers must be non-negative")
    ineq = float(lam @ fv) if fv.size else 0.0
    eq = float(nuv @ hv) if hv.size else 0.0
    slack = float(np.sum(np.abs(lam * fv))) if fv.size else 0.0
    feasible = bool(np.all(fv <= 1e-12) and np.all(np.abs(hv) <= 1e-12))
    return RichResult(
        title="Lagrangian",
        summary_lines=[("objective", f0), ("inequality term", ineq),
                       ("equality term", eq), ("feasible", feasible)],
        payload={
            "value": f0 + ineq + eq, "objective": f0, "ineq_term": ineq,
            "eq_term": eq, "feasible": feasible,
            "complementary_slackness": slack,
            "lambda": lam, "nu": nuv,
            "method": "boyd_lagrangian",
        },
    )


def cheatsheet():
    return "cvxdul: lambda >= 0 is what makes the dual a LOWER bound; nu is free in sign"
