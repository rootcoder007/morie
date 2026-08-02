# morie.fn -- function file (rootcoder007/morie)
"""Quadratically constrained QP -- Boyd & Vandenberghe Sec. 4.4."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_quadratic_constraint"]


def boyd_quadratic_constraint(P0, q0, P=(), q=(), r=(), x0=None,
                              require_convex=True):
    r"""Solve :math:`\min \tfrac12 x^{\top}P_0x + q_0^{\top}x` subject to
    :math:`\tfrac12 x^{\top}P_ix + q_i^{\top}x + r_i \le 0`.

    A QCQP is convex only when EVERY :math:`P_i` is positive
    semidefinite, objective and constraints alike. The distinction is
    not cosmetic: with all :math:`P_i \succeq 0` this is an SOCP and
    solves in polynomial time, while a single indefinite constraint
    matrix makes the problem NP-hard in general -- boolean least squares
    and partitioning are QCQPs of exactly that kind, written with
    :math:`x_i^2 = 1`.

    So a nonconvex QCQP is refused here by default rather than handed to
    a local solver that would return a stationary point looking exactly
    like an optimum. When one is genuinely wanted, pass
    ``require_convex=False`` and read the result as LOCAL; for a
    certified lower bound instead, see :func:`boyd_qcqp_relaxation`.

    Parameters
    ----------
    P0 : array-like
        Objective Hessian, ``(n, n)``, symmetric.
    q0 : array-like
        Objective linear term, length ``n``.
    P : sequence of array-like
        One ``(n, n)`` matrix per constraint. Empty for a plain QP.
    q : sequence of array-like
        One length-``n`` vector per constraint.
    r : array-like
        One scalar per constraint.
    x0 : array-like, optional
        Starting point; defaults to zeros.
    require_convex : bool
        Refuse indefinite ``P0`` or ``P_i``.

    Returns
    -------
    RichResult
        ``x``, ``objective``, ``constraints`` (the values
        :math:`f_i(x)`), ``active``, ``feasible``, ``convex``,
        ``min_eigenvalues``, ``converged``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Minimise ``|x|^2/2 - 2*x1`` over the unit disc. The unconstrained
    minimiser is (2, 0), well outside, so the answer is where the
    objective's gradient points through the boundary.

    >>> import numpy as np
    >>> r1 = boyd_quadratic_constraint(np.eye(2), [-2.0, 0.0],
    ...                                P=[np.eye(2)], q=[[0.0, 0.0]],
    ...                                r=[-0.5])
    >>> [round(float(v), 5) for v in r1["x"]]
    [1.0, 0.0]
    >>> round(float(r1["objective"]), 5)
    -1.5

    The constraint is active, and the whole problem is certified convex
    because both Hessians are positive semidefinite.

    >>> [bool(a) for a in r1["active"]], bool(r1["convex"])
    ([True], True)

    With no quadratic constraints this reduces to a plain QP, and the
    unconstrained minimiser comes back.

    >>> qp = boyd_quadratic_constraint(np.eye(2), [-2.0, 0.0])
    >>> [round(float(v), 5) for v in qp["x"]]
    [2.0, 0.0]
    >>> round(float(qp["objective"]), 5)
    -2.0

    An indefinite constraint matrix is refused. ``x1^2 - x2^2 <= 1``
    carves out a hyperbolic region, not a convex one, and a local
    solver's answer there would be indistinguishable from an optimum
    while being neither.

    >>> boyd_quadratic_constraint(np.eye(2), [0.0, 0.0],
    ...                           P=[np.diag([2.0, -2.0])],
    ...                           q=[[0.0, 0.0]], r=[-1.0])
    Traceback (most recent call last):
        ...
    ValueError: constraint 0 has an indefinite P (min eigenvalue -2), so the QCQP is nonconvex and NP-hard in general; pass require_convex=False to accept a local solution, or use boyd_qcqp_relaxation for a certified lower bound
    """
    from scipy.optimize import minimize

    P0m = np.atleast_2d(np.asarray(P0, dtype=float))
    q0v = np.atleast_1d(np.asarray(q0, dtype=float)).ravel()
    n = q0v.size
    if P0m.shape != (n, n):
        raise ValueError(f"P0 has shape {P0m.shape}, expected ({n}, {n})")
    P0m = 0.5 * (P0m + P0m.T)
    Ps = [0.5 * (np.atleast_2d(np.asarray(Pi, dtype=float))
                 + np.atleast_2d(np.asarray(Pi, dtype=float)).T) for Pi in P]
    qs = [np.atleast_1d(np.asarray(qi, dtype=float)).ravel() for qi in q]
    rs = np.atleast_1d(np.asarray(r, dtype=float)).ravel() if len(Ps) else np.zeros(0)
    m = len(Ps)
    if not (len(qs) == rs.size == m):
        raise ValueError(
            f"P, q, r must have the same length; got {m}, {len(qs)}, {rs.size}")
    for i in range(m):
        if Ps[i].shape != (n, n):
            raise ValueError(f"P[{i}] has shape {Ps[i].shape}, expected ({n}, {n})")
        if qs[i].size != n:
            raise ValueError(f"q[{i}] has {qs[i].size} entries, expected {n}")
    lam0 = float(np.linalg.eigvalsh(P0m)[0])
    lams = np.array([float(np.linalg.eigvalsh(Pi)[0]) for Pi in Ps])
    convex = bool(lam0 >= -1e-10 and np.all(lams >= -1e-10))
    if require_convex and lam0 < -1e-10:
        raise ValueError(
            f"P0 has an indefinite Hessian (min eigenvalue {lam0:g}), so the "
            f"objective is nonconvex; pass require_convex=False to accept a "
            f"local solution")
    if require_convex and m and np.min(lams) < -1e-10:
        bad = int(np.argmin(lams))
        raise ValueError(
            f"constraint {bad} has an indefinite P (min eigenvalue "
            f"{lams[bad]:g}), so the QCQP is nonconvex and NP-hard in "
            f"general; pass require_convex=False to accept a local "
            f"solution, or use boyd_qcqp_relaxation for a certified "
            f"lower bound")

    def obj(x):
        return float(0.5 * x @ P0m @ x + q0v @ x)

    cons = [{
        "type": "ineq",
        # SLSQP wants g(x) >= 0, so the sign flips relative to the
        # f_i(x) <= 0 convention the problem is stated in.
        "fun": lambda x, Pi=Ps[i], qi=qs[i], ri=rs[i]: -(
            0.5 * float(x @ Pi @ x) + float(qi @ x) + float(ri)),
        "jac": lambda x, Pi=Ps[i], qi=qs[i]: -(Pi @ x + qi),
    } for i in range(m)]
    z0 = (np.zeros(n) if x0 is None
          else np.atleast_1d(np.asarray(x0, dtype=float)).ravel())
    if z0.size != n:
        raise ValueError(f"x0 has {z0.size} entries, expected {n}")
    res = minimize(obj, z0, jac=lambda x: P0m @ x + q0v, constraints=cons,
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
    x = np.asarray(res.x, dtype=float)
    vals = np.array([0.5 * float(x @ Ps[i] @ x) + float(qs[i] @ x) + float(rs[i])
                     for i in range(m)])
    tolv = 1e-07 * np.maximum(1.0, np.abs(rs)) if m else np.zeros(0)
    return RichResult(
        title="Quadratically constrained QP",
        summary_lines=[("n", int(n)), ("constraints", int(m)),
                       ("objective", obj(x)), ("convex", convex),
                       ("active", int(np.sum(np.abs(vals) <= tolv)))],
        payload={
            "x": x, "objective": obj(x), "constraints": vals,
            "active": np.abs(vals) <= tolv,
            "feasible": bool(np.all(vals <= tolv)) if m else True,
            "convex": convex,
            "min_eigenvalues": np.r_[lam0, lams],
            "converged": bool(res.success), "message": str(res.message),
            "method": "boyd_quadratic_constraint",
        },
    )


def cheatsheet():
    return "cvxqcr: convex iff EVERY P_i is PSD -- one indefinite constraint makes it NP-hard, not merely harder"
