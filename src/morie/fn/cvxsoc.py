# morie.fn -- function file (rootcoder007/morie)
"""Second-order cone program -- Boyd & Vandenberghe Sec. 4.4.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_socp"]


def boyd_socp(f, A, b, c, d, x0=None):
    r"""Solve :math:`\min f^{\top}x` s.t.
    :math:`\lVert A_ix + b_i\rVert_2 \le c_i^{\top}x + d_i`.

    Each constraint says a point lies in a second-order (ice-cream)
    cone. The class strictly contains QP -- and therefore LP -- because
    a quadratic constraint is one cone constraint while every linear
    constraint is the degenerate case :math:`A_i = 0`. It is strictly
    contained in SDP, via a Schur complement.

    The reason SOCP earns its own name rather than being handed to an
    SDP solver: robust optimisation lands here naturally. "Minimise
    :math:`f^{\top}x` for the WORST case of an uncertain constraint over
    an ellipsoid" turns into exactly this norm inequality, so
    uncertainty that would otherwise need sampling becomes one
    deterministic cone with no relaxation and no loss.

    Note the right-hand side :math:`c_i^{\top}x + d_i` is itself a
    variable quantity, and must be nonnegative at any feasible point --
    a norm has nowhere to go below zero.

    Parameters
    ----------
    f : array-like
        Objective, length ``n``.
    A : sequence of array-like
        One ``(k_i, n)`` matrix per constraint. A zero matrix gives a
        purely linear constraint.
    b : sequence of array-like
        One length-``k_i`` vector per constraint.
    c : sequence of array-like
        One length-``n`` vector per constraint.
    d : array-like
        One scalar per constraint.
    x0 : array-like, optional
        Starting point; defaults to zeros.

    Returns
    -------
    RichResult
        ``x``, ``objective``, ``lhs`` (the norms), ``rhs``, ``slack``,
        ``active``, ``feasible``, ``converged``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.
    Lobo, M. S., Vandenberghe, L., Boyd, S., & Lebret, H. (1998).
        Applications of second-order cone programming. *Linear Algebra
        and its Applications*, 284(1-3), 193-228.

    Examples
    --------
    Minimise ``x1`` over the unit disc -- one cone constraint,
    ``|x| <= 1``. The answer is the leftmost point of the disc.

    >>> import numpy as np
    >>> r = boyd_socp([1.0, 0.0], [np.eye(2)], [np.zeros(2)],
    ...               [np.zeros(2)], [1.0])
    >>> [round(float(v), 5) for v in r["x"]]
    [-1.0, 0.0]
    >>> round(float(r["objective"]), 5)
    -1.0

    The constraint is active, as it must be: a LINEAR objective over a
    bounded set attains its minimum on the boundary, never inside.

    >>> [bool(a) for a in r["active"]]
    [True]

    A quadratic constraint is one cone, so QP is a special case.
    Minimise ``-x1`` subject to ``|x - (2,0)| <= 1`` and the optimum is
    the far edge of the shifted disc.

    >>> q = boyd_socp([-1.0, 0.0], [np.eye(2)], [[-2.0, 0.0]],
    ...               [np.zeros(2)], [1.0])
    >>> [round(float(v), 4) for v in q["x"]]
    [3.0, 0.0]

    An LP is the degenerate case ``A_i = 0``: with no norm term the
    constraint reads ``0 <= c'x + d``, an ordinary halfspace. Minimising
    ``-x1 - x2`` over the unit box recovers the corner.

    >>> Z, z = np.zeros((1, 2)), np.zeros(1)
    >>> box = boyd_socp([-1.0, -1.0], [Z, Z, Z, Z], [z, z, z, z],
    ...                 [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
    ...                 [1.0, 1.0, 0.0, 0.0])
    >>> [round(float(v), 4) for v in box["x"]]
    [1.0, 1.0]

    The right-hand sides come back nonnegative, which is not a check on
    the solver so much as a reminder of what the cone constraint
    implies: it bounds a norm, so it also bounds its own right-hand side
    from below.

    >>> bool(np.all(r["rhs"] >= -1e-08))
    True
    """
    from ._sci_core import minimize

    fv = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    n = fv.size
    As = [np.atleast_2d(np.asarray(Ai, dtype=float)) for Ai in A]
    bs = [np.atleast_1d(np.asarray(bi, dtype=float)).ravel() for bi in b]
    cs = [np.atleast_1d(np.asarray(ci, dtype=float)).ravel() for ci in c]
    ds = np.atleast_1d(np.asarray(d, dtype=float)).ravel()
    m = len(As)
    if not (len(bs) == len(cs) == ds.size == m):
        raise ValueError(
            f"A, b, c, d must have the same length; got {m}, {len(bs)}, "
            f"{len(cs)}, {ds.size}")
    for i in range(m):
        if As[i].shape[1] != n:
            raise ValueError(
                f"A[{i}] has {As[i].shape[1]} columns, expected {n}")
        if bs[i].size != As[i].shape[0]:
            raise ValueError(
                f"b[{i}] has {bs[i].size} entries, expected {As[i].shape[0]}")
        if cs[i].size != n:
            raise ValueError(f"c[{i}] has {cs[i].size} entries, expected {n}")

    cons = []
    for i in range(m):
        Ai, bi, ci, di = As[i], bs[i], cs[i], ds[i]
        # Squaring keeps the derivative defined at the cone's apex,
        # where the norm itself is not differentiable; on the halfspace
        # rhs >= 0 the two forms cut out the same set.
        cons.append({
            "type": "ineq",
            "fun": lambda x, Ai=Ai, bi=bi, ci=ci, di=di: (
                float(ci @ x + di) ** 2 - float(np.sum((Ai @ x + bi) ** 2))),
            "jac": lambda x, Ai=Ai, bi=bi, ci=ci, di=di: (
                2.0 * float(ci @ x + di) * ci - 2.0 * Ai.T @ (Ai @ x + bi)),
        })
        # ...but squaring ALONE also admits the mirror nappe, where the
        # right-hand side is negative and its square still dominates. So
        # the sign has to be pinned separately or the solver can return
        # a point on the wrong half of the double cone entirely.
        cons.append({
            "type": "ineq",
            "fun": lambda x, ci=ci, di=di: float(ci @ x + di),
            "jac": lambda x, ci=ci: ci,
        })

    z0 = (np.zeros(n) if x0 is None
          else np.atleast_1d(np.asarray(x0, dtype=float)).ravel())
    if z0.size != n:
        raise ValueError(f"x0 has {z0.size} entries, expected {n}")
    res = minimize(lambda x: float(fv @ x), z0, jac=lambda x: fv,
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 1000, "ftol": 1e-12})
    x = np.asarray(res.x, dtype=float)
    lhs = np.array([float(np.linalg.norm(As[i] @ x + bs[i])) for i in range(m)])
    rhs = np.array([float(cs[i] @ x + ds[i]) for i in range(m)])
    slack = rhs - lhs
    tolr = 1e-06 * np.maximum(1.0, np.abs(rhs))
    return RichResult(
        title="Second-order cone program",
        summary_lines=[("n", int(n)), ("cones", int(m)),
                       ("objective", float(fv @ x)),
                       ("min slack", float(slack.min()) if m else float("nan")),
                       ("active", int(np.sum(slack <= tolr)))],
        payload={
            "x": x, "objective": float(fv @ x),
            "lhs": lhs, "rhs": rhs, "slack": slack,
            "active": slack <= tolr,
            "feasible": bool(np.all(slack >= -tolr)),
            "converged": bool(res.success), "message": str(res.message),
            "method": "boyd_socp",
        },
    )


def cheatsheet():
    return "cvxsoc: LP/QP inside, SDP outside; ellipsoidal-uncertainty robust LPs land here EXACTLY, not by relaxation"


# compact alias per ledger/NAMING.md
boydsocp = boyd_socp
