# morie.fn -- function file (rootcoder007/morie)
"""Separating hyperplane -- Boyd & Vandenberghe Sec. 2.5.1."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_separating_hyperplane"]


def boyd_separating_hyperplane(C, D, tol=1e-08):
    r"""Find :math:`a \ne 0, b` with :math:`a^\top x \le b` on C and
    :math:`a^\top x \ge b` on D.

    The separating hyperplane theorem: two DISJOINT convex sets can always
    be separated. Both hypotheses are needed -- two disjoint nonconvex
    sets generally cannot be, and two convex sets that touch can only be
    separated non-strictly.

    The theorem is the geometric root of duality. A Lagrange multiplier is
    the normal vector of a hyperplane separating the achievable
    (objective, constraint) pairs from the region that would beat the
    optimum, which is why strong duality and separation are the same fact
    twice.

    Here the sets are given as point clouds and separation is computed as
    the maximum-margin hyperplane between their hulls -- so a failure to
    separate is evidence the hulls intersect.

    Parameters
    ----------
    C, D : array-like
        Point sets, one point per row.
    tol : float
        Margin tolerance.

    Returns
    -------
    RichResult
        ``a``, ``b``, ``separable``, ``margin``, ``violations``,
        ``strictly_separable``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Two clearly disjoint clouds separate, with a positive margin.

    >>> import numpy as np
    >>> C = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> D = np.array([[5.0, 5.0], [6.0, 5.0], [5.0, 6.0]])
    >>> r = boyd_separating_hyperplane(C, D)
    >>> bool(r["separable"] and r["margin"] > 0)
    True

    Every point of C is on one side and every point of D on the other --
    the separation actually holds rather than merely being reported.

    >>> a, b = r["a"], r["b"]
    >>> bool(np.all(C @ a <= b + 1e-8) and np.all(D @ a >= b - 1e-8))
    True

    Sets that TOUCH can still be separated, but only non-strictly. Here
    (0.5, 0.5) sits exactly on the triangle's hypotenuse, so a separating
    hyperplane exists with zero margin -- the theorem's distinction
    between separable and strictly separable, made visible.

    >>> E = np.array([[0.5, 0.5], [4.0, 4.0]])
    >>> t = boyd_separating_hyperplane(C, E)
    >>> bool(t["separable"]), bool(t["strictly_separable"])
    (True, False)

    Genuinely overlapping hulls cannot be separated at all, and that is
    reported rather than a hyperplane with violations being returned as
    success.

    >>> G = np.array([[0.2, 0.2], [4.0, 4.0]])
    >>> bool(boyd_separating_hyperplane(C, G)["separable"])
    False
    """
    from scipy.optimize import linprog

    Cm = np.atleast_2d(np.asarray(C, dtype=float))
    Dm = np.atleast_2d(np.asarray(D, dtype=float))
    n = Cm.shape[1]
    if Dm.shape[1] != n:
        raise ValueError("C and D must have the same dimension")
    # Variables (a, b, s): maximise the margin s subject to
    #   a'x - b + s <= 0  for x in C,  -a'y + b + s <= 0 for y in D,
    # with a bounded so the LP cannot escape by scaling a.
    nc, nd = Cm.shape[0], Dm.shape[0]
    A_ub = np.block([
        [Cm, -np.ones((nc, 1)), np.ones((nc, 1))],
        [-Dm, np.ones((nd, 1)), np.ones((nd, 1))],
    ])
    b_ub = np.zeros(nc + nd)
    c = np.r_[np.zeros(n), 0.0, -1.0]
    bounds = [(-1.0, 1.0)] * n + [(None, None), (None, 1.0)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        return RichResult(
            title="Separating hyperplane",
            summary_lines=[("separable", False)],
            warnings=["the separation LP did not solve"],
            payload={"a": np.full(n, np.nan), "b": float("nan"),
                     "separable": False, "margin": float("nan"),
                     "violations": -1, "strictly_separable": False,
                     "method": "boyd_separating_hyperplane"})
    a = np.asarray(res.x[:n], dtype=float)
    b = float(res.x[n])
    s = float(res.x[n + 1])
    viol = int(np.sum(Cm @ a > b + tol) + np.sum(Dm @ a < b - tol))
    sep = bool(viol == 0 and np.linalg.norm(a) > 1e-9)
    return RichResult(
        title="Separating hyperplane",
        summary_lines=[("dimension", int(n)), ("margin", s),
                       ("separable", sep), ("violations", viol)],
        warnings=[] if sep else
        ["no separating hyperplane was found; for convex sets that means "
         "they intersect"],
        payload={
            "a": a, "b": b, "separable": sep, "margin": s,
            "violations": viol, "strictly_separable": bool(sep and s > tol),
            "method": "boyd_separating_hyperplane",
        },
    )


def cheatsheet():
    return "cvxsep: needs BOTH disjoint and convex; it is duality restated geometrically"
