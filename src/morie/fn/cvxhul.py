# morie.fn -- function file (rootcoder007/morie)
"""Convex hull -- Boyd & Vandenberghe Sec. 2.1.4."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_convex_hull"]


def boyd_convex_hull(S, query=None, tol=1e-09):
    r"""The convex hull
    :math:`\operatorname{conv} S = \{\sum_i \theta_i x_i :
    \theta \ge 0, \sum_i \theta_i = 1\}`.

    The hull is the SMALLEST convex set containing S, and by
    Caratheodory's theorem every point of it is a combination of at most
    :math:`n+1` of the original points in :math:`\mathbb R^n` -- however
    many points S has. That bound is why hull membership is a small LP
    rather than a search over subsets.

    Membership here is decided by solving that LP, not by a geometric
    construction, so it works in any dimension and reports the actual
    weights. Points strictly inside have many nonzero weights; a vertex
    has exactly one.

    Parameters
    ----------
    S : array-like
        Generating points, one per row.
    query : array-like, optional
        Point(s) to test for membership.
    tol : float
        Feasibility tolerance.

    Returns
    -------
    RichResult
        ``vertices`` (indices of extreme points), ``n_vertices``,
        ``in_hull``, ``weights``, ``caratheodory_bound``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A point inside a triangle is in the hull; one outside is not.

    >>> import numpy as np
    >>> P = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    >>> boyd_convex_hull(P, query=[1.0, 1.0])["in_hull"]
    True
    >>> boyd_convex_hull(P, query=[3.0, 3.0])["in_hull"]
    False

    An interior point of a square is NOT a vertex, so the hull has three
    vertices, not four.

    >>> Q = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.5, 0.5]])
    >>> int(boyd_convex_hull(Q)["n_vertices"])
    3

    Caratheodory: in the plane every hull point is a combination of at
    most three generators, whatever the size of S.

    >>> r = boyd_convex_hull(P, query=[1.0, 1.0])
    >>> int(r["caratheodory_bound"])
    3
    >>> bool(int(np.sum(r["weights"] > 1e-9)) <= 3)
    True
    """
    from scipy.optimize import linprog

    X = np.atleast_2d(np.asarray(S, dtype=float))
    m, n = X.shape
    if m == 0:
        raise ValueError("S must contain at least one point")

    def member(q):
        # sum theta_i x_i = q, sum theta = 1, theta >= 0.
        A_eq = np.vstack([X.T, np.ones(m)])
        b_eq = np.r_[q, 1.0]
        res = linprog(np.zeros(m), A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0.0, None)] * m, method="highs")
        return res.status == 0, (np.asarray(res.x, dtype=float)
                                 if res.status == 0 else np.full(m, np.nan))

    # A generator is a vertex exactly when it is NOT in the hull of the
    # others -- which is the definition, and cheap enough at this scale.
    verts = []
    for i in range(m):
        others = np.delete(np.arange(m), i)
        if others.size == 0:
            verts.append(i)
            continue
        A_eq = np.vstack([X[others].T, np.ones(others.size)])
        b_eq = np.r_[X[i], 1.0]
        res = linprog(np.zeros(others.size), A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0.0, None)] * others.size, method="highs")
        if res.status != 0:
            verts.append(i)
    verts = np.asarray(verts, dtype=int)

    inh = None
    wts = None
    if query is not None:
        q = np.atleast_1d(np.asarray(query, dtype=float)).ravel()
        if q.size != n:
            raise ValueError(f"query must have {n} coordinates")
        inh, wts = member(q)
    return RichResult(
        title="Convex hull",
        summary_lines=[("points", int(m)), ("dimension", int(n)),
                       ("vertices", int(verts.size)),
                       ("in hull", inh if inh is not None else "n/a")],
        payload={
            "vertices": verts, "n_vertices": int(verts.size),
            "in_hull": inh, "weights": wts,
            "caratheodory_bound": int(n + 1),
            "method": "boyd_convex_hull",
        },
    )


def cheatsheet():
    return "cvxhul: Caratheodory caps it at n+1 points however big S is -- so membership is a small LP"
