# morie.fn -- function file (rootcoder007/morie)
"""Chebyshev center of a polyhedron -- Boyd & Vandenberghe Sec. 4.3.1."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_chebyshev_center"]


def boyd_chebyshev_center(A, b, max_iter=500, tol=1e-10):
    r"""The centre of the largest ball inscribed in
    :math:`\{x : Ax \le b\}`:

    .. math::
        \max_{x, r} \; r \quad \text{s.t.} \quad
        a_i^\top x + r\lVert a_i\rVert_2 \le b_i, \; r \ge 0.

    The :math:`\lVert a_i \rVert_2` term is the point of the
    formulation. The distance from x to the hyperplane
    :math:`a_i^\top x = b_i` is :math:`(b_i - a_i^\top x)/\lVert a_i
    \rVert`, so requiring that distance to exceed r is LINEAR in
    (x, r) once written this way -- an apparently geometric problem
    becomes an LP.

    Omitting the norm silently solves a different problem whenever the
    rows are unnormalised, and gives an answer that looks perfectly
    reasonable.

    Parameters
    ----------
    A : array-like
        Constraint matrix, ``(m, n)``.
    b : array-like
        Right-hand side, ``(m,)``.
    max_iter, tol : int, float
        Controls for the internal LP solve.

    Returns
    -------
    RichResult
        ``center``, ``radius``, ``active`` (constraints touching the
        ball), ``distances``, ``bounded``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    The unit square has centre (0.5, 0.5) and inradius 0.5.

    >>> import numpy as np
    >>> A = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    >>> b = np.array([0.0, 1.0, 0.0, 1.0])
    >>> r = boyd_chebyshev_center(A, b)
    >>> [round(float(v), 4) for v in r["center"]], round(r["radius"], 4)
    ([0.5, 0.5], 0.5)

    All four sides touch the inscribed ball.

    >>> int(r["active"].sum())
    4

    Scaling a row must not change the geometry -- this is exactly what
    the ||a_i|| term protects, and a formulation that drops it fails here.

    >>> A2 = A.copy(); b2 = b.copy()
    >>> A2[1] *= 10.0; b2[1] *= 10.0
    >>> r2 = boyd_chebyshev_center(A2, b2)
    >>> bool(abs(r2["radius"] - r["radius"]) < 1e-6)
    True
    """
    Am = np.atleast_2d(np.asarray(A, dtype=float))
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    m, n = Am.shape
    if bv.size != m:
        raise ValueError(f"A has {m} rows but b has {bv.size} entries")
    norms = np.sqrt((Am ** 2).sum(axis=1))
    if np.any(norms <= 0):
        raise ValueError("A has a zero row, which is not a constraint")

    # Variables are (x, r); maximise r subject to a_i'x + r||a_i|| <= b_i.
    # Written this way the problem is a linear program, which is the whole
    # point of the ||a_i|| scaling.
    from scipy.optimize import linprog

    c = np.r_[np.zeros(n), -1.0]
    A_ub = np.column_stack([Am, norms])
    res = linprog(c, A_ub=A_ub, b_ub=bv,
                  bounds=[(None, None)] * n + [(0.0, None)],
                  method="highs")
    if not res.success:
        return RichResult(
            title="Chebyshev center",
            summary_lines=[("n", int(n)), ("constraints", int(m)),
                           ("status", res.message)],
            warnings=["the LP did not solve: the polyhedron is empty or "
                      "unbounded in the direction of the inscribed ball"],
            payload={"center": np.full(n, np.nan), "radius": float("nan"),
                     "active": np.zeros(m, dtype=bool),
                     "distances": np.full(m, np.nan), "bounded": False,
                     "norms": norms, "method": "boyd_chebyshev_center"})
    x = np.asarray(res.x[:n], dtype=float)
    r = float(res.x[n])
    w = (bv - Am @ x) / norms
    return RichResult(
        title="Chebyshev center",
        summary_lines=[("n", int(n)), ("constraints", int(m)),
                       ("radius", r)],
        payload={
            "center": x, "radius": r,
            "active": np.abs(w - r) <= 1e-7 * max(1.0, abs(r)),
            "distances": w, "bounded": True, "norms": norms,
            "method": "boyd_chebyshev_center",
        },
    )


def cheatsheet():
    return "cvxchb: the ||a_i|| term is what makes it an LP AND what makes it row-scale invariant"
