# morie.fn -- function file (rootcoder007/morie)
"""Minimum-norm solution -- Boyd & Vandenberghe Sec. 6.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_minimum_norm"]


def boyd_minimum_norm(A, b, norm=2):
    r"""Solve :math:`\min \lVert x\rVert` subject to :math:`Ax = b`.

    The norm chosen decides the CHARACTER of the answer, not just its
    size. The :math:`\ell_2` solution is dense and lies in the row space
    of A; the :math:`\ell_1` solution is SPARSE, sitting at a vertex of
    the feasible polytope; the :math:`\ell_\infty` solution spreads the
    mass as evenly as the constraints permit.

    That is the whole basis of compressed sensing: the same underdetermined
    system, solved under a different norm, gives a qualitatively different
    answer, and :math:`\ell_1` recovers a sparse truth that
    :math:`\ell_2` smears across every coordinate.

    Parameters
    ----------
    A, b : array-like
        The underdetermined system.
    norm : {1, 2, "inf"}
        Norm to minimise.

    Returns
    -------
    RichResult
        ``x``, ``norm_value``, ``n_nonzero``, ``residual``,
        ``feasible``, ``in_row_space``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    The l2 answer is dense -- every coordinate is used.

    >>> import numpy as np
    >>> A = np.array([[1.0, 1.0, 1.0]])
    >>> r2 = boyd_minimum_norm(A, [3.0], norm=2)
    >>> [round(float(v), 6) for v in r2["x"]]
    [1.0, 1.0, 1.0]

    The l1 answer is SPARSE: one coordinate carries everything, which is
    the compressed-sensing phenomenon in miniature.

    >>> r1 = boyd_minimum_norm(A, [3.0], norm=1)
    >>> int(r1["n_nonzero"])
    1

    The sup-norm answer spreads as evenly as possible -- the opposite
    extreme from l1 on the same system.

    >>> ri = boyd_minimum_norm(A, [3.0], norm="inf")
    >>> bool(np.max(ri["x"]) - np.min(ri["x"]) < 1e-6)
    True

    All three satisfy the constraint exactly; they differ in character,
    not in feasibility.

    >>> [bool(r["feasible"]) for r in (r1, r2, ri)]
    [True, True, True]
    """
    from scipy.optimize import linprog

    Am = np.atleast_2d(np.asarray(A, dtype=float))
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    m, n = Am.shape
    if bv.size != m:
        raise ValueError(f"A has {m} rows but b has {bv.size}")
    if norm == 2:
        x = np.linalg.lstsq(Am, bv, rcond=None)[0]
        val = float(np.linalg.norm(x))
    elif norm == 1:
        # min sum t s.t. -t <= x <= t, Ax = b.
        c = np.r_[np.zeros(n), np.ones(n)]
        A_ub = np.block([[np.eye(n), -np.eye(n)], [-np.eye(n), -np.eye(n)]])
        b_ub = np.zeros(2 * n)
        A_eq = np.hstack([Am, np.zeros((m, n))])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=bv,
                      bounds=[(None, None)] * n + [(0.0, None)] * n,
                      method="highs")
        x = np.asarray(res.x[:n], dtype=float) if res.status == 0 else np.full(n, np.nan)
        val = float(np.abs(x).sum())
    elif norm in ("inf", np.inf, float("inf")):
        c = np.r_[np.zeros(n), 1.0]
        ones = np.ones((n, 1))
        A_ub = np.block([[np.eye(n), -ones], [-np.eye(n), -ones]])
        b_ub = np.zeros(2 * n)
        A_eq = np.hstack([Am, np.zeros((m, 1))])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=bv,
                      bounds=[(None, None)] * n + [(0.0, None)],
                      method="highs")
        x = np.asarray(res.x[:n], dtype=float) if res.status == 0 else np.full(n, np.nan)
        val = float(np.max(np.abs(x)))
    else:
        raise ValueError('norm must be 1, 2 or "inf"')
    resid = Am @ x - bv
    # The l2 solution is the unique one lying in the row space of A;
    # the others generally are not, which is the geometric statement of
    # the same fact.
    proj = Am.T @ np.linalg.lstsq(Am @ Am.T, Am @ x, rcond=None)[0]
    return RichResult(
        title=f"Minimum {norm}-norm solution",
        summary_lines=[("n", int(n)), ("norm", str(norm)),
                       ("value", val),
                       ("nonzeros", int(np.sum(np.abs(x) > 1e-08)))],
        payload={
            "x": x, "norm_value": val,
            "n_nonzero": int(np.sum(np.abs(x) > 1e-08)),
            "residual": resid,
            "feasible": bool(np.max(np.abs(resid)) < 1e-07),
            "in_row_space": bool(np.max(np.abs(x - proj)) < 1e-07),
            "norm": str(norm), "method": "boyd_minimum_norm",
        },
    )


def cheatsheet():
    return "cvxmin: same system, different norm, different CHARACTER -- l1 sparse, l2 dense, linf flat"
