# morie.fn -- function file (rootcoder007/morie)
"""LP duality -- Boyd & Vandenberghe Sec. 5.2."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_linear_program_dual"]


def boyd_linear_program_dual(A, b, c):
    r"""The dual of :math:`\min c^\top x` s.t. :math:`Ax = b`,
    :math:`x \ge 0`, namely :math:`\max b^\top y` s.t.
    :math:`A^\top y \le c`.

    LP duality is STRONG whenever either problem is feasible -- no
    constraint qualification is needed, unlike the general convex case
    where Slater's condition does that work. So the two optima coincide
    exactly, and the dual solution is not an approximation but the same
    number reached from below.

    The dual variables are the shadow prices: :math:`y_i` is the rate at
    which the optimal value changes per unit increase in :math:`b_i`,
    which is usually the answer the person asking the question actually
    wanted.

    Parameters
    ----------
    A, b, c : array-like
        Primal data.

    Returns
    -------
    RichResult
        ``y`` (dual solution), ``dual_value``, ``primal_value``,
        ``gap``, ``strong_duality``, ``shadow_prices``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Primal and dual optima agree exactly -- strong duality, with no
    constraint qualification required.

    >>> import numpy as np
    >>> A = np.array([[1.0, 1.0]])
    >>> r = boyd_linear_program_dual(A, [4.0], [1.0, 2.0])
    >>> round(r["primal_value"], 8) == round(r["dual_value"], 8)
    True

    >>> bool(r["strong_duality"])
    True

    The dual variable is the shadow price: raising b by one raises the
    optimal cost by y.

    >>> y = float(r["y"][0])
    >>> r2 = boyd_linear_program_dual(A, [5.0], [1.0, 2.0])
    >>> bool(abs((r2["primal_value"] - r["primal_value"]) - y) < 1e-8)
    True
    """
    from scipy.optimize import linprog

    Am = np.atleast_2d(np.asarray(A, dtype=float))
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    cv = np.atleast_1d(np.asarray(c, dtype=float)).ravel()
    m, n = Am.shape
    if bv.size != m or cv.size != n:
        raise ValueError(
            f"A is ({m}, {n}) but b has {bv.size} and c has {cv.size}")
    primal = linprog(cv, A_eq=Am, b_eq=bv, bounds=[(0.0, None)] * n,
                     method="highs")
    # max b'y  ==  min -b'y, subject to A'y <= c, y free.
    dual = linprog(-bv, A_ub=Am.T, b_ub=cv, bounds=[(None, None)] * m,
                   method="highs")
    p_ok = primal.status == 0
    d_ok = dual.status == 0
    pv = float(primal.fun) if p_ok else float("nan")
    dv = float(-dual.fun) if d_ok else float("nan")
    gap = pv - dv if (p_ok and d_ok) else float("nan")
    return RichResult(
        title="LP dual",
        summary_lines=[("primal", pv), ("dual", dv), ("gap", gap)],
        warnings=[] if (p_ok and d_ok) else
        ["one of the two problems did not solve; for an LP that means "
         "primal infeasible or unbounded, which the dual mirrors"],
        payload={
            "y": np.asarray(dual.x, dtype=float) if d_ok else np.full(m, np.nan),
            "x": np.asarray(primal.x, dtype=float) if p_ok else np.full(n, np.nan),
            "dual_value": dv, "primal_value": pv, "gap": gap,
            "strong_duality": bool(p_ok and d_ok and abs(gap) < 1e-7),
            "shadow_prices": np.asarray(dual.x, dtype=float) if d_ok else None,
            "method": "boyd_linear_program_dual",
        },
    )


def cheatsheet():
    return "cvxlpl: LP duality is strong with NO constraint qualification; y are shadow prices"
