# morie.fn -- function file (rootcoder007/morie)
"""Chebyshev (l-infinity) fitting -- Boyd & Vandenberghe Sec. 6.1.1."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_linf_fitting"]


def boyd_linf_fitting(A, b):
    r"""Minimise :math:`\lVert Ax - b\rVert_\infty`, as the LP

    .. math::
        \min t \quad\text{s.t.}\quad -t\mathbf 1 \le Ax - b \le t\mathbf 1.

    The exact opposite trade to :math:`\ell_1`. Only the WORST residual
    enters the objective, so a single outlier moves the whole fit -- this
    is the least robust of the three fits, not a middle ground between
    :math:`\ell_1` and :math:`\ell_2`.

    It is the right choice when the requirement really is a uniform
    guarantee (a tolerance every point must meet), and the wrong one
    whenever the data may contain a bad point. At the optimum several
    residuals tie at :math:`\pm t` -- the equioscillation that
    characterises Chebyshev approximation.

    Parameters
    ----------
    A, b : array-like
        Design and observations.

    Returns
    -------
    RichResult
        ``x``, ``residual``, ``linf_norm``, ``n_active`` (residuals at
        the maximum), ``equioscillates``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.c_[np.ones(5), np.arange(5.0)]
    >>> b = np.array([0.0, 1.0, 2.0, 3.0, 4.5])
    >>> r = boyd_linf_fitting(A, b)
    >>> bool(r["linf_norm"] < 0.2)
    True

    Several residuals tie at the maximum -- the equioscillation property.

    >>> int(r["n_active"]) >= 2
    True

    Least robust of the three fits: one outlier moves it further than it
    moves the l1 fit.

    >>> from morie.fn.cvxlnf import boyd_l1_fitting
    >>> bo = np.array([0.0, 1.0, 2.0, 3.0, 40.0])
    >>> xi = boyd_linf_fitting(A, bo)["x"]
    >>> x1 = boyd_l1_fitting(A, bo)["x"]
    >>> bool(abs(xi[1] - 1.0) > abs(x1[1] - 1.0))
    True
    """
    from scipy.optimize import linprog

    Am = np.atleast_2d(np.asarray(A, dtype=float))
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    m, n = Am.shape
    if bv.size != m:
        raise ValueError(f"A has {m} rows but b has {bv.size}")
    c = np.r_[np.zeros(n), 1.0]
    ones = np.ones((m, 1))
    A_ub = np.block([[Am, -ones], [-Am, -ones]])
    b_ub = np.r_[bv, -bv]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * n + [(0.0, None)], method="highs")
    if res.status != 0:
        return RichResult(
            title="l-infinity fitting",
            summary_lines=[("status", str(res.message))],
            warnings=["the Chebyshev LP did not solve"],
            payload={"x": np.full(n, np.nan), "residual": np.full(m, np.nan),
                     "linf_norm": float("nan"), "n_active": 0,
                     "equioscillates": False,
                     "method": "boyd_linf_fitting"})
    x = np.asarray(res.x[:n], dtype=float)
    resid = Am @ x - bv
    t = float(np.max(np.abs(resid)))
    active = np.abs(np.abs(resid) - t) <= 1e-8 * max(1.0, t)
    return RichResult(
        title="l-infinity fitting",
        summary_lines=[("m", int(m)), ("n", int(n)), ("linf norm", t),
                       ("active residuals", int(active.sum()))],
        payload={
            "x": x, "residual": resid, "linf_norm": t,
            "n_active": int(active.sum()),
            "equioscillates": bool(active.sum() >= n + 1),
            "active": active, "method": "boyd_linf_fitting",
        },
    )


def cheatsheet():
    return "cvxlif: only the WORST residual counts -- least robust of the three, not a middle ground"
