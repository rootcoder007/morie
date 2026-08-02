# morie.fn -- function file (rootcoder007/morie)
"""l1 fitting -- Boyd & Vandenberghe Sec. 6.1.1."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_l1_fitting"]


def boyd_l1_fitting(A, b):
    r"""Minimise :math:`\lVert Ax - b\rVert_1`, solved as the LP

    .. math::
        \min \textstyle\sum_i t_i \quad\text{s.t.}\quad
        -t \le Ax - b \le t.

    The :math:`\ell_1` fit is ROBUST in a way least squares is not: the
    penalty grows linearly, so a gross outlier pulls with bounded
    influence instead of dominating. The fitted line passes exactly
    through n of the points -- the LP optimum is a vertex -- which is the
    regression analogue of the median, and why residuals come out exactly
    zero at those points rather than merely small.

    Parameters
    ----------
    A, b : array-like
        Design and observations.

    Returns
    -------
    RichResult
        ``x``, ``residual``, ``l1_norm``, ``n_exact`` (residuals at
        zero), ``status``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    An outlier moves the l1 fit far less than the least-squares fit.

    >>> import numpy as np
    >>> A = np.c_[np.ones(6), np.arange(6.0)]
    >>> b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 40.0])
    >>> x1 = boyd_l1_fitting(A, b)["x"]
    >>> x2 = np.linalg.lstsq(A, b, rcond=None)[0]
    >>> bool(abs(x1[1] - 1.0) < abs(x2[1] - 1.0))
    True

    The fit interpolates exactly at as many points as there are
    parameters -- the vertex property, and the sense in which this is a
    median rather than a mean.

    >>> int(boyd_l1_fitting(A, b)["n_exact"]) >= 2
    True
    """
    from scipy.optimize import linprog

    Am = np.atleast_2d(np.asarray(A, dtype=float))
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    m, n = Am.shape
    if bv.size != m:
        raise ValueError(f"A has {m} rows but b has {bv.size}")
    # Variables (x, t): minimise sum t with Ax - b <= t and -(Ax - b) <= t.
    c = np.r_[np.zeros(n), np.ones(m)]
    A_ub = np.block([[Am, -np.eye(m)], [-Am, -np.eye(m)]])
    b_ub = np.r_[bv, -bv]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * n + [(0.0, None)] * m,
                  method="highs")
    if res.status != 0:
        return RichResult(
            title="l1 fitting",
            summary_lines=[("status", str(res.message))],
            warnings=["the l1 LP did not solve"],
            payload={"x": np.full(n, np.nan), "residual": np.full(m, np.nan),
                     "l1_norm": float("nan"), "n_exact": 0,
                     "status": "failed", "method": "boyd_l1_fitting"})
    x = np.asarray(res.x[:n], dtype=float)
    resid = Am @ x - bv
    return RichResult(
        title="l1 fitting",
        summary_lines=[("m", int(m)), ("n", int(n)),
                       ("l1 norm", float(np.abs(resid).sum())),
                       ("exact fits", int(np.sum(np.abs(resid) <= 1e-8)))],
        payload={
            "x": x, "residual": resid,
            "l1_norm": float(np.abs(resid).sum()),
            "n_exact": int(np.sum(np.abs(resid) <= 1e-8)),
            "status": "optimal", "method": "boyd_l1_fitting",
        },
    )


def cheatsheet():
    return "cvxlnf: robust like a median; the fit passes EXACTLY through n points (LP vertex)"
