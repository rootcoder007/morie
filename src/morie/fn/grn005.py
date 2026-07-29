# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Normal equation: closed-form minimiser of the linear-regression MSE."""

import numpy as np

from ._richresult import RichResult
from .grmse import geron_linreg_mse_cost

__all__ = ["geron_ch4_normal_equation"]

_METHOD = "Normal equation (Eq 4-5)"


def geron_ch4_normal_equation(X, y):
    r"""Solve :math:`\hat\theta = (X^{\mathsf T}X)^{-1}X^{\mathsf T}y`.

    Géron Eq 4-5.  The inverse is never formed: ``X^T X theta = X^T y``
    is solved directly, which is both faster and better conditioned.
    When ``X^T X`` is singular -- perfectly collinear features, or fewer
    rows than columns -- this raises rather than returning a silently
    arbitrary member of the solution set.

    The attained cost is reported by delegating to
    :func:`morie.fn.grmse.geron_linreg_mse_cost`.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix, bias column included if wanted.
    y : array-like, shape (m,)

    Returns
    -------
    RichResult
        Payload keys ``theta``, ``cost``, ``rank``, ``condition_number``,
        ``residuals``, ``estimate`` (= ``theta``), ``n``, ``method``.

    References
    ----------
    Geron (2026), Ch 4, Eq 4-5, p. 138.

    Examples
    --------
    Three points on the line ``y = x`` recover intercept 0, slope 1:

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    >>> r = geron_ch4_normal_equation(X, [1.0, 2.0, 3.0])
    >>> [round(t, 10) for t in r["theta"]]
    [0.0, 1.0]
    >>> round(r["cost"], 12)
    0.0

    Shifting every target up by 2 moves only the intercept:

    >>> r2 = geron_ch4_normal_equation(X, [3.0, 4.0, 5.0])
    >>> [round(t, 10) for t in r2["theta"]]
    [2.0, 1.0]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D of shape (m, n), got shape {X.shape}.")
    m, n = X.shape
    if m == 0 or n == 0:
        raise ValueError(f"X must be non-empty, got shape {X.shape}.")
    if y.size != m:
        raise ValueError(f"y has {y.size} entries but X has {m} rows.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite.")
    if m < n:
        raise ValueError(
            f"X has {m} rows and {n} columns; X^T X is singular whenever m < n, "
            f"so the normal equation has no unique solution. Use a regularised "
            f"fit (grridg / grlaso) instead."
        )

    G = X.T @ X
    rank = int(np.linalg.matrix_rank(G))
    if rank < n:
        raise ValueError(
            f"X^T X is rank {rank} < {n}: the features are collinear, so "
            f"(X^T X)^-1 does not exist. Drop the redundant column(s) or regularise."
        )
    theta = np.linalg.solve(G, X.T @ y)
    cond = float(np.linalg.cond(G))
    fit = geron_linreg_mse_cost(X, y, theta)

    return RichResult(
        title="Normal equation",
        summary_lines=[("Parameters", int(n)), ("MSE", fit["cost"]),
                       ("cond(X^T X)", cond)],
        payload={
            "theta": theta.tolist(),
            "cost": fit["cost"],
            "residuals": fit["residuals"],
            "rank": rank,
            "condition_number": cond,
            "estimate": theta.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grn005: theta = (X^T X)^-1 X^T y, solved not inverted -- Geron Eq 4-5"
