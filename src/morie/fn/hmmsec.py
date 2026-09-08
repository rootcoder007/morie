# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MSE cost function for linear regression."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_linreg_mse_cost"]


def geron_linreg_mse_cost(X, y, theta):
    """
    MSE cost function for linear regression.

    Formula: MSE(theta) = (1/m) sum_i (theta^T x^(i) - y^(i))^2

    The cost and its gradient (2/m) X^T (X theta - y) are returned
    together, because the gradient is what every solver in Geron chapter
    4 consumes. The MSE surface for a linear model is convex with a
    single minimum, so a zero gradient certifies the global optimum;
    ``grad_norm`` is reported for exactly that check.

    ``theta`` may be one entry longer than ``X`` has columns, in which
    case the leading entry is treated as the bias and a column of ones is
    prepended.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Feature matrix.
    y : array-like, shape (m,)
        Targets.
    theta : array-like, shape (n,) or (n + 1,)
        Parameter vector.

    Returns
    -------
    result : RichResult
        Keys: cost, gradient, grad_norm, residuals, predictions,
        estimate, n, method.

    Examples
    --------
    >>> r = geron_linreg_mse_cost([[1.0, 0.0], [1.0, 1.0]], [0.0, 2.0], [0.0, 1.0])
    >>> float(r["cost"])
    0.5
    >>> [float(v) for v in r["gradient"]]
    [-1.0, -1.0]

    At the least-squares solution the gradient vanishes:

    >>> r0 = geron_linreg_mse_cost([[1.0, 1.0], [1.0, 2.0]], [3.0, 5.0], [1.0, 2.0])
    >>> float(r0["cost"]), round(float(r0["grad_norm"]), 12)
    (0.0, 0.0)

    References
    ----------
    Geron Ch 4
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_linreg_mse_cost: X must be 2-D, got ndim={A.ndim}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    th = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
    m = A.shape[0]
    if m == 0:
        raise ValueError("geron_linreg_mse_cost: X has no rows")
    if yv.size != m:
        raise ValueError(f"geron_linreg_mse_cost: X has {m} rows but y has {yv.size} entries")
    if th.size == A.shape[1] + 1:
        A = np.hstack([np.ones((m, 1)), A])
    elif th.size != A.shape[1]:
        raise ValueError(
            f"geron_linreg_mse_cost: theta has {th.size} entries but X has {A.shape[1]} columns "
            "(theta must match the columns, or be one longer for an implicit bias)"
        )
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yv)) or not np.all(np.isfinite(th)):
        raise ValueError("geron_linreg_mse_cost: inputs contain non-finite values")

    pred = A @ th
    resid = pred - yv
    cost = float(np.mean(resid**2))
    grad = (2.0 / m) * (A.T @ resid)
    return RichResult(
        title="Linear-regression MSE cost",
        summary_lines=[("MSE", cost), ("Gradient norm", float(np.linalg.norm(grad)))],
        interpretation="The MSE surface is convex for a linear model, so a zero gradient is the global minimum.",
        payload={
            "cost": cost,
            "mse": cost,
            "gradient": grad,
            "grad_norm": float(np.linalg.norm(grad)),
            "residuals": resid,
            "predictions": pred,
            "estimate": cost,
            "n": int(m),
            "method": "MSE cost and analytic gradient for linear regression",
        },
    )


def cheatsheet():
    return "hmmsec: MSE cost function for linear regression"
