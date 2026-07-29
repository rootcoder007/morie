# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Root mean squared error."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_rmse"]


def geron_rmse(y_true, y_pred):
    """
    Root mean squared error.

    Formula: RMSE = sqrt((1/m) sum_i (y_hat_i - y_i)^2)

    The RMSE is the l2 norm of the residual vector divided by sqrt(m). It
    is in the units of the target and, unlike the MAE (l1 norm), it grows
    quadratically with a single large residual, which is why Geron
    recommends the MAE instead when the data have heavy outlier tails.
    Both norms are returned so the gap between them is visible.

    Parameters
    ----------
    y_true, y_pred : array-like
        Observed and predicted values, same length.

    Returns
    -------
    result : RichResult
        Keys: rmse, mse, mae, residuals, estimate, n, method.

    Examples
    --------
    >>> float(geron_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])["rmse"])
    0.0
    >>> round(float(geron_rmse([0.0, 0.0], [3.0, 4.0])["rmse"]), 6)
    3.535534
    >>> round(float(geron_rmse([0.0, 0.0], [3.0, 4.0])["mae"]), 6)
    3.5

    References
    ----------
    Geron Ch 2
    """
    a = np.atleast_1d(np.asarray(y_true, dtype=float)).ravel()
    b = np.atleast_1d(np.asarray(y_pred, dtype=float)).ravel()
    if a.size == 0:
        raise ValueError("geron_rmse: y_true is empty")
    if a.size != b.size:
        raise ValueError(f"geron_rmse: y_true has {a.size} entries but y_pred has {b.size}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("geron_rmse: inputs contain non-finite values")

    resid = b - a
    mse = float(np.mean(resid**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(resid)))
    return RichResult(
        title="Root mean squared error",
        summary_lines=[("RMSE", rmse), ("MAE", mae), ("n", int(a.size))],
        interpretation=("RMSE >= MAE always; a large ratio means a few big residuals dominate the error."),
        payload={
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "residuals": resid,
            "estimate": rmse,
            "n": int(a.size),
            "method": "Root mean squared error (l2 norm of residuals / sqrt(m))",
        },
    )


def cheatsheet():
    return "hmrms: Root mean squared error"
