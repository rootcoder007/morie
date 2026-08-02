# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean absolute error -- the L1 norm of the residuals."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_mae"]

_METHOD = "Mean absolute error (Eq 2-2)"


def geron_mae(y_true, y_pred):
    r"""Géron Eq 2-2.

    .. math::
        \mathrm{MAE} = \frac{1}{m}\sum_{i=1}^{m}
        \bigl|h(x^{(i)}) - y^{(i)}\bigr|

    MAE is the mean of ``|residual|``; RMSE is the root mean of
    ``residual^2``.  Because squaring rewards the tail, RMSE >= MAE
    always, with equality only when every residual has the same
    magnitude.  Both are reported so the gap is visible: a large
    ``rmse / mae`` ratio means a few instances dominate the error.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (m,)

    Returns
    -------
    RichResult
        Payload keys ``mae``, ``rmse``, ``max_error``,
        ``median_absolute_error``, ``residuals``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 2, Eq 2-2 (Mean Absolute Error).

    Examples
    --------
    Residuals ``1, 1, 1, 5``: MAE is 2, RMSE is ``sqrt(28/4)`` = 2.6458,
    and the ratio flags the outlier:

    >>> r = geron_mae([0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, 5.0])
    >>> r["mae"]
    2.0
    >>> round(r["rmse"], 7)
    2.6457513
    >>> r["max_error"]
    5.0

    Equal-size residuals close the gap exactly:

    >>> r2 = geron_mae([0.0, 0.0], [2.0, -2.0])
    >>> r2["mae"] == r2["rmse"]
    True
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size != y_pred.size:
        raise ValueError(f"y_true has {y_true.size} entries but y_pred has {y_pred.size}.")
    if y_true.size == 0:
        raise ValueError("MAE over zero instances is undefined.")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("y_true and y_pred must be finite.")

    resid = y_pred - y_true
    a = np.abs(resid)
    mae = float(np.mean(a))
    rmse = float(np.sqrt(np.mean(resid**2)))

    return RichResult(
        title="Mean absolute error",
        summary_lines=[("MAE", mae), ("RMSE", rmse), ("m", int(a.size))],
        payload={
            "mae": mae,
            "rmse": rmse,
            "max_error": float(a.max()),
            "median_absolute_error": float(np.median(a)),
            "residuals": resid.tolist(),
            "estimate": mae,
            "n": int(a.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmae: MAE = mean|y_pred - y_true|; RMSE reported alongside -- Geron Eq 2-2"
