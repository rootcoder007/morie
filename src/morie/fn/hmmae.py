# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean absolute error."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_mae"]

_METHOD = "Mean absolute error (l1 norm of the residuals / m)"


def geron_mae(y_true, y_pred):
    """
    Mean absolute error.

    Formula: MAE = (1/m) sum_i |y_hat_i - y_i|

    The l1 norm of the residual vector divided by ``m``, as against
    RMSE's l2 norm.  The higher the norm index the more it weights large
    residuals, so MAE is the metric to prefer when the data have
    outliers you do not want to chase.  RMSE is returned alongside for
    exactly that comparison: ``MAE <= RMSE`` always, and a large ratio
    between them means a few residuals dominate.

    Parameters
    ----------
    y_true : array-like
        Observed targets.
    y_pred : array-like
        Predictions, same length.

    Returns
    -------
    result : RichResult
        Keys: mae, rmse, max_error, median_absolute_error, ratio,
        estimate, n, method.

    Examples
    --------
    Residuals 1, 1, 1, 3 give MAE 1.5 and RMSE ``sqrt(12/4) = sqrt(3)``:

    >>> r = geron_mae([0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, 3.0])
    >>> float(r["mae"])
    1.5
    >>> round(r["rmse"], 6)
    1.732051
    >>> float(r["max_error"])
    3.0

    A perfect fit scores zero:

    >>> float(geron_mae([1.0, 2.0], [1.0, 2.0])["mae"])
    0.0

    References
    ----------
    Géron Ch 2
    """
    yt = np.atleast_1d(np.asarray(y_true, dtype=float)).ravel()
    yp = np.atleast_1d(np.asarray(y_pred, dtype=float)).ravel()
    if yt.size == 0:
        raise ValueError("geron_mae: y_true is empty")
    if yt.size != yp.size:
        raise ValueError(f"geron_mae: y_true has {yt.size} entries but y_pred has {yp.size}")
    if not np.all(np.isfinite(yt)) or not np.all(np.isfinite(yp)):
        raise ValueError("geron_mae: y_true and y_pred must be finite")

    resid = np.abs(yp - yt)
    mae = float(np.mean(resid))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    ratio = float(rmse / mae) if mae > 0 else 1.0

    return RichResult(
        title="Mean absolute error",
        summary_lines=[("MAE", mae), ("RMSE", rmse), ("Worst residual", float(np.max(resid)))],
        interpretation=(
            "MAE never exceeds RMSE; the further apart they are, the more a handful of large residuals "
            "is driving the squared-error metric."
        ),
        payload={
            "mae": mae,
            "rmse": rmse,
            "max_error": float(np.max(resid)),
            "median_absolute_error": float(np.median(resid)),
            "ratio": ratio,
            "residuals": resid,
            "estimate": mae,
            "n": int(yt.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmae: MAE = mean|y_hat - y| (l1), reported next to RMSE (l2) for outlier sensitivity"
