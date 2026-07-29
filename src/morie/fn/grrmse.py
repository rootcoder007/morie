# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Root mean squared error -- L2 norm of prediction residuals."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_rmse"]

_METHOD = "Root mean squared error"


def geron_rmse(y_true, y_pred):
    r"""RMSE of a prediction vector.

    .. math::
        \mathrm{RMSE} = \sqrt{\frac{1}{m}\sum_{i=1}^{m}
                        \bigl(h(x^{(i)}) - y^{(i)}\bigr)^{2}}

    Divides by :math:`m`, not :math:`m-1`: this is a performance measure
    on a held-out set, not a variance estimate.  MAE is reported next to
    it because the gap between the two is what tells you whether a few
    large residuals are driving the score.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (m,)
        Must be the same length and finite.

    Returns
    -------
    RichResult
        Payload keys ``rmse``, ``mse``, ``mae``, ``max_error``,
        ``residuals``, ``estimate`` (rmse), ``n``, ``method``.

    References
    ----------
    Géron Ch 2, Eq 2-1 (Root Mean Squared Error).

    Examples
    --------
    Residuals ``[1, -1, 2]``: MSE ``= 6/3 = 2``, RMSE ``= sqrt(2)``:

    >>> r = geron_rmse([1.0, 2.0, 3.0], [2.0, 1.0, 5.0])
    >>> round(r["mse"], 10)
    2.0
    >>> round(r["rmse"], 6)
    1.414214

    RMSE >= MAE always, with equality only when all residuals are equal
    in magnitude:

    >>> r["rmse"] >= r["mae"]
    True
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size == 0:
        raise ValueError("y_true is empty; RMSE is undefined for zero instances.")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true has {yt.size} entries but y_pred has {yp.size}.")
    if not np.all(np.isfinite(yt)) or not np.all(np.isfinite(yp)):
        raise ValueError("y_true and y_pred must be finite.")

    res = yp - yt
    mse = float(np.mean(res**2))
    rmse = float(np.sqrt(mse))
    return RichResult(
        title="Root mean squared error",
        summary_lines=[("RMSE", rmse), ("MAE", float(np.mean(np.abs(res))))],
        payload={
            "rmse": rmse,
            "mse": mse,
            "mae": float(np.mean(np.abs(res))),
            "max_error": float(np.max(np.abs(res))),
            "residuals": res.tolist(),
            "estimate": rmse,
            "n": int(yt.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grrmse: RMSE = sqrt(mean((y_hat - y)^2)); divides by m, reports MAE and max |residual| too"
