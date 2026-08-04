# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Root mean squared error.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 566
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["rmse", "joseph_rmse"]

_METHOD = "Root mean squared error"


def rmse(y, yhat):
    """Root mean squared error.

    Root mean squared error, ch. 19 p. 566.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.rmse``.
    yhat : as documented for the shelf core
        See ``morie.fn._joseph.rmse``.

    Returns
    -------
    result : RichResult
        Payload keys: rmse, mse, mae, bias.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 566
    """
    res = _core.rmse(y=y, yhat=yhat)
    return RichResult(
        title=_METHOD,
        summary_lines=[("rmse", res["rmse"]), ("mse", res["mse"]), ("mae", res["mae"]), ("bias", res["bias"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_rmse = rmse


def cheatsheet():
    return "rmse: Root mean squared error"
