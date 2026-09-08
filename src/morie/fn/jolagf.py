# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lag features.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 170
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["lagfeat", "joseph_lag_feature"]

_METHOD = "Lag features"


def lagfeat(x, lags):
    """Lag features.

    Lag features, ch. 6 p. 170.

    Returns the design rows for which every requested lag exists, so
    the matrix has no missing cells.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.lagfeat``.
    lags : as documented for the shelf core
        See ``morie.fn._joseph.lagfeat``.

    Returns
    -------
    result : RichResult
        Payload keys: nrows, ncols, mean.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 170
    """
    res = _core.lagfeat(x=x, lags=lags)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nrows", res["nrows"]), ("ncols", res["ncols"]), ("mean", res["mean"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_lag_feature = lagfeat


def cheatsheet():
    return "lagfeat: Lag features"
