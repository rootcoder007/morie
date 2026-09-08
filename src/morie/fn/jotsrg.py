# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time series recast as a regression problem.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 5 p. 118
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["tsregmat", "joseph_ts_as_regression"]

_METHOD = "Time series recast as a regression problem"


def tsregmat(x, lags, horizon=1):
    """Time series recast as a regression problem.

    Time series as a regression problem, ch. 5 p. 118.

    Builds the supervised design: one row per usable time index, the
    requested lags as columns, and the value ``horizon`` steps ahead as
    the target.  Every multi-step strategy below consumes this.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.tsregmat``.
    lags : as documented for the shelf core
        See ``morie.fn._joseph.tsregmat``.
    horizon : as documented for the shelf core
        See ``morie.fn._joseph.tsregmat``.

    Returns
    -------
    result : RichResult
        Payload keys: nrows, ncols, ymean, xmean.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 5 p. 118
    """
    res = _core.tsregmat(x=x, lags=lags, horizon=horizon)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nrows", res["nrows"]), ("ncols", res["ncols"]), ("ymean", res["ymean"]), ("xmean", res["xmean"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_ts_as_regression = tsregmat


def cheatsheet():
    return "tsregmat: Time series recast as a regression problem"
