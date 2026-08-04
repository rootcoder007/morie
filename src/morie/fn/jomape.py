# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean absolute percentage error.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 568
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["mapets", "joseph_mape"]

_METHOD = "Mean absolute percentage error"


def mapets(y, yhat):
    """Mean absolute percentage error.

    Mean absolute percentage error, ch. 19 p. 568.

    The book's own warning is honoured: MAPE "breaks down when the
    actual observation is zero (due to division by zero)" (p. 568), so
    zero actuals raise rather than silently returning infinity.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.mapets``.
    yhat : as documented for the shelf core
        See ``morie.fn._joseph.mapets``.

    Returns
    -------
    result : RichResult
        Payload keys: mape, mdape, maxape.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 568
    """
    res = _core.mapets(y=y, yhat=yhat)
    return RichResult(
        title=_METHOD,
        summary_lines=[("mape", res["mape"]), ("mdape", res["mdape"]), ("maxape", res["maxape"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_mape = mapets


def cheatsheet():
    return "mapets: Mean absolute percentage error"
