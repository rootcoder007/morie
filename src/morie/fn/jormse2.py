# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Root mean squared scaled error.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 572
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["rmsse", "joseph_rmsse"]

_METHOD = "Root mean squared scaled error"


def rmsse(y, yhat, insample, season=1):
    """Root mean squared scaled error.

    Root mean squared scaled error, ch. 19 p. 572.

    Quoted from p. 572: the squared errors are scaled by the in-sample
    mean squared error of the naive forecast,

        RMSSE = sqrt( (1/H) sum_t e_t^2
                      / ( (1/(T-1)) sum_{i=2..T} (y_i - y_{i-1})^2 ) )

    This is the scaled error "used in the M5 Forecasting Competition in
    2020" (p. 572).  ``season`` generalizes the naive lag to a seasonal
    naive one; leave it at 1 for the printed formula.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.rmsse``.
    yhat : as documented for the shelf core
        See ``morie.fn._joseph.rmsse``.
    insample : as documented for the shelf core
        See ``morie.fn._joseph.rmsse``.
    season : as documented for the shelf core
        See ``morie.fn._joseph.rmsse``.

    Returns
    -------
    result : RichResult
        Payload keys: rmsse, scale, mase.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 572
    """
    res = _core.rmsse(y=y, yhat=yhat, insample=insample, season=season)
    return RichResult(
        title=_METHOD,
        summary_lines=[("rmsse", res["rmsse"]), ("scale", res["scale"]), ("mase", res["mase"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_rmsse = rmsse


def cheatsheet():
    return "rmsse: Root mean squared scaled error"
