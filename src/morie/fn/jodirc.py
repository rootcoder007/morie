# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Direct multi-step forecasting.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 18 p. 548
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["dirmulti", "joseph_direct_multistep"]

_METHOD = "Direct multi-step forecasting"


def dirmulti(x, lags, horizon):
    """Direct multi-step forecasting.

    Direct multi-step forecasting, ch. 18 p. 548.

    One model PER horizon, each trained to predict h steps ahead
    directly from the same observed lags -- so no forecast is ever fed
    back in, and errors cannot compound the way they do in the
    recursive strategy.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.dirmulti``.
    lags : as documented for the shelf core
        See ``morie.fn._joseph.dirmulti``.
    horizon : as documented for the shelf core
        See ``morie.fn._joseph.dirmulti``.

    Returns
    -------
    result : RichResult
        Payload keys: first, last, mean, nmodels.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 18 p. 548
    """
    res = _core.dirmulti(x=x, lags=lags, horizon=horizon)
    return RichResult(
        title=_METHOD,
        summary_lines=[("first", res["first"]), ("last", res["last"]), ("mean", res["mean"]), ("nmodels", res["nmodels"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_direct_multistep = dirmulti


def cheatsheet():
    return "dirmulti: Direct multi-step forecasting"
