# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recursive multi-step forecasting.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 18 p. 546
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["recmulti", "joseph_recursive_multistep"]

_METHOD = "Recursive multi-step forecasting"


def recmulti(x, lags, horizon):
    """Recursive multi-step forecasting.

    Recursive multi-step forecasting, ch. 18 p. 546.

    One model, trained for a single step, applied repeatedly with its
    own forecasts fed back in as lags.  The base learner is ordinary
    least squares on the lag design, so the strategy -- which is what
    the book is teaching -- is what is being demonstrated, and nothing
    is fitted at random.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.recmulti``.
    lags : as documented for the shelf core
        See ``morie.fn._joseph.recmulti``.
    horizon : as documented for the shelf core
        See ``morie.fn._joseph.recmulti``.

    Returns
    -------
    result : RichResult
        Payload keys: first, last, mean, nmodels.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 18 p. 546
    """
    res = _core.recmulti(x=x, lags=lags, horizon=horizon)
    return RichResult(
        title=_METHOD,
        summary_lines=[("first", res["first"]), ("last", res["last"]), ("mean", res["mean"]), ("nmodels", res["nmodels"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_recursive_multistep = recmulti


def cheatsheet():
    return "recmulti: Recursive multi-step forecasting"
