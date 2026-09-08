# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Symmetric mean absolute percentage error.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 569
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["smape", "joseph_smape"]

_METHOD = "Symmetric mean absolute percentage error"


def smape(y, yhat):
    """Symmetric mean absolute percentage error.

    Symmetric MAPE, ch. 19 p. 569.

    Quoted from p. 569:
        sMAPE = (1/H) sum_t 200 |e_t| / (|y_t| + |yhat_t|)

    Note the 200 in the numerator, which is the book's own convention:
    the symmetric denominator is the SUM of the magnitudes, not their
    average, so the factor is 200 rather than 100.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.smape``.
    yhat : as documented for the shelf core
        See ``morie.fn._joseph.smape``.

    Returns
    -------
    result : RichResult
        Payload keys: smape, smdape, n.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 569
    """
    res = _core.smape(y=y, yhat=yhat)
    return RichResult(
        title=_METHOD,
        summary_lines=[("smape", res["smape"]), ("smdape", res["smdape"]), ("n", res["n"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_smape = smape


def cheatsheet():
    return "smape: Symmetric mean absolute percentage error"
