# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Relative mean absolute error.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 571
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["relmae", "joseph_relative_mae"]

_METHOD = "Relative mean absolute error"


def relmae(y, yhat, benchmark):
    """Relative mean absolute error.

    Relative MAE against a benchmark forecast, ch. 19 p. 571.

    RelMAE = MAE(model) / MAE(benchmark); below 1 the model beats the
    benchmark.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.relmae``.
    yhat : as documented for the shelf core
        See ``morie.fn._joseph.relmae``.
    benchmark : as documented for the shelf core
        See ``morie.fn._joseph.relmae``.

    Returns
    -------
    result : RichResult
        Payload keys: relmae, mae, benchmae.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 19 p. 571
    """
    res = _core.relmae(y=y, yhat=yhat, benchmark=benchmark)
    return RichResult(
        title=_METHOD,
        summary_lines=[("relmae", res["relmae"]), ("mae", res["mae"]), ("benchmae", res["benchmae"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_relative_mae = relmae


def cheatsheet():
    return "relmae: Relative mean absolute error"
