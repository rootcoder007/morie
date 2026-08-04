# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Expanding-window cross-validation.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 5 p. 130
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["expandcv", "joseph_expanding_window_cv"]

_METHOD = "Expanding-window cross-validation"


def expandcv(n, initial, testsize, step=None):
    """Expanding-window cross-validation.

    Expanding-window cross-validation, ch. 5 p. 130.

    The training window GROWS: every fold starts at index 0, so no
    history is ever discarded.

    Parameters
    ----------
    n : as documented for the shelf core
        See ``morie.fn._joseph.expandcv``.
    initial : as documented for the shelf core
        See ``morie.fn._joseph.expandcv``.
    testsize : as documented for the shelf core
        See ``morie.fn._joseph.expandcv``.
    step : as documented for the shelf core
        See ``morie.fn._joseph.expandcv``.

    Returns
    -------
    result : RichResult
        Payload keys: nfolds, firsttrainend, lasttrainend.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 5 p. 130
    """
    res = _core.expandcv(n=n, initial=initial, testsize=testsize, step=step)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nfolds", res["nfolds"]), ("firsttrainend", res["firsttrainend"]), ("lasttrainend", res["lasttrainend"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_expanding_window_cv = expandcv


def cheatsheet():
    return "expandcv: Expanding-window cross-validation"
