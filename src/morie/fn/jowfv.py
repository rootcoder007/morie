# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Walk-forward validation.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 5 p. 126
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["walkfwd", "joseph_walk_forward_validation"]

_METHOD = "Walk-forward validation"


def walkfwd(y, yhat, initial, testsize, step=None):
    """Walk-forward validation.

    Walk-forward validation, ch. 5 p. 126.

    Scores an already-produced forecast series fold by fold on an
    expanding-window layout, and reports the fold RMSEs plus their mean
    and spread -- which is the number the book actually reads off a
    walk-forward run.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.walkfwd``.
    yhat : as documented for the shelf core
        See ``morie.fn._joseph.walkfwd``.
    initial : as documented for the shelf core
        See ``morie.fn._joseph.walkfwd``.
    testsize : as documented for the shelf core
        See ``morie.fn._joseph.walkfwd``.
    step : as documented for the shelf core
        See ``morie.fn._joseph.walkfwd``.

    Returns
    -------
    result : RichResult
        Payload keys: rmse, sd, best, worst.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 5 p. 126
    """
    res = _core.walkfwd(y=y, yhat=yhat, initial=initial, testsize=testsize, step=step)
    return RichResult(
        title=_METHOD,
        summary_lines=[("rmse", res["rmse"]), ("sd", res["sd"]), ("best", res["best"]), ("worst", res["worst"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_walk_forward_validation = walkfwd


def cheatsheet():
    return "walkfwd: Walk-forward validation"
