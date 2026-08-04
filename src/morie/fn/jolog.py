# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Log transformation.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 163
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["logtrans", "joseph_log_transform"]

_METHOD = "Log transformation"


def logtrans(x, offset=0.0):
    """Log transformation.

    Log transformation with an optional offset, ch. 6 p. 163.

    The offset is the book's own remedy for series containing zeros.
    ``ratio`` reports the variance-stabilization achieved: the
    coefficient of variation before and after.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.logtrans``.
    offset : as documented for the shelf core
        See ``morie.fn._joseph.logtrans``.

    Returns
    -------
    result : RichResult
        Payload keys: mean, sd, cvbefore, cvafter.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 163
    """
    res = _core.logtrans(x=x, offset=offset)
    return RichResult(
        title=_METHOD,
        summary_lines=[("mean", res["mean"]), ("sd", res["sd"]), ("cvbefore", res["cvbefore"]), ("cvafter", res["cvafter"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_log_transform = logtrans


def cheatsheet():
    return "logtrans: Log transformation"
