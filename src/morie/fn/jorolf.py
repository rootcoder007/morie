# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rolling-window features.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 176
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["rollfeat", "joseph_rolling_window_feature"]

_METHOD = "Rolling-window features"


def rollfeat(x, window, minperiods=None):
    """Rolling-window features.

    Rolling-window features, ch. 6 p. 176.

    Trailing mean, standard deviation, minimum and maximum over the
    last ``window`` observations, computed only where at least
    ``minperiods`` observations are available (default: the full
    window, so no partial window ever leaks a shorter average).

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.rollfeat``.
    window : as documented for the shelf core
        See ``morie.fn._joseph.rollfeat``.
    minperiods : as documented for the shelf core
        See ``morie.fn._joseph.rollfeat``.

    Returns
    -------
    result : RichResult
        Payload keys: nrows, lastmean, meanofmeans.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 176
    """
    res = _core.rollfeat(x=x, window=window, minperiods=minperiods)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nrows", res["nrows"]), ("lastmean", res["lastmean"]), ("meanofmeans", res["meanofmeans"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_rolling_window_feature = rollfeat


def cheatsheet():
    return "rollfeat: Rolling-window features"
