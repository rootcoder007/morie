# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differencing, ordinary and seasonal.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 pp. 155-158
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["diffser", "joseph_differencing"]

_METHOD = "Differencing, ordinary and seasonal"


def diffser(x, order=1, season=1):
    """Differencing, ordinary and seasonal.

    Differencing, ch. 6 pp. 155-158.

    ``order`` successive lag-``season`` differences.  Seasonal
    differencing is the same operator at lag m.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.diffser``.
    order : as documented for the shelf core
        See ``morie.fn._joseph.diffser``.
    season : as documented for the shelf core
        See ``morie.fn._joseph.diffser``.

    Returns
    -------
    result : RichResult
        Payload keys: mean, var, dropped, n.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 pp. 155-158
    """
    res = _core.diffser(x=x, order=order, season=season)
    return RichResult(
        title=_METHOD,
        summary_lines=[("mean", res["mean"]), ("var", res["var"]), ("dropped", res["dropped"]), ("n", res["n"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_differencing = diffser


def cheatsheet():
    return "diffser: Differencing, ordinary and seasonal"
