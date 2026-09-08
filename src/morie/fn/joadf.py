# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Augmented Dickey-Fuller unit-root test.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 149
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["adfur", "joseph_adf_unit_root_test"]

_METHOD = "Augmented Dickey-Fuller unit-root test"


def adfur(x, lags=1):
    """Augmented Dickey-Fuller unit-root test.

    Augmented Dickey-Fuller unit-root test, ch. 6 p. 149.

    Regresses ``diff(x)_t`` on a constant, ``x_{t-1}`` and ``lags``
    lagged differences; the statistic is the t-ratio on ``x_{t-1}``.
    The null is a unit root, so a statistic BELOW the critical value
    rejects non-stationarity -- which is the direction the book uses on
    p. 149.

    Critical values are MacKinnon's (1991) response surface for the
    constant-only regression, ``tau = b0 + b1/n + b2/n^2``; they are
    returned rather than a p-value, because interpolating a p-value
    would need a table the book does not print.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.adfur``.
    lags : as documented for the shelf core
        See ``morie.fn._joseph.adfur``.

    Returns
    -------
    result : RichResult
        Payload keys: stat, se, crit5, stationary5.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 149
    """
    res = _core.adfur(x=x, lags=lags)
    return RichResult(
        title=_METHOD,
        summary_lines=[("stat", res["stat"]), ("se", res["se"]), ("crit5", res["crit5"]), ("stationary5", res["stationary5"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_adf_unit_root_test = adfur


def cheatsheet():
    return "adfur: Augmented Dickey-Fuller unit-root test"
