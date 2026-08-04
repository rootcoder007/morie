# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seasonal naive baseline.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 8 p. 219
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["seasnaive", "joseph_seasonal_naive"]

_METHOD = "Seasonal naive baseline"


def seasnaive(x, season, horizon):
    """Seasonal naive baseline.

    Seasonal naive baseline, ch. 8 p. 219.

    Each forecast repeats the observation from the same point in the
    previous season -- the benchmark the scaled metrics divide by.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.seasnaive``.
    season : as documented for the shelf core
        See ``morie.fn._joseph.seasnaive``.
    horizon : as documented for the shelf core
        See ``morie.fn._joseph.seasnaive``.

    Returns
    -------
    result : RichResult
        Payload keys: first, last, mean.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 8 p. 219
    """
    res = _core.seasnaive(x=x, season=season, horizon=horizon)
    return RichResult(
        title=_METHOD,
        summary_lines=[("first", res["first"]), ("last", res["last"]), ("mean", res["mean"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_seasonal_naive = seasnaive


def cheatsheet():
    return "seasnaive: Seasonal naive baseline"
