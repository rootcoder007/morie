# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Missing-data imputation for time series.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 2 pp. 44-52
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["tsimpute", "joseph_missing_data_imputation_ts"]

_METHOD = "Missing-data imputation for time series"


def tsimpute(x, method='linear', season=1):
    """Missing-data imputation for time series.

    Missing-data imputation for time series, ch. 2 pp. 44-52.

    ``x`` may contain ``None`` for a gap.  ``method`` is one of the
    book's own options: ``ffill`` (last observation carried forward),
    ``bfill``, ``linear`` interpolation between the flanking
    observations, ``mean`` of the observed values, or ``seasonal``
    (the mean of the same seasonal position).  Leading or trailing
    gaps that a method cannot reach fall back to the series mean, so
    the output never contains a hole.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.tsimpute``.
    method : as documented for the shelf core
        See ``morie.fn._joseph.tsimpute``.
    season : as documented for the shelf core
        See ``morie.fn._joseph.tsimpute``.

    Returns
    -------
    result : RichResult
        Payload keys: nmissing, n, mean.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 2 pp. 44-52
    """
    res = _core.tsimpute(x=x, method=method, season=season)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nmissing", res["nmissing"]), ("n", res["n"]), ("mean", res["mean"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_missing_data_imputation_ts = tsimpute


def cheatsheet():
    return "tsimpute: Missing-data imputation for time series"
