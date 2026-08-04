# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calendar and time features.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 168
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["calfeat", "joseph_calendar_features"]

_METHOD = "Calendar and time features"


def calfeat(dates):
    """Calendar and time features.

    Calendar / time features, ch. 6 p. 168.

    ``dates`` is a sequence of (year, month, day) triples.  Produces
    the book's time-based features -- year, month, day, day of week,
    day of year, quarter, week-of-year, weekend flag, month-start and
    month-end flags -- plus the cyclic sine/cosine encoding of month
    and day of week that the book recommends so December sits next to
    January.

    The calendar arithmetic is proleptic Gregorian and written out
    here, so both language arms agree without either depending on a
    date library.

    Parameters
    ----------
    dates : as documented for the shelf core
        See ``morie.fn._joseph.calfeat``.

    Returns
    -------
    result : RichResult
        Payload keys: n, nweekend, meandoy, meanmonthsin.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 168
    """
    res = _core.calfeat(dates=dates)
    return RichResult(
        title=_METHOD,
        summary_lines=[("n", res["n"]), ("nweekend", res["nweekend"]), ("meandoy", res["meandoy"]), ("meanmonthsin", res["meanmonthsin"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_calendar_features = calfeat


def cheatsheet():
    return "calfeat: Calendar and time features"
