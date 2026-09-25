# morie.fn -- function file (rootcoder007/morie)
"""Epidemic curve (epi curve) construction."""

from . import _array_core as np
from . import _frame_core as pd

from ._containers import DescriptiveResult

import datetime as _dt


def _iter_dates(dates):
    if isinstance(dates, (str, bytes)):
        return [dates]
    if hasattr(dates, "tolist") and not isinstance(dates, np.oarr):
        try:
            vals = dates.tolist()
            return vals if isinstance(vals, list) else [vals]
        except Exception:
            pass
    return list(dates)


def _as_date(v):
    """A calendar date from a datetime64, date, datetime, Timestamp or
    ISO string."""
    if hasattr(v, "item") and not isinstance(v, (_dt.date, str)):
        v = v.item()
    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    if isinstance(v, str):
        return _dt.datetime.fromisoformat(v.strip()).date()
    raise TypeError(f"cannot read {v!r} as a date")


def epidemic_curve(
    dates: np.ndarray,
    bin_width: str = "week",
) -> DescriptiveResult:
    """Construct an epidemic curve by binning event dates.

    Parameters
    ----------
    dates : array-like
        Array of event dates (datetime64, str, or pd.Timestamp).
    bin_width : str, default "week"
        Bin width: "day", "week", or "month".

    Returns
    -------
    DescriptiveResult
        value = DataFrame with columns ['bin_start', 'count'].

    References
    ----------
    CDC (2012). Principles of Epidemiology in Public Health Practice.
    3rd ed. Lesson 6: Investigating an Outbreak.
    """
    freq_map = {"day": 1, "week": 7, "month": None}
    if bin_width not in freq_map:
        raise ValueError(f"bin_width must be one of {list(freq_map.keys())}")
    days = [_as_date(v) for v in _iter_dates(dates)]
    n_total = len(days)
    if not days:
        raise ValueError("dates is empty")

    # Bins are labelled by their first day: the day itself, the Monday a
    # week starts on, or the first of the month. The earlier pandas form
    # used "W-MON", which labels each week by the Monday it ENDS on while
    # the column says bin_start; empty bins between the first and last
    # case are kept as zeros so the curve has no gaps.
    def start_of(d):
        if bin_width == "day":
            return d
        if bin_width == "week":
            return d - _dt.timedelta(days=d.weekday())
        return d.replace(day=1)

    def next_start(d):
        if bin_width == "day":
            return d + _dt.timedelta(days=1)
        if bin_width == "week":
            return d + _dt.timedelta(days=7)
        return (d.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)

    counts = {}
    for d in days:
        b = start_of(d)
        counts[b] = counts.get(b, 0) + 1
    bins = []
    b, last = min(counts), max(counts)
    while b <= last:
        bins.append(b)
        b = next_start(b)
    df = pd.DataFrame({"bin_start": bins,
                       "count": [int(counts.get(b, 0)) for b in bins]})

    return DescriptiveResult(
        name="Epidemic curve",
        value=df,
        extra={"bin_width": bin_width, "n_total": n_total},
    )


epi_c = epidemic_curve


def cheatsheet() -> str:
    return "epidemic_curve({}) -> Epidemic curve (epi curve) construction."


# compact alias per ledger/NAMING.md
epidemiccurve = epidemic_curve
